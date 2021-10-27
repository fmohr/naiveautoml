package naiveautoml;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Map.Entry;
import java.util.Objects;
import java.util.Random;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

import org.api4.java.algorithm.Timeout;
import org.api4.java.common.attributedobjects.ObjectEvaluationFailedException;

import com.google.common.util.concurrent.AtomicDouble;

import ai.libs.jaicore.components.api.IComponent;
import ai.libs.jaicore.components.api.IComponentInstance;
import ai.libs.jaicore.components.api.IComponentRepository;
import ai.libs.jaicore.components.exceptions.ComponentInstantiationFailedException;
import ai.libs.jaicore.components.model.ComponentUtil;
import ai.libs.mlplan.weka.weka.WekaPipelineFactory;
import weka.classifiers.Classifier;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.Kernel;
import weka.core.OptionHandler;

public class RandomParameterOptimizer {

	private final Random random;
	private final WekaPipelineFactory factory = new WekaPipelineFactory();

	public RandomParameterOptimizer(final Random random) {
		super();
		this.random = random;
		//		this.factory.setLoggerName(LoggerUtil.LOGGER_NAME_TESTEDALGORITHM);
	}

	public Collection<Entry<PipelineDescription, EvaluationReports>> optimize(final PipelineDescription basis, final Phase1Results phase1Results, final IComponentRepository components, final RacePool pool, final Timeout to) throws InterruptedException, ObjectEvaluationFailedException, ExecutionException, ComponentInstantiationFailedException {
		Objects.requireNonNull(basis, "The basis candidate must not be null");
		List<Integer> attributesTmp = null;
		if (basis.getAttributes() != null) {
			attributesTmp = new ArrayList<>(basis.getAttributes());
		}
		List<Integer> attributes = attributesTmp;
		String baseLearner = basis.getBaseLearner();
		String metaLearner = basis.getMetaLearner();

		String[] parts = baseLearner.split("-");
		String baseLearnerName = parts[0];
		String kernelName = parts.length > 1 ? parts[1] : null;
		List<String> blackListedParams = Util.blackListedParameters(baseLearner);

		IComponent baseLearnerComponent = components.getComponent(baseLearnerName);
		int maxNumberOfPossibleConfigurations = 1;//ComponentUtil.getNumberOfParametrizations(baseLearnerComponent);
		int MAX_EVALUATIONS = 1000;
		int MAX_EVALUATIONS_WITHOUT_IMPROVEMENT = 100;
		final boolean exhaustible = maxNumberOfPossibleConfigurations < 2 * MAX_EVALUATIONS;

		ConcurrentMap<PipelineDescription, EvaluationReports> score = new ConcurrentHashMap<>();
		final long deadline = System.currentTimeMillis() + to.milliseconds() - 5000;
		ExecutorService tp = Executors.newFixedThreadPool(8);
		AtomicInteger numEvaluations = new AtomicInteger();
		AtomicDouble bestScore = new AtomicDouble(1.0);
		AtomicInteger numEvaluationsWithoutImprovement = new AtomicInteger(0);


		BlockingQueue<IComponentInstance> candidates = new LinkedBlockingQueue<>();
		if (exhaustible) {
			System.out.println("EXHAUSTIBLE");
			//			candidates.addAll(ComponentUtil.getAllInstantiations(baseLearnerComponent));
		}

		for (int i = 0; i < 8; i++) {
			tp.submit(() -> {
				int consecutiveSkips = 0;
				while (System.currentTimeMillis() < deadline && numEvaluations.get() < MAX_EVALUATIONS && consecutiveSkips < 10) {
					try {
						IComponentInstance baseLearnerVariant;
						if (exhaustible) {
							if (candidates.isEmpty()) {
								return;
							}
							baseLearnerVariant = candidates.poll();
						}
						else {
							baseLearnerVariant = ComponentUtil.getRandomParameterizationOfComponent(baseLearnerComponent, this.random);
						}
						Classifier classifier = this.factory.getComponentInstantiation(baseLearnerVariant).getClassifier();
						if (classifier instanceof SMO) {
							if (kernelName != null) {
								((SMO) classifier).setKernel(Kernel.forName(kernelName, null));
								((SMO) classifier).setOptions(new String[] {"-C", parts[2]});
							}
							System.out.println(Arrays.toString(((SMO) classifier).getOptions()));
						}
						String[] options = ((OptionHandler)classifier).getOptions();
						final PipelineDescription variant = new PipelineDescription(attributes, baseLearner, options, metaLearner, null);
						if (!score.containsKey(variant)) {
							consecutiveSkips = 0;
							Future<EvaluationReports> reports = pool.submitCandidate(variant);
							EvaluationReports reportsRetr = reports.get();
							score.put(variant, reportsRetr != null ? reportsRetr : new EvaluationReports());
							double bestScoreInThisEvaluation = reportsRetr != null ? reportsRetr.getBestAverageScoreSeen() : 1.0;
							synchronized (bestScore) {
								numEvaluations.incrementAndGet();
								if (bestScoreInThisEvaluation < bestScore.get()) {
									bestScore.set(bestScoreInThisEvaluation);
									numEvaluationsWithoutImprovement.set(0);
								}
								else{
									int numEvalsWithoutImprovement = numEvaluationsWithoutImprovement.incrementAndGet();
									if (!exhaustible && numEvalsWithoutImprovement >= MAX_EVALUATIONS_WITHOUT_IMPROVEMENT) {
										System.out.println("No improvement in " + numEvalsWithoutImprovement + "steps. Stopping optimization.");
										return;
									}
								}


							}
						}
						else {
							System.out.println("SKIPPING. Already in cache with score " + score.get(variant) + "!");
							consecutiveSkips ++;
						}
					}
					catch (Exception e) {
						e.printStackTrace();
					}
				}
			});
		}
		tp.shutdown();
		tp.awaitTermination(1, TimeUnit.DAYS);
		return score.entrySet().stream().sorted((e1, e2) -> Double.compare(e1.getValue().getBestAverageScoreSeen(), e2.getValue().getBestAverageScoreSeen())).limit(10).collect(Collectors.toList());
	}
}
