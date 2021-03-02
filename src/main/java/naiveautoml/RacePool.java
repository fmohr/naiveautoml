package naiveautoml;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.api4.java.ai.ml.core.dataset.splitter.SplitFailedException;
import org.api4.java.algorithm.Timeout;
import org.api4.java.common.control.ILoggingCustomizable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.util.concurrent.AtomicDouble;

import ai.libs.jaicore.logging.LoggerUtil;
import ai.libs.jaicore.ml.classification.loss.dataset.EClassificationPerformanceMeasure;
import ai.libs.jaicore.ml.core.evaluation.evaluator.FixedSplitClassifierEvaluator;
import ai.libs.jaicore.ml.core.filter.SplitterUtil;
import ai.libs.jaicore.ml.weka.classification.learner.IWekaClassifier;
import ai.libs.jaicore.ml.weka.dataset.IWekaInstances;
import ai.libs.jaicore.timing.TimedComputation;

public class RacePool implements ILoggingCustomizable {

	private Logger logger = LoggerFactory.getLogger(RacePool.class);

	public static final Timeout TIMEOUT_FOR_SINGLE_EXECUTIONS = new Timeout(60, TimeUnit.SECONDS);
	public static final double TOLERANCE_IN_LEARNINGCURVEESTIMATION = 0.05;
	public static final double POSSIBLE_IMPROVEMENT_BY_LEARNEROPTIMIZATION = 0.05;

	private final IWekaInstances instances;
	private final List<Integer> anchorPoints = new ArrayList<>();
	private int maxAnchorPoint;
	private final AtomicDouble bestObservedScore = new AtomicDouble(1);
	private final Map<Long, Double> bestSolutionTimestamps = new HashMap<>();
	private final ExecutorService pool = Executors.newFixedThreadPool(1);

	private boolean initialized = false;

	private final ConcurrentMap<Integer, List<List<IWekaInstances>>> splitsPerAnchorPoint = new ConcurrentHashMap<>();
	private final ConcurrentMap<PipelineDescription, EvaluationReports> evaluationReports = new ConcurrentHashMap<>();

	private final int numIterations;

	private long deadline;

	public RacePool(final RacePool earlierPool) {
		this(earlierPool, earlierPool.anchorPoints);
	}

	public RacePool(final RacePool earlierPool, final List<Integer> anchorPoints) {
		this(earlierPool.instances, anchorPoints);
		this.splitsPerAnchorPoint.putAll(earlierPool.splitsPerAnchorPoint);
		this.evaluationReports.putAll(earlierPool.evaluationReports);
		this.bestSolutionTimestamps.putAll(earlierPool.getBestSolutionTimestamps());
		this.bestObservedScore.set(earlierPool.bestObservedScore.get());
		System.out.println("Initialized race pool with " + this.evaluationReports.size() + " reports in cache.");
	}

	public RacePool(final IWekaInstances instances) {
		this(instances, Arrays.asList((int) Math.round(instances.size() * 0.7)), 5); // conduct 5-MCCV
	}

	public RacePool(final IWekaInstances instances, final List<Integer> anchorPoints) {
		this(instances, anchorPoints, 2);
	}

	public RacePool(final IWekaInstances instances, final List<Integer> anchorPoints, final int numIterations) {
		super();
		Objects.requireNonNull(anchorPoints);
		if (anchorPoints.isEmpty()) {
			throw new IllegalArgumentException("No anchor points defined (list of anchors is empty)!");
		}
		this.instances = instances;
		int maxSize = 0;
		if (anchorPoints.size() == 1) {
			this.anchorPoints.addAll(anchorPoints);
		} else {
			for (int size : anchorPoints) {
				if (size <= this.instances.size() * .7) {
					this.anchorPoints.add(size);
					maxSize = Math.max(maxSize, size);
				}
			}
		}
		if (this.anchorPoints.isEmpty()) {
			throw new IllegalStateException("No anchor points added.");
		}
		this.maxAnchorPoint = maxSize;
		this.numIterations = numIterations;
	}

	public boolean isEvidencePresentThatPipelineEvaluationWillFail(final PipelineDescription desc, final int numInstances) {
		String baseLearner = desc.getBaseLearner();
		for (Entry<PipelineDescription, EvaluationReports> entry : this.evaluationReports.entrySet()) {
			PipelineDescription challenger = entry.getKey();

			/* check whether base learner is identical */
			boolean sameBaseLearner = challenger.getBaseLearner().equals(baseLearner);
			if (sameBaseLearner) {
				if ((challenger.getBaseLearnerParams() == null) == (desc.getBaseLearnerParams() == null)) {
					if (challenger.getBaseLearnerParams() != null && !challenger.getBaseLearnerParams().equals(desc.getBaseLearnerParams())) {
						sameBaseLearner = false;
					}
				} else {
					sameBaseLearner = false;
				}
			}

			if (sameBaseLearner) {
				if (entry.getValue().hasFailedReportForSmallerSize(numInstances)) {
					return true;
				}
			}
		}
		return false;
	}

	private synchronized void initialize() {

		if (this.initialized) {
			return;
		}
		Objects.requireNonNull(this.anchorPoints, "Anchor points is null!");
		if (this.anchorPoints.isEmpty()) {
			throw new IllegalStateException("List of anchor points is empty!");
		}
		this.logger.info("Initializing race pool. Anchor points are {}", this.anchorPoints);

		/* compute all possible folds over which we split here */
		for (int size : this.anchorPoints) {
			if (size >= this.instances.size() * 0.7) {
				size = (int) Math.ceil(this.instances.size() * 0.7);
			}
			if (this.splitsPerAnchorPoint.containsKey(size)) {
				this.logger.debug("Skipping folds for size {} since they are already in cache.", size);
				continue;
			}
			List<List<IWekaInstances>> splits = new ArrayList<>();
			int maximumPossibleNumberOfValidationInstances = this.instances.size() - size;
			int numberOfTestInstances = Math.min(maximumPossibleNumberOfValidationInstances, 1000);
			double portionForFolds = (size + numberOfTestInstances) * 1.0 / this.instances.size();
			double trainFoldPortion = size * 1.0 / (size + numberOfTestInstances);
			this.logger.info("Creating train and test folds of sizes {} (train) and {} (test)", size, numberOfTestInstances);
			for (int seed = 0; seed < this.numIterations; seed++) {
				try {
					IWekaInstances coreFold = portionForFolds < 1 ? SplitterUtil.getLabelStratifiedTrainTestSplit(this.instances, seed, portionForFolds).get(0) : this.instances;
					splits.add(SplitterUtil.getLabelStratifiedTrainTestSplit(coreFold, seed, trainFoldPortion));
				} catch (SplitFailedException | InterruptedException e) {
					throw new RuntimeException(e);
				}
			}
			this.splitsPerAnchorPoint.put(size, splits);
			if (size >= this.instances.size() * 0.7) {
				this.maxAnchorPoint = size;
				break;
			}
		}
		this.initialized = true;
	}

	public Future<EvaluationReports> submitCandidate(final PipelineDescription pl) {
		this.initialize();
		return this.pool.submit(() -> {
			try {

				if (this.deadline > 0 && System.currentTimeMillis() > this.deadline) {
					System.out.println("Deadline already hit, not executing anymore.");
					this.logger.debug("Deadline already hit, not executing anymore.");
					return new EvaluationReports();
				}

				Objects.requireNonNull(this.anchorPoints, "Anchor points must not be null!");
				if (this.anchorPoints.isEmpty()) {
					throw new IllegalStateException("Set of anchor points is empty!");
				}

				this.logger.info("Considering evaluation of candidate pipeline {}. Anchor points would be {}", pl, this.anchorPoints);
				Map<Integer, Double> learningCurve = new HashMap<>();
				IWekaClassifier classifier = Util.getClassifierFromDescription(pl);
				String baseLearnerClass = pl.getBaseLearner();
				EvaluationReports evaluationReportsForThisLearner = new EvaluationReports();
				this.evaluationReports.put(pl, evaluationReportsForThisLearner);
				for (int size : this.anchorPoints) {

					/* ignore if results are already cached */
					if (this.evaluationReports.containsKey(pl)) {
						EvaluationReports reports = this.evaluationReports.get(pl).getReportsForTrainSize(size);
						if (!reports.isEmpty()) {
							this.logger.info("Not re-evaluating {} on size {} since {} results are already cached.", pl, size, reports.size());
							evaluationReportsForThisLearner.addAll(reports);
							continue;
						}
					}

					/* early stop */
					if (this.isEvidencePresentThatPipelineEvaluationWillFail(pl, size)) {
						this.logger.info("There is evidence that pipeline {} will not be executed successfully on size {}. Canceling execution.", pl, size);
						break;
					}

					/* gather evaluations */
					else {

						if (this.splitsPerAnchorPoint.get(size) == null) {
							throw new IllegalStateException("No split present for size " + size + ". We have these splits: " + this.splitsPerAnchorPoint.keySet());
						}

						DescriptiveStatistics scoreStats = new DescriptiveStatistics();
						double bestExpectedPossibleScore = 1.0;
						this.logger.debug("Starting evaluation of pipeline {}", pl);
						for (int seed = 0; seed < this.numIterations; seed++) {
							FixedSplitClassifierEvaluator evaluator = new FixedSplitClassifierEvaluator(NaiveAutoML.reduceDimensionality(this.splitsPerAnchorPoint.get(size).get(seed).get(0), pl.getAttributes()),
									NaiveAutoML.reduceDimensionality(this.splitsPerAnchorPoint.get(size).get(seed).get(1), pl.getAttributes()), EClassificationPerformanceMeasure.ERRORRATE);
							long timestampStartComputation = System.currentTimeMillis();
							boolean failed = false;
							double score = Double.NaN;
							try {
								score = TimedComputation.compute(() -> evaluator.evaluate(classifier), TIMEOUT_FOR_SINGLE_EXECUTIONS, "timeout of base learner eval");
							} catch (Exception e) {
								failed = true;
								this.logger.warn("Failed: \n{}", LoggerUtil.getExceptionInfo(e));
							}
							int runtime = (int) (System.currentTimeMillis() - timestampStartComputation);
							evaluationReportsForThisLearner.add(new EvaluationReport(size, score, runtime, failed));
							scoreStats.addValue(score);

							/* if this execution failed, cancel job */
							if (failed) {
								this.logger.debug("Canceling further executions of candidate {} due to a fail.", pl);
								break;
							}

							/* obtain projection for the learning curve of this candidate */
							if (size < this.maxAnchorPoint) {
								learningCurve.put(size, scoreStats.getMean());
								bestExpectedPossibleScore = Util.getBestExpectedScoreAtPointUnderConvexityAssumption(learningCurve, this.instances.size(), 0.01 / 1000) - TOLERANCE_IN_LEARNINGCURVEESTIMATION;
								if (bestExpectedPossibleScore > score) {
									throw new IllegalStateException("Learning curve prognostics " + bestExpectedPossibleScore + " must not be worse than currently best value in curve: " + learningCurve);
								}
								if (bestExpectedPossibleScore - POSSIBLE_IMPROVEMENT_BY_LEARNEROPTIMIZATION >= this.bestObservedScore.get()) {
									this.logger.info("CUTTING learning curve process of {} + {} after {} itartions on size {}. Best anticipated possible score is {}, which is worse than the currently best known score.", baseLearnerClass,
											pl.getMetaLearner(), seed + 1, size, bestExpectedPossibleScore - POSSIBLE_IMPROVEMENT_BY_LEARNEROPTIMIZATION);
									break;
								}
							}
						}

						/* summary of this candidate */
						double meanScore = scoreStats.getMean();
						if (meanScore < this.bestObservedScore.doubleValue()) {
							this.bestObservedScore.set(meanScore);
							this.logger.info("NEW BEST SCORE {}", meanScore);
							this.bestSolutionTimestamps.put(System.currentTimeMillis(), meanScore);
						}
						this.logger.info("Observed scores for {} ({}) + {} on {} train points: {} ({}). Best possible score {}", baseLearnerClass, pl.isCheapVersion() ? "cheap" : "default", pl.getMetaLearner(), size, meanScore,
								Arrays.toString(scoreStats.getValues()), bestExpectedPossibleScore);
					}
				}

				/* if there is a cheap variant of this pipeline in the cache, remove it */
				if (!pl.isCheapVersion()) {
					this.logger.debug("Removing cheap version of pipeline if existent. Current entries: {}", this.evaluationReports.size());
					EvaluationReports r = this.evaluationReports.remove(pl.getCheapVersion());
					this.logger.debug("Did {}remove cheap version of pipeline if existent. Current entries: {}", r == null ? "not " : "", this.evaluationReports.size());
				}
				this.logger.info("Added run report for pipeline {}. There are now {} run reports.", pl, this.evaluationReports.size());
				return evaluationReportsForThisLearner;
			} catch (Throwable e) {
				e.printStackTrace();
				throw new RuntimeException(e);
			}
		});
	}

	public void closeAndAwaitTermination() throws InterruptedException {
		this.pool.shutdown();
		this.pool.awaitTermination(1, TimeUnit.DAYS);
	}

	public List<Entry<PipelineDescription, EvaluationReports>> getBestCandidates(final int n) {
		return this.evaluationReports.entrySet().stream().sorted((e1, e2) -> Double.compare(e1.getValue().getBestAverageScoreSeen(), e2.getValue().getBestAverageScoreSeen())).limit(n).collect(Collectors.toList());
	}

	public List<Entry<PipelineDescription, EvaluationReports>> getBestPortion(final double portion) {
		int n = (int) Math.ceil(this.evaluationReports.size() * portion);
		this.logger.info("Computing best {}-portion of candidates. There are {} candidates in total. Computing {} candidates.", portion, this.evaluationReports.size(), n);
		return this.getBestCandidates(n);
	}

	public List<Entry<PipelineDescription, EvaluationReports>> getCandidatesThatStillCanCompete(final int targetSize) {
		double bestScore = this.evaluationReports.values().stream().filter(v -> !Double.isNaN(v.getBestAverageScoreSeen())).mapToDouble(v -> v.getBestAverageScoreSeen()).min().getAsDouble();
		return this.evaluationReports.entrySet().stream().filter(e -> {
			if (e.getValue().isEmpty() || Double.isNaN(e.getValue().getBestAverageScoreSeen())) {
				return false;
			}
			double bestExpectedScore = Util.getBestExpectedScoreAtPointUnderConvexityAssumption(e.getValue().getLearningCurveAsMap(), targetSize, 0.01 / 1000);
			this.logger.debug("Best expected score for candidate {} is {}", e.getKey(), bestExpectedScore);
			return bestExpectedScore <= bestScore + TOLERANCE_IN_LEARNINGCURVEESTIMATION + POSSIBLE_IMPROVEMENT_BY_LEARNEROPTIMIZATION;
		}).sorted((e1, e2) -> Double.compare(e1.getValue().getBestAverageScoreSeen(), e2.getValue().getBestAverageScoreSeen())).collect(Collectors.toList());
	}

	public EvaluationReports getReports(final PipelineDescription pl) {
		return this.evaluationReports.get(pl);
	}

	public void injectResults(final Map<PipelineDescription, EvaluationReports> reports) {
		this.evaluationReports.putAll(reports);
	}

	public void merge(final RacePool pool) {
		this.bestSolutionTimestamps.putAll(pool.getBestSolutionTimestamps());
		this.injectResults(pool.getBestCandidates(100000000));
	}

	public void injectResults(final Collection<Entry<PipelineDescription, EvaluationReports>> reports) {
		reports.forEach(e -> {
			this.evaluationReports.put(e.getKey(), e.getValue());
			if (e.getValue().getBestAverageScoreSeen() < this.bestObservedScore.get()) {
				this.bestObservedScore.set(e.getValue().getBestAverageScoreSeen());
			}
		});
	}

	public long getDeadline() {
		return this.deadline;
	}

	public void setDeadline(final long deadline) {
		this.deadline = deadline;
	}

	public Map<Long, Double> getBestSolutionTimestamps() {
		return this.bestSolutionTimestamps;
	}

	@Override
	public String getLoggerName() {
		return this.logger.getName();
	}

	@Override
	public void setLoggerName(final String name) {
		this.logger = LoggerFactory.getLogger(name);
	}
}