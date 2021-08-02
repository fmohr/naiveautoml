package naiveautoml;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;
import java.util.Random;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledInstance;
import org.api4.java.algorithm.Timeout;
import org.api4.java.common.attributedobjects.ObjectEvaluationFailedException;
import org.api4.java.common.control.ILoggingCustomizable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.util.concurrent.AtomicDouble;

import ai.libs.jaicore.components.api.IComponentRepository;
import ai.libs.jaicore.components.exceptions.ComponentInstantiationFailedException;
import ai.libs.jaicore.logging.LoggerUtil;
import ai.libs.jaicore.ml.classification.loss.dataset.EClassificationPerformanceMeasure;
import ai.libs.jaicore.ml.core.dataset.serialization.OpenMLDatasetReader;
import ai.libs.jaicore.ml.core.evaluation.evaluator.FixedSplitClassifierEvaluator;
import ai.libs.jaicore.ml.core.evaluation.evaluator.MonteCarloCrossValidationEvaluator;
import ai.libs.jaicore.ml.core.filter.SplitterUtil;
import ai.libs.jaicore.ml.weka.WekaUtil;
import ai.libs.jaicore.ml.weka.classification.learner.IWekaClassifier;
import ai.libs.jaicore.ml.weka.classification.learner.WekaClassifier;
import ai.libs.jaicore.ml.weka.dataset.IWekaInstances;
import ai.libs.jaicore.ml.weka.dataset.WekaInstances;
import ai.libs.mlplan.weka.MLPlanWekaBuilder;
import naiveautoml.ga.FeatureSetOptimizer;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.Ranker;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.SingleClassifierEnhancer;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class NaiveAutoML implements ILoggingCustomizable {

	public static final int NUM_NOIMPROVEITERATIONS_FILTERING = 100;
	public static final List<Integer> ANCHOR_POINTS_PHASE1 = Arrays.asList(100, 200, 500);
	public static final List<Integer> ANCHOR_POINTS_PHASE2 = Arrays.asList(100, 200, 500, 1000, 5000, Integer.MAX_VALUE);
	public static final List<Integer> ANCHOR_POINTS_PHASE3 = Arrays.asList(100, 200, 500, 1000, 5000, Integer.MAX_VALUE);
	public static final int NUM_CPUS = 1;

	public static final Timeout TIMEOUT = new Timeout(1, TimeUnit.HOURS);
	public static final Timeout TIMEOUT_STAGE_META = new Timeout(1, TimeUnit.SECONDS);
	public static final Timeout TIMEOUT_STAGE_TUNING = new Timeout(10, TimeUnit.SECONDS);
	public static final Timeout TIMEOUT_STAGE_TUNING_PER_TEMPLATE = new Timeout(1, TimeUnit.MINUTES);
	public static final Timeout TIMEOUT_STAGE_VALIDATION = new Timeout(5, TimeUnit.MINUTES);

	public static final double VALIDATION_FOLD_SIZE = 0.2; // only applied if validation is activated

	private final boolean iterativeEvaluation;

	private final boolean enable_metalearners;
	private final boolean enable_feature_projection;
	private final boolean enable_wrapping;
	private final boolean enable_parametertuning;
	private final boolean enable_consolidation;
	private final boolean enable_final_evaluation;

	List<Integer> attributeCandidates = new ArrayList<>();
	List<Double> pvectorForAttributeCandidates = new ArrayList<>();
	Map<Integer, Integer> attributeVotes = new HashMap<>();

	private Logger logger = LoggerFactory.getLogger(NaiveAutoML.class);
	private long deadline;

	private Phase1Results phase1Results;
	private Phase2Results phase2Results;

	public NaiveAutoML(final boolean enable_feature_projection, final boolean enable_consolidation, final boolean enable_metalearners, final boolean enable_wrapping, final boolean enable_parametertuning, final boolean enable_final_evaluation, final boolean iterativeEvaluation) {
		super();
		this.enable_feature_projection = enable_feature_projection;
		this.enable_consolidation = enable_consolidation;
		this.enable_metalearners = enable_metalearners;
		this.enable_wrapping = enable_wrapping;
		this.enable_parametertuning = enable_parametertuning;
		this.enable_final_evaluation = enable_final_evaluation;
		this.iterativeEvaluation = iterativeEvaluation;
	}

	public FinalResult findBestPipelineDescription(final IWekaInstances originalData, final Timeout to) throws Exception {


		this.logger.info("Running Naive AutoML with the following configuration:\n\tTimeout: {}s\n\tFiltering: {}\n\tWrapping: {}\n\tMeta-Learners: {}\n\tParameter Tuning: {}\n\tValidation Phase: {}", TIMEOUT.seconds(), this.enable_feature_projection, this.enable_wrapping, this.enable_metalearners, this.enable_parametertuning, this.enable_final_evaluation);

		this.deadline = System.currentTimeMillis() + TIMEOUT.milliseconds();

		List<IWekaInstances> trainValidationSplit = null;
		IWekaInstances instancesForOptimization;
		if (this.enable_final_evaluation) {
			trainValidationSplit = SplitterUtil.getLabelStratifiedTrainTestSplit(originalData, 0, 1 - VALIDATION_FOLD_SIZE);
			instancesForOptimization = trainValidationSplit.get(0);
			this.logger.info("Separated data into a train fold of size {} and a validation fold of size {}.", instancesForOptimization.size(), trainValidationSplit.get(1).size());
		}
		else {
			instancesForOptimization = originalData;
		}

		Random random = new Random(0);

		Map<String, Integer> runtimesPerStage = new HashMap<>();
		long start;
		long end;

		/* PHASE 1 (probing) */
		start = System.currentTimeMillis();
		this.phase1Results = this.getProbingResults(instancesForOptimization);
		end = System.currentTimeMillis();
		runtimesPerStage.put("probing", (int)(end - start));

		this.logger.info("Finished phase 1. Stepping over to phase 2.");

		/* PHASE 2 (filtering) */
		start = end;
		if (start < this.deadline) {
			Phase1Results filteringResultPool = this.doFiltering(instancesForOptimization, random);
			this.phase1Results.getRacepool().merge(filteringResultPool.getRacepool());
		}
		else {
			this.logger.info("Skipping filtering since deadline has been reached.");
		}
		end = System.currentTimeMillis();
		runtimesPerStage.put("filtering", (int)(end - start));

		/* PHASE 2 (meta models) */
		start = end;
		this.phase2Results = this.doMetaLearningStage(instancesForOptimization, this.phase1Results);
		end = System.currentTimeMillis();
		runtimesPerStage.put("meta", (int)(end - start));

		this.logger.info("Finished phase meta-learner phase. Stepping over to wrapper phase.");

		/* PHASE 3 (wrapping) */
		start = end;
		RacePool resultsAfterWrapping = this.applyWrapping(instancesForOptimization, random, this.phase1Results.getFilteringResult(), this.phase2Results);
		end = System.currentTimeMillis();
		runtimesPerStage.put("wrapping", (int)(end - start));

		this.logger.info("Wrapper stage finished. Going to Parameter Tuning Stage.");

		/* PHASE 4 (tuning) */
		start = end;
		Phase3Results phase3Results = this.getResultsAfterTuning(instancesForOptimization, resultsAfterWrapping, random);
		end = System.currentTimeMillis();
		runtimesPerStage.put("tuning", (int)(end - start));


		this.logger.info("Finished tuning phase. Stepping over to validation phase.");

		/* PHASE 5 (model selection) */
		start = end;
		FinalResult finalResult = this.doFinalValidation(trainValidationSplit, phase3Results.getPool().getCandidatesThatStillCanCompete(originalData.size()));
		end = System.currentTimeMillis();
		runtimesPerStage.put("validation", (int)(end - start));

		return new FinalResult(finalResult.getFinalChoice(), finalResult.getEstimatedOutOfSampleError(), runtimesPerStage, phase3Results.getPool().getBestSolutionTimestamps());
	}

	public RacePool applyWrapping(final IWekaInstances data, final Random random, final FilteringResult filteringResult, final Phase2Results results) throws InterruptedException {
		RacePool syntheticPool = new RacePool(results.getPool());
		syntheticPool.closeAndAwaitTermination(); // work-around

		if (this.enable_wrapping) {
			Objects.requireNonNull(filteringResult);
			Collection<Entry<PipelineDescription, EvaluationReports>> competitiveCandidates = results.getPool().getCandidatesThatStillCanCompete(data.size());
			ExecutorService pool = Executors.newFixedThreadPool(8);
			this.logger.info("Applying wrapping to {} competitive candidates.", competitiveCandidates.size());
			for (Entry<PipelineDescription, EvaluationReports> entry : competitiveCandidates) {
				pool.submit(() -> {
					this.logger.info("Executing wrapper for {}", entry.getKey());
					FeatureSetOptimizer opt = new FeatureSetOptimizer(data, filteringResult);
					try {
						syntheticPool.injectResults(opt.wrapperOptimization(entry.getKey(), random, new Timeout(3, TimeUnit.MINUTES)));
					} catch (InterruptedException e) {
						e.printStackTrace();
					}
					this.logger.info("Wrapper finished for {}", entry.getKey());
				});
				this.logger.info("Enqueued wrapping tasks for pl {}", entry.getKey());
			}
			pool.shutdown();
			pool.awaitTermination(1, TimeUnit.HOURS);
		}
		else {
			this.logger.info("Wrapping disabled. Keeping same candidates.");
		}
		return syntheticPool;
	}

	public FilteringResult getFilterRankings(final IWekaInstances data, final Random random) throws Exception {

		final IWekaClassifier pilot = Util.getCheapDefaultClassifier(RandomForest.class.getName());
		final IWekaClassifier majorityClassifier = Util.getDefaultClassifier(ZeroR.class.getName());
		IWekaInstances smallInstancesWeka;
		if (data.size() > 1000) { // reduce dataset to 1000 instances
			this.logger.info("Reducing number of instances.");
			smallInstancesWeka = SplitterUtil.getLabelStratifiedTrainTestSplit(data, random.nextInt(), 1000.0 / data.size()).get(0);
		} else {
			smallInstancesWeka = data;
		}
		MonteCarloCrossValidationEvaluator mccvSmallData = new MonteCarloCrossValidationEvaluator(smallInstancesWeka, 3, 0.7, random);

		final Map<String, int[]> rankings = new HashMap<>();
		ExecutorService pool = Executors.newFixedThreadPool(NUM_CPUS);
		final Map<String, Integer> locallyBestNumberOfFeaturesPerPreprocessor = new HashMap<>();
		final Map<String, List<Double>> scoreListPerPreprocessor = new HashMap<>();
		final Map<String, Double> scorePerPreprocessor = new HashMap<>();
		long timestartStartPreprocessing = System.currentTimeMillis();
		this.logger.info("Determining merits of attributes using a dataset of size {} x {}", smallInstancesWeka.size(), smallInstancesWeka.getNumAttributes());
		for (String preprocessor : WekaUtil.getFeatureEvaluators()) {
			if (preprocessor.contains("PrincipalComponents")) {
				continue;
			}
			pool.submit(new Runnable() {
				@Override
				public void run() {

					try {
						NaiveAutoML.this.getPreprocessor(preprocessor, smallInstancesWeka, rankings).run();

						List<Double> scoreHistory = new ArrayList<>();
						int[] ranking = rankings.get(preprocessor);
						NaiveAutoML.this.logger.info("Checking best montone subset using {}", preprocessor);
						int bestN = -1;
						double bestScore = 1;
						int noImprovement = 0;
						for (int n = 1; n <= ranking.length; n++) {
							WekaInstances reducedInstances = reduceDimensionality(smallInstancesWeka, ranking, n);
							double scoreForThisAttributeSet = new MonteCarloCrossValidationEvaluator(reducedInstances, 5, 0.7, random).evaluate(Util.getCheapDefaultClassifier(RandomForest.class.getName()));
							scoreHistory.add(scoreForThisAttributeSet);
							if (scoreForThisAttributeSet < bestScore) {
								bestN = n;
								bestScore = scoreForThisAttributeSet;
								noImprovement = 0;
							} else {
								noImprovement++;
							}
							NaiveAutoML.this.logger.info(n + ": " + scoreForThisAttributeSet);
							if (noImprovement > NUM_NOIMPROVEITERATIONS_FILTERING) {
								NaiveAutoML.this.logger.info("No improvement in " + NUM_NOIMPROVEITERATIONS_FILTERING + " iterations, stopping.");
								break;
							}
						}
						for (int n = 0; n < bestN; n++) {
							int att = ranking[n];
							NaiveAutoML.this.attributeVotes.put(att, NaiveAutoML.this.attributeVotes.computeIfAbsent(att, a -> 0) + (bestN - n));
						}
						locallyBestNumberOfFeaturesPerPreprocessor.put(preprocessor, bestN);
						scorePerPreprocessor.put(preprocessor, bestScore);
						NaiveAutoML.this.logger.info("Best score for pre-processor " + preprocessor + " was " + bestScore + " obtained using " + bestN + " attributes.");
						scoreListPerPreprocessor.put(preprocessor, scoreHistory);
					} catch (Exception e) {
						e.printStackTrace();
					}
				}
			});
		}
		pool.shutdown();
		pool.awaitTermination(10, TimeUnit.MINUTES);
		long timestampPreprocessingFinished = System.currentTimeMillis();
		this.logger.info("Runtime for pre-processing was " + ((timestampPreprocessingFinished - timestartStartPreprocessing) / 1000) + "s.");
		System.out.println(scoreListPerPreprocessor.entrySet().stream().map(e -> "\"" + e.getKey() + "\": " + e.getValue()).collect(Collectors.joining(",\n")));
		return new FilteringResult(rankings, locallyBestNumberOfFeaturesPerPreprocessor, scoreListPerPreprocessor, scorePerPreprocessor);
	}

	public Phase1Results getProbingResults(final IWekaInstances data) throws Exception {

		// Future<Double> majoritClassifierScoreOpt = pool.submit(() -> mccvSmallData.evaluate(majorityClassifier));
		// Future<Double> rfScoreOpt = pool.submit(() -> mccvSmallData.evaluate(pilot));
		// double majorityClassifierScore = majoritClassifierScoreOpt.get();
		// double rfScore = rfScoreOpt.get();
		// int totalweight = this.attributeVotes.values().stream().mapToInt(i -> i).sum();
		// for (Entry<Integer, Integer> entry : this.attributeVotes.entrySet()) {
		// this.attributeCandidates.add(entry.getKey());
		// this.pvectorForAttributeCandidates.add(entry.getValue() * 1.0 / totalweight);
		// }
		// this.logger.info("Majority Classifier Score: " + majorityClassifierScore + ". RF Base Score: " + rfScore + ". PVector for common attributes: " + this.pvectorForAttributeCandidates);

		/* STEP 1:
		 * - get attribute ranking for all pre-processors
		 * - get local minimum in monotone evaluations using a RF with 10 trees
		 * - get score of majority classifier
		 **/

		RacePool rp = this.iterativeEvaluation ? new RacePool(data, ANCHOR_POINTS_PHASE1) : new RacePool(data);
		rp.setLoggerName(this.getLoggerName() + ".pool");
		rp.setDeadline(this.deadline);

		/* in any case, evaluate the cheap learners on the original set */
		for (String bl : getBaseLearners()) {
			rp.submitCandidate(new PipelineDescription(null, bl, false)); // do never use cheap learners
		}
		rp.closeAndAwaitTermination();
		return new Phase1Results(null, rp);
	}

	public Phase1Results doFiltering(final IWekaInstances data, final Random random) throws Exception {
		RacePool rp = this.iterativeEvaluation ? new RacePool(data, ANCHOR_POINTS_PHASE1) : new RacePool(data);
		FilteringResult filterResult = null;
		if (this.enable_feature_projection) {
			this.logger.info("Do Filtering.");

			filterResult = this.getFilterRankings(data, random);
			List<Integer> attributesChosenByPreprocessingOptimizer = filterResult.getAttributSetWithBestScore();
			this.logger.info("Filtering determined best feature set {}", attributesChosenByPreprocessingOptimizer);

			if (attributesChosenByPreprocessingOptimizer.size() < data.getNumAttributes()) {
				/* create learning curves for each base learner*/
				for (String bl : getBaseLearners()) {
					rp.submitCandidate(new PipelineDescription(attributesChosenByPreprocessingOptimizer, bl, Util.hasCheapDefaultClassifier(bl)));
				}
			}
			else {
				this.logger.info("Filtering reveals that full attribute set is best. Not re-evaluating.");
			}
		}
		else {
			this.logger.info("Skipping Filtering.");
		}
		rp.closeAndAwaitTermination();
		return new Phase1Results(filterResult, rp);
	}

	public Phase2Results getConsolidationResults(final IWekaInstances data, final Phase1Results results) throws InterruptedException {

		RacePool pool = this.iterativeEvaluation ? new RacePool(results.getRacepool(), ANCHOR_POINTS_PHASE2) : new RacePool(results.getRacepool());
		pool.setLoggerName(this.getLoggerName() + ".pool");

		if (this.enable_consolidation) {
			List<Entry<PipelineDescription, EvaluationReports>> bestCandidatesWithScores = results.getRacepool().getCandidatesThatStillCanCompete(data.size());
			List<PipelineDescription> bestCandidates = bestCandidatesWithScores.stream().map(Entry::getKey).collect(Collectors.toList());
			StringBuilder sb = new StringBuilder();
			bestCandidatesWithScores.forEach(e -> sb.append("\n\t" + e.getKey() + ": " + e.getValue().getBestAverageScoreSeen()));
			this.logger.info("Re-Evaluating selected Cheap candidates using default parametrization. These are the candidates. {}", sb);

			for (PipelineDescription pl : bestCandidates) {
				pool.submitCandidate(new PipelineDescription(pl.getAttributes(), pl.getBaseLearner(), false));
			}
		} else {
			this.logger.info("Consolidation disabled. Continuing with the results we already had before.");
		}
		pool.closeAndAwaitTermination();
		return new Phase2Results(pool);
	}

	public Phase2Results doMetaLearningStage(final IWekaInstances data, final Phase1Results results) throws InterruptedException {
		if (this.enable_metalearners && System.currentTimeMillis() < this.deadline) { // insert meta-learners only after the bare learners
			int numClasses = WekaUtil.getClassesActuallyContainedInDataset(data.getInstances()).size();
			RacePool pool = this.iterativeEvaluation ? new RacePool(results.getRacepool(), ANCHOR_POINTS_PHASE2) : new RacePool(results.getRacepool());
			pool.setDeadline(Math.min(this.deadline, System.currentTimeMillis() + TIMEOUT_STAGE_META.milliseconds()));
			List<PipelineDescription> pipelines = results.getRacepool().getBestCandidates(10000).stream().map(e -> e.getKey()).collect(Collectors.toList());
			this.logger.info("Running Meta-Learner stage. Basis are {} pipelines.", pipelines.size());
			for (PipelineDescription pl : pipelines) {
				String baseLearner = pl.getBaseLearner();
				for (String metaLearner : WekaUtil.getMetaLearners()) {
					if (metaLearner.contains("Stacking") || metaLearner.contains("Vote") || metaLearner.contains("RandomSubspace") || metaLearner.contains("AttributeSelectedClassifier")) {
						continue;
					}
					if (metaLearner.contains("MultiClass") && numClasses <= 2) {
						continue;
					}
					if (baseLearner.contains("RandomForest") && metaLearner.contains("Bagging")) {
						continue;
					}
					PipelineDescription pld = new PipelineDescription(pl.getAttributes(), baseLearner, null, metaLearner, null);
					this.logger.debug("Enqueing evaluation of {}", pld);
					pool.submitCandidate(pld);
				}
			}
			pool.closeAndAwaitTermination();
			return new Phase2Results(pool);
		}
		else {
			this.logger.info("Skipping Meta-Learner stage.");
			return new Phase2Results(results.getRacepool());
		}
	}

	public Phase3Results getResultsAfterTuning(final IWekaInstances data, final RacePool candidatePool, final Random rand)
			throws ObjectEvaluationFailedException, InterruptedException, ComponentInstantiationFailedException, IOException, ExecutionException {

		/* creating new pool for tuning */
		RacePool pool = this.iterativeEvaluation ? new RacePool(candidatePool, ANCHOR_POINTS_PHASE3) : new RacePool(candidatePool);
		pool.setLoggerName(this.getLoggerName() + ".pool");
		pool.setDeadline(Math.min(this.deadline, System.currentTimeMillis() + TIMEOUT_STAGE_TUNING.milliseconds()));
		if (this.enable_parametertuning && this.deadline > System.currentTimeMillis()) {
			IComponentRepository components = MLPlanWekaBuilder.forClassification().withDataset(data).build().getHASCO().getInput().getComponents();
			RandomParameterOptimizer optimizer = new RandomParameterOptimizer(rand);
			for (PipelineDescription pl : candidatePool.getCandidatesThatStillCanCompete(data.size()).stream().map(e -> e.getKey()).collect(Collectors.toList())) {
				this.logger.info("Optimizing {}", pl);
				optimizer.optimize(pl, this.phase1Results, components, pool, TIMEOUT_STAGE_TUNING_PER_TEMPLATE);
			}
		} else {
			this.logger.info("Parameter tuning disabled. Continuing with candidates found already.");
		}
		pool.closeAndAwaitTermination();

		return new Phase3Results(pool);
	}

	public FinalResult doFinalValidation(final List<IWekaInstances> trainValidationSplit, final List<Entry<PipelineDescription, EvaluationReports>> candidatesWithPreviousEvaluations) throws InterruptedException {
		if (this.enable_final_evaluation && System.currentTimeMillis() < this.deadline) {
			AtomicReference<PipelineDescription> bestPipeline = new AtomicReference<>();
			AtomicDouble bestWeightedScore = new AtomicDouble(1);
			AtomicDouble bestValidationScore = new AtomicDouble(1);
			AtomicDouble internalScoreOfSolutionWithBestValidationScore = new AtomicDouble(1);
			ExecutorService pool = Executors.newFixedThreadPool(NUM_CPUS);

			final int MAX_EVALUATIONS = 100;
			AtomicInteger realizedEvaluations = new AtomicInteger(0);
			long deadline_here = Math.min(this.deadline, System.currentTimeMillis() + TIMEOUT_STAGE_VALIDATION.milliseconds());
			for (Entry<PipelineDescription, EvaluationReports> entry : candidatesWithPreviousEvaluations.stream().sorted((e1, e2) -> Double.compare(e1.getValue().getBestAverageScoreSeen(), e2.getValue().getBestAverageScoreSeen()))
					.collect(Collectors.toList())) {
				PipelineDescription pl = entry.getKey();
				pool.submit(() -> {
					try {
						if (System.currentTimeMillis() > deadline_here) {
							this.logger.debug("Skipping evaluation, no time left.");
							return;
						}
						IWekaInstances train = trainValidationSplit.get(0);
						IWekaInstances validate = trainValidationSplit.get(1);
						if (pl.getAttributes() != null) {
							train = reduceDimensionality(train, pl.getAttributes());
							validate = reduceDimensionality(validate, pl.getAttributes());
						}
						this.logger.info("Evaluating " + pl.getBaseLearner() + " + " + pl.getMetaLearner() + " on train data of size " + train.size() + " x " + train.getNumAttributes() + " and validation data of size " + validate.size()
						+ " x " + validate.getNumAttributes());
						double scoreValidation = new FixedSplitClassifierEvaluator(train, validate, EClassificationPerformanceMeasure.ERRORRATE).evaluate(Util.getClassifierFromDescription(pl));
						double scoreInternal = entry.getValue().getBestAverageScoreSeen();
						double weightedScore = 0.5 * (scoreInternal + scoreValidation);
						this.logger.info("Score of " + pl.getBaseLearner() + " + " + pl.getMetaLearner() + " = " + weightedScore + " = " + scoreInternal + " (int) -> " + scoreValidation + " (ext) for pipeline " + pl);
						synchronized (bestPipeline) {
							if (bestPipeline.get() == null || weightedScore < bestWeightedScore.get()) {
								bestWeightedScore.set(weightedScore);
								bestValidationScore.set(scoreValidation);
								internalScoreOfSolutionWithBestValidationScore.set(scoreInternal);
								bestPipeline.set(pl);
							}
						}

						if (realizedEvaluations.incrementAndGet() >= MAX_EVALUATIONS) {
							this.logger.info("Maximum number of final evaluations reached. Stopping.");
							return;
						}
					} catch (Exception e) {
						e.printStackTrace();
					}
				});
			}
			pool.shutdown();
			pool.awaitTermination(1, TimeUnit.DAYS);
			this.logger.info("Ready. Selected model is " + bestPipeline + " with internal score " + internalScoreOfSolutionWithBestValidationScore + " and validation score " + bestValidationScore);
			return new FinalResult(bestPipeline.get(), bestWeightedScore.get(), null, null);
		} else {
			List<Entry<PipelineDescription, EvaluationReports>> validCandidates = candidatesWithPreviousEvaluations.stream().filter(e -> !Double.isNaN(e.getValue().getBestAverageScoreSeen())).collect(Collectors.toList());
			this.logger.info("Final selection phase disabled, returning the best of the {}/{} valid previously found solutions.", validCandidates.size(), candidatesWithPreviousEvaluations.size());
			Entry<PipelineDescription, EvaluationReports> entry = validCandidates.stream().min((e1, e2) -> Double.compare(e1.getValue().getBestAverageScoreSeen(), e2.getValue().getBestAverageScoreSeen())).get();
			this.logger.info("The final result will be {} with performance  {}", entry.getKey(), entry.getValue().getBestAverageScoreSeen());
			return new FinalResult(entry.getKey(), entry.getValue().getBestAverageScoreSeen(), null, null);
		}
	}

	public Runnable getPreprocessor(final String preprocessorName, final IWekaInstances originalInstances, final Map<String, int[]> attributePriorityMap) {
		return new Runnable() {

			final Class<?> searcherClass = preprocessorName.contains("CfsSubsetEval") ? BestFirst.class : Ranker.class;

			@Override
			public void run() {
				try {
					ASEvaluation ce = ASEvaluation.forName(preprocessorName, null);
					Instances instances = originalInstances.getInstances();
					NaiveAutoML.this.logger.info("Bulding " + preprocessorName);
					ce.buildEvaluator(instances);
					NaiveAutoML.this.logger.info("Completed " + preprocessorName);
					ASSearch searcher = ASSearch.forName(this.searcherClass.getName(), null);
					attributePriorityMap.put(preprocessorName, searcher.search(ce, instances));
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		};
	}

	public static IWekaInstances reduceDimensionality(final IWekaInstances data, final Collection<Integer> attributes) throws Exception {
		if (attributes == null) {
			return data;
		}
		int[] ranking = new int[attributes.size()];
		Iterator<Integer> it = attributes.iterator();
		for (int i = 0; i < ranking.length; i++) {
			ranking[i] = it.next();
		}
		return reduceDimensionality(data, ranking, ranking.length);
	}

	public static WekaInstances reduceDimensionality(final IWekaInstances data, final int[] attributeOrder, final int newDimensionality) throws Exception {
		int[] m_selectedAttributeSet = new int[newDimensionality + 1];
		Instances instances = data.getInstances();
		m_selectedAttributeSet[newDimensionality] = instances.classIndex();
		for (int i = 0; i < newDimensionality; i++) {
			m_selectedAttributeSet[i] = attributeOrder[i];
		}
		Remove m_attributeFilter = new Remove();
		m_attributeFilter.setAttributeIndicesArray(m_selectedAttributeSet);
		m_attributeFilter.setInvertSelection(true);
		m_attributeFilter.setInputFormat(instances);
		return new WekaInstances(Filter.useFilter(instances, m_attributeFilter));
	}

	public static Collection<String> getBaseLearners() {
		List<String> learners = new ArrayList<>(WekaUtil.getBasicClassifiers());
		//		String annClasSName = MultilayerPerceptron.class.getName();
		//		learners.remove(annClasSName);
		//		learners.add(0, annClasSName);
		//		String svmClassName = SMO.class.getName();
		//		learners.remove(svmClassName);
		//		for (double c : Arrays.asList(0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)) {
		//			for (String kernelName : Arrays.asList(Puk.class.getName(), PolyKernel.class.getName(), NormalizedPolyKernel.class.getName(), RBFKernel.class.getName())) {
		//				learners.add(svmClassName + "-" + kernelName + "-" + c);
		//			}
		//		}
		return learners;
	}

	@Override
	public String getLoggerName() {
		return this.logger.getName();
	}

	@Override
	public void setLoggerName(final String name) {
		this.logger = LoggerFactory.getLogger(name);
	}

	public static void main(final String[] args) throws Exception {

		// long seed = 0;
		// ILabeledDataset<ILabeledInstance> dsOrig = OpenMLDatasetReader.deserializeDataset(23); // albert
		// ILabeledDataset<ILabeledInstance> dsOrig = OpenMLDatasetReader.deserializeDataset(4538); // GesturePhaseSegmentationProcessed
		ILabeledDataset<ILabeledInstance> dsOrig = new OpenMLDatasetReader().deserializeDataset(1485); // madelon
		//		ILabeledDataset<ILabeledInstance> dsOrig = OpenMLDatasetReader.deserializeDataset(1468);
		// ILabeledDataset<ILabeledInstance> dsOrig = OpenMLDatasetReader.deserializeDataset(1457); // amazon
		// ILabeledDataset<ILabeledInstance> dsOrig = OpenMLDatasetReader.deserializeDataset(41169); // helena
		// ILabeledDataset<ILabeledInstance> dsOrig = OpenMLDatasetReader.deserializeDataset(40981); // Australian
		// ILabeledDataset<ILabeledInstance> dsOrig = OpenMLDatasetReader.deserializeDataset(4134); // Biosresponse
		// ILabeledDataset<ILabeledInstance> dsOrig = OpenMLDatasetReader.deserializeDataset(1590); // adult
		// ILabeledDataset<ILabeledInstance> dsOrig = OpenMLDatasetReader.deserializeDataset(42742);//
		//		ILabeledDataset<ILabeledInstance> dsOrig = OpenMLDatasetReader.deserializeDataset(30); // page-blocks
		// ILabeledDataset<ILabeledInstance> dsOrig = OpenMLDatasetReader.deserializeClassificationDataset(41066); // secom
		// ILabeledDataset<ILabeledInstance> dsOrig = OpenMLDatasetReader.deserializeClassificationDataset(1501); // semeion
		// ILabeledDataset<ILabeledInstance> dsOrig = OpenMLDatasetReader.deserializeClassificationDataset(4136); // dexter
		// ILabeledDataset<ILabeledInstance> dsOrig = OpenMLDatasetReader.deserializeClassificationDataset(4137); // dorothea
		// ILabeledDataset<ILabeledInstance> dsOrig = OpenMLDatasetReader.deserializeClassificationDataset(41064); // convex

		System.out.println("Applying approach to dataset with dimensions " + dsOrig.size() + " x " + dsOrig.getNumAttributes());

		DescriptiveStatistics stats = new DescriptiveStatistics();
		WekaInstances ds = new WekaInstances(dsOrig);
		for (int seed = 0; seed < 1; seed++) {
			List<WekaInstances> trainTestSplit = SplitterUtil.getLabelStratifiedTrainTestSplit(ds, seed, 0.9);

			System.out.println("Instances admitted for training: " + trainTestSplit.get(0).size());
			long start = System.currentTimeMillis();
			NaiveAutoML nautoml = new NaiveAutoMLBuilder().withPreprocessing().withParameterTuning().withFinalSelectionPhase().build();
			nautoml.setLoggerName(LoggerUtil.LOGGER_NAME_TESTEDALGORITHM);
			FinalResult result = nautoml.findBestPipelineDescription(trainTestSplit.get(0), new Timeout(1, TimeUnit.MINUTES));
			long runtime = System.currentTimeMillis() - start;

			/* compile solution classifier */
			PipelineDescription bestPipeline = result.getFinalChoice();
			Classifier baseClassifier = Util.getClassifier(bestPipeline.getBaseLearner(), bestPipeline.getBaseLearnerParams()).getClassifier();
			SingleClassifierEnhancer metaClassifier = bestPipeline.getMetaLearner() != null ? (SingleClassifierEnhancer) AbstractClassifier.forName(bestPipeline.getMetaLearner(), bestPipeline.getMetaLearnerParams()) : null;
			if (metaClassifier != null) {
				metaClassifier.setClassifier(baseClassifier);
			}
			IWekaClassifier solution = new WekaClassifier(metaClassifier != null ? metaClassifier : baseClassifier);

			/* validate solution */
			System.out.println("Evaluating solution " + solution.getClassifier().getClass().getName() + " on attributes " + bestPipeline.getAttributes());
			Collection<Integer> selectedAttributes = bestPipeline.getAttributes();
			IWekaInstances reducedTrainSet = selectedAttributes != null ? reduceDimensionality(trainTestSplit.get(0), selectedAttributes) : trainTestSplit.get(0);
			IWekaInstances reducedTestSet = selectedAttributes != null ? reduceDimensionality(trainTestSplit.get(1), selectedAttributes) : trainTestSplit.get(1);
			FixedSplitClassifierEvaluator eval = new FixedSplitClassifierEvaluator(reducedTrainSet, reducedTestSet, EClassificationPerformanceMeasure.ERRORRATE);
			double score = eval.evaluate(solution);
			stats.addValue(score);
			System.out.println("Final test-fold score: " + score + ". Search time was " + (runtime / 1000) + "s.");
		}
		System.out.println(stats);
	}
}
