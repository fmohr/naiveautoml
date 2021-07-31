package naiveautoml.experiments;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledInstance;
import org.api4.java.algorithm.Timeout;
import org.api4.java.algorithm.exceptions.AlgorithmTimeoutedException;
import org.moeaframework.util.io.FileUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.google.common.eventbus.Subscribe;
import com.google.common.util.concurrent.AtomicDouble;

import ai.libs.jaicore.basic.FileUtil;
import ai.libs.jaicore.experiments.Experiment;
import ai.libs.jaicore.experiments.ExperimentDBEntry;
import ai.libs.jaicore.experiments.ExperimenterFrontend;
import ai.libs.jaicore.experiments.IExperimentIntermediateResultProcessor;
import ai.libs.jaicore.experiments.IExperimentSetEvaluator;
import ai.libs.jaicore.experiments.exceptions.ExperimentDBInteractionFailedException;
import ai.libs.jaicore.experiments.exceptions.ExperimentEvaluationFailedException;
import ai.libs.jaicore.experiments.exceptions.ExperimentFailurePredictionException;
import ai.libs.jaicore.logging.LoggerUtil;
import ai.libs.jaicore.ml.classification.loss.dataset.EClassificationPerformanceMeasure;
import ai.libs.jaicore.ml.core.dataset.serialization.CSVDatasetAdapter;
import ai.libs.jaicore.ml.core.dataset.serialization.OpenMLDatasetReader;
import ai.libs.jaicore.ml.core.evaluation.evaluator.FixedSplitClassifierEvaluator;
import ai.libs.jaicore.ml.core.filter.SplitterUtil;
import ai.libs.jaicore.ml.weka.WekaUtil;
import ai.libs.jaicore.ml.weka.classification.learner.IWekaClassifier;
import ai.libs.jaicore.ml.weka.classification.learner.WekaClassifier;
import ai.libs.jaicore.ml.weka.dataset.IWekaInstances;
import ai.libs.jaicore.ml.weka.dataset.WekaInstances;
import ai.libs.jaicore.processes.ProcessIDNotRetrievableException;
import ai.libs.jaicore.processes.ProcessUtil;
import ai.libs.mlplan.core.events.ClassifierFoundEvent;
import ai.libs.mlplan.weka.MLPlan4Weka;
import ai.libs.mlplan.weka.MLPlanWekaBuilder;
import naiveautoml.FinalResult;
import naiveautoml.NaiveAutoML;
import naiveautoml.NaiveAutoMLBuilder;
import naiveautoml.PipelineDescription;
import naiveautoml.Util;
import naiveautoml.autoweka.AutoWEKAClassifierWrapper;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.SingleClassifierEnhancer;

public class NaiveAutoMLExperimentRunner implements IExperimentSetEvaluator {

	private static final Logger logger = LoggerFactory.getLogger("experimenter");

	public static final Timeout TIMEOUT_SOTA_TOOLS = new Timeout(24, TimeUnit.HOURS);


	public NaiveAutoML getNaiveAutoML(final String algo) {

		/* get flags */
		boolean monotone = algo.contains("monotone");
		boolean validation = algo.contains("validation");
		boolean tuning = algo.contains("tuning") || (monotone && validation);
		boolean meta = algo.contains("meta") || (monotone && tuning);
		boolean filtering = algo.contains("filtering") || (monotone && meta);

		NaiveAutoMLBuilder builder = new NaiveAutoMLBuilder();
		if (filtering) {
			builder.withPreprocessing();
		}
		if (meta) {
			builder.withMetaLearners();
		}
		if (tuning) {
			builder.withParameterTuning();
		}
		if (validation) {
			builder.withFinalSelectionPhase();
		}
		if (!algo.contains("noniterative")) {
			builder.withIterativeEvaluations();
		}
		return builder.build();
	}

	@Override
	public void evaluate(final ExperimentDBEntry experimentEntry, final IExperimentIntermediateResultProcessor processor) throws ExperimentEvaluationFailedException, ExperimentFailurePredictionException, InterruptedException {
		try {
			Map<String, String> keys = experimentEntry.getExperiment().getValuesOfKeyFields();
			int openmlid = Integer.valueOf(keys.get("openmlid"));
			int seed = Integer.valueOf(keys.get("seed"));
			String algo = keys.get("algorithm");
			int timeout = Integer.valueOf(keys.get("timeout"));

			/* create split and final evaluator */
			OpenMLDatasetReader reader = new OpenMLDatasetReader();
			reader.setLoggerName(logger.getName() + ".openmlreader");
			ILabeledDataset<ILabeledInstance> dsOrig = reader.deserializeDataset(openmlid);
			logger.info("Applying approach to dataset with dimensions {} x {}. Now creating instances object.", dsOrig.size(), dsOrig.getNumAttributes());
			//			WekaInstances ds = new WekaInstances(dsOrig);
			logger.info("Done. Creating stratified split.");
			List<WekaInstances> trainTestSplit = SplitterUtil.getLabelStratifiedTrainTestSplit(dsOrig, seed, 0.9).stream().map(i -> new WekaInstances(i)).collect(Collectors.toList());
			dsOrig = null; // make memory

			/* find best pipeline */
			logger.info("Instances admitted for training: {}", trainTestSplit.get(0).size());
			long start = System.currentTimeMillis();

			String chosenModel;
			double finalErrorRate;
			double finalRequestedMetric = Double.NaN;
			JsonNode onlineData = null;
			if (algo.startsWith("scientist")) {

				if (algo.contains("java")) {
					logger.info("Running Naive AutoML on dataset {}", openmlid);
					NaiveAutoML nautoml = this.getNaiveAutoML(algo);

					nautoml.setLoggerName(LoggerUtil.LOGGER_NAME_EVALUATEDALGORITHM);
					FinalResult report = nautoml.findBestPipelineDescription(trainTestSplit.get(0), new Timeout(1, TimeUnit.MINUTES));
					long runtime = System.currentTimeMillis() - start;
					PipelineDescription bestPipeline = report.getFinalChoice();


					Map<String, Integer> runtimesPerStage = null;
					final Map<Integer, Double> bestScoreHistory = new HashMap<>();
					runtimesPerStage = report.getRuntimesPerStage();
					for (Entry<Long, Double> hEntry : report.getResultHistory().entrySet()) {
						bestScoreHistory.put((int)(hEntry.getKey() - start), hEntry.getValue());
					}
					System.out.println(bestScoreHistory);
					onlineData = this.compileHistoryMaps(runtimesPerStage, bestScoreHistory);

					/* compile solution classifier */
					Classifier baseClassifier = Util.getClassifier(bestPipeline.getBaseLearner(), bestPipeline.getBaseLearnerParams()).getClassifier();
					SingleClassifierEnhancer metaClassifier = bestPipeline.getMetaLearner() != null ? (SingleClassifierEnhancer)AbstractClassifier.forName(bestPipeline.getMetaLearner(), bestPipeline.getMetaLearnerParams()) : null;
					if (metaClassifier != null) {
						metaClassifier.setClassifier(baseClassifier);
					}
					IWekaClassifier solution = new WekaClassifier(metaClassifier != null ? metaClassifier : baseClassifier);

					/* validate solution */
					logger.info("Evaluating solution " + solution.getClassifier().getClass().getName() + " on attributes " + bestPipeline.getAttributes());
					Collection<Integer> selectedAttributes = bestPipeline.getAttributes();
					IWekaInstances reducedTrainSet = selectedAttributes != null ? NaiveAutoML.reduceDimensionality(trainTestSplit.get(0), selectedAttributes) : trainTestSplit.get(0);
					IWekaInstances reducedTestSet = selectedAttributes  != null ? NaiveAutoML.reduceDimensionality(trainTestSplit.get(1), selectedAttributes) : trainTestSplit.get(1);
					FixedSplitClassifierEvaluator eval = new FixedSplitClassifierEvaluator(reducedTrainSet, reducedTestSet, EClassificationPerformanceMeasure.ERRORRATE);
					finalErrorRate = eval.evaluate(solution);
					chosenModel = bestPipeline.toString();
					logger.info("Final test-fold score: " + finalErrorRate + ". Search time was " + (runtime / 1000) + "s.");
				}
				else if (algo.contains("python")) {
					StringBuilder sb = new StringBuilder();
					boolean monotone = algo.contains("monotone");
					boolean validation = algo.contains("validation");
					boolean tuning = algo.contains("tuning") || (monotone && validation);
					boolean wrapping = algo.contains("wrapping") || (monotone && tuning);
					boolean meta = algo.contains("meta") || (monotone && wrapping);
					boolean filtering = algo.contains("filtering") || (monotone && meta);
					boolean scaling = algo.contains("scaling") || (monotone && filtering);

					sb.append("(");
					sb.append(scaling ? "True" : "False");
					sb.append(",");
					sb.append(filtering ? "True" : "False");
					sb.append(",");
					sb.append(meta ? "True" : "False");
					sb.append(",");
					sb.append("False"); // generally disable wrapping
					sb.append(",");
					sb.append(tuning ? "True" : "False");
					sb.append(",");
					sb.append(validation ? ".2" : "None");
					sb.append(",");
					sb.append(algo.contains("noniterative") ? "False" : "True");
					sb.append(")");
					List<Object> evalResults = this.runPythonExperiment(trainTestSplit, "runnaive.py", sb.toString(), timeout);
					chosenModel = (String)evalResults.get(0);
					finalErrorRate = (double)evalResults.get(1);
					onlineData = new ObjectMapper().readTree((String)evalResults.get(2));
				}
				else {
					throw new IllegalArgumentException("Unknown description for naive tool: " + algo);
				}
			}
			else if (algo.equals("mlplan")) {
				logger.info("Running MLPlan on dataset " + openmlid);

				/* note that we set number of CPUs to 3 to make it more robust. Anyway, only one CPU will be used for evaluations. But if memory errors etc. occur, then only the thread dies, not the whole application. */
				MLPlan4Weka mlplan = MLPlanWekaBuilder.forClassification().withDataset(trainTestSplit.get(0)).withNumCpus(3).withTimeOut(TIMEOUT_SOTA_TOOLS).withNodeEvaluationTimeOut(new Timeout(60, TimeUnit.SECONDS)).withMCCVBasedCandidateEvaluationInSearchPhase(5, .7).build();
				mlplan.setLoggerName(LoggerUtil.LOGGER_NAME_EVALUATEDALGORITHM);
				long timestampStart = System.currentTimeMillis();
				AtomicDouble bestScore = new AtomicDouble(1);
				Map<String, Integer> runtimesPerStage = null;
				final Map<Integer, Double> bestScoreHistory = new HashMap<>();
				mlplan.registerListener(new Object() {

					@Subscribe
					public void receiveSolution(final ClassifierFoundEvent e) {
						if (e.getScore() < bestScore.get()) {
							bestScore.set(e.getScore());
							bestScoreHistory.put((int)(System.currentTimeMillis() - timestampStart), e.getScore());
						}
					}
				});
				try {
					mlplan.call();
				}
				catch (AlgorithmTimeoutedException e) {
					System.out.println("Caught back control after timeout.");
				}
				IWekaClassifier solution = mlplan.getSelectedClassifier();
				chosenModel = WekaUtil.getClassifierDescriptor(solution.getClassifier());
				FixedSplitClassifierEvaluator eval = new FixedSplitClassifierEvaluator(trainTestSplit.get(0), trainTestSplit.get(1), EClassificationPerformanceMeasure.ERRORRATE);
				finalErrorRate = eval.evaluate(solution);
				onlineData = this.compileHistoryMaps(runtimesPerStage, bestScoreHistory);
			}
			else if (algo.equals("auto-weka")) {
				Map<String, Integer> runtimesPerStage = null;
				final Map<Integer, Double> bestScoreHistory = new HashMap<>();
				AutoWEKAClassifierWrapper aw = new AutoWEKAClassifierWrapper();
				aw.setTimeLimit((int)TIMEOUT_SOTA_TOOLS.minutes());
				aw.setParallelRuns(1);
				aw.buildClassifier(trainTestSplit.get(0).getInstances());
				IWekaClassifier solution = new WekaClassifier(aw.getChosenModel());
				chosenModel = WekaUtil.getClassifierDescriptor(solution.getClassifier());
				FixedSplitClassifierEvaluator eval = new FixedSplitClassifierEvaluator(trainTestSplit.get(0), trainTestSplit.get(1), EClassificationPerformanceMeasure.ERRORRATE);
				finalErrorRate = eval.evaluate(solution);
				bestScoreHistory.putAll(aw.getHistory());
				onlineData = this.compileHistoryMaps(runtimesPerStage, bestScoreHistory);
			}
			else if (algo.equals("auto-sklearn")) {
				List<Object> evalResults = this.runPythonExperiment(trainTestSplit, "runautosklearn.py", timeout);
				chosenModel = (String)evalResults.get(0);
				finalErrorRate = (double)evalResults.get(1);
				finalRequestedMetric = (double)evalResults.get(2);
				onlineData = new ObjectMapper().readTree((String)evalResults.get(3));
			}
			else if (algo.equals("naive-python")) {
				List<Object> evalResults = this.runPythonExperiment(trainTestSplit, "runnaive.py", timeout);
				chosenModel = (String)evalResults.get(0);
				finalErrorRate = (double)evalResults.get(1);
				finalRequestedMetric = (double)evalResults.get(2);
				onlineData = new ObjectMapper().readTree((String)evalResults.get(3));
			}
			else if (algo.equals("seminaive-python")) {
				List<Object> evalResults = this.runPythonExperiment(trainTestSplit, "runseminaive.py", timeout);
				chosenModel = (String)evalResults.get(0);
				finalErrorRate = (double)evalResults.get(1);
				finalRequestedMetric = (double)evalResults.get(2);
				onlineData = new ObjectMapper().readTree((String)evalResults.get(3));
			}
			else if (algo.equals("gama")) {
				List<Object> evalResults = this.runPythonExperiment(trainTestSplit, "rungama.py", timeout);
				chosenModel = (String)evalResults.get(0);
				finalErrorRate = (double)evalResults.get(1);
				finalRequestedMetric = (double)evalResults.get(2);
				onlineData = new ObjectMapper().readTree((String)evalResults.get(3));
			}
			else {
				throw new IllegalArgumentException();
			}

			Map<String, Object> results = new HashMap<>();

			results.put("chosenmodel", chosenModel);
			results.put("errorrate", finalErrorRate);
			results.put("metric", finalRequestedMetric);
			if (onlineData != null) {
				results.put("onlinedata", onlineData.toString());
			}
			processor.processResults(results);
			System.out.println("Results written (error rate " + finalErrorRate + " for model " + chosenModel + "), exiting.");
		}
		catch (Exception e) {
			throw new ExperimentEvaluationFailedException(e);
		}
	}

	public JsonNode compileHistoryMaps(final Map<String, Integer> runtimesPerStage, final Map<Integer, Double> bestScoreHistory) {
		ObjectMapper om = new ObjectMapper();
		ObjectNode infoNode = om.createObjectNode();
		if (runtimesPerStage != null) {
			ObjectNode runtimeNode = om.createObjectNode();
			infoNode.set("stageruntimes", runtimeNode);
			for (Entry<String, Integer> e : runtimesPerStage.entrySet()) {
				runtimeNode.put(e.getKey(), e.getValue());
			}
		}
		if (bestScoreHistory != null) {
			ArrayNode historyNode = om.createArrayNode();
			infoNode.set("history", historyNode);
			for (Entry<Integer, Double> e : bestScoreHistory.entrySet()) {
				ArrayNode entryNode = om.createArrayNode();
				entryNode.add(e.getKey());
				entryNode.add(e.getValue());
				historyNode.add(entryNode);
			}
		}
		return infoNode;
	}

	public List<Object> runPythonExperiment(final List<WekaInstances> trainTestSplit, final String file, final int timeout) throws InterruptedException, IOException, ProcessIDNotRetrievableException {
		return this.runPythonExperiment(trainTestSplit, file, null, timeout);
	}

	public List<Object> runPythonExperiment(final List<WekaInstances> trainTestSplit, final String file, final String options, final int timeout) throws InterruptedException, IOException, ProcessIDNotRetrievableException {

		String id = UUID.randomUUID().toString();

		/* serialize files */
		File workingDirectory = new File("python/singularity");
		File folder = new File(workingDirectory.getAbsolutePath() + File.separator + "tmp/" + id);
		String trainFile =  folder + "/train.csv";
		String testFile = folder +  "/test.csv";
		FileUtils.mkdir(new File(workingDirectory.getAbsolutePath() + File.separator + "tmp/" + id));
		CSVDatasetAdapter.writeDataset(new File(trainFile), trainTestSplit.get(0));
		CSVDatasetAdapter.writeDataset(new File(testFile), trainTestSplit.get(1));
		System.out.println("Current memory usage is " + ((Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / 1024 / 1024) + "MB");

		//				ArffDatasetAdapter.serializeDataset(new File(trainFile), trainTestSplit.get(0));
		//				ArffDatasetAdapter.serializeDataset(new File(testFile), trainTestSplit.get(1));
		String labelAttribute = trainTestSplit.get(0).getLabelAttribute().getName();
		trainTestSplit.clear(); // make space for memory

		String singularityImage = "test.simg";
		System.out.println("Executing " + new File(workingDirectory + File.separator + file) + " in singularity.");
		System.out.println("Options: " + options);
		List<String> cmd = Arrays.asList("singularity", "exec", singularityImage, "bash", "-c", "python3 " + file + (options == null ? "" : " \"" + options + "\"") + " " + trainFile + " " + testFile + " " + labelAttribute + " " + timeout);
		System.out.println("Running " + cmd);
		ProcessBuilder pb = new ProcessBuilder(cmd);
		pb.directory(workingDirectory);
		pb.redirectErrorStream(true);
		System.out.println("Clearing memory.");
		Thread.sleep(2000);
		System.gc();
		Thread.sleep(2000);
		System.out.println("Starting process. Current memory usage is " + ((Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / 1024 / 1024) + "MB");
		Process p = pb.start();
		System.out.println("PID: " + ProcessUtil.getPID(p));
		try (BufferedReader br = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
			String line;
			while ((line = br.readLine()) != null) {
				System.out.println(" --> " + line);
			}

			System.out.println("awaiting termination");
			while (p.isAlive()) {
				Thread.sleep(1000);
			}
			System.out.println("ready");

			Object chosenModel = FileUtil.readFileAsString(new File(folder + File.separator + "model.txt"));
			Object finalErrorRate = Double.valueOf(FileUtil.readFileAsString(new File(folder + File.separator + "error_rate.txt")));
			Object finalRequestedMetric = Double.valueOf(FileUtil.readFileAsString(new File(folder + File.separator + "score.txt")));
			File onlineDataFile = new File(folder + File.separator + "onlinedata.txt");
			String onlinedata =  onlineDataFile.exists() ? FileUtil.readFileAsString(onlineDataFile) : "[]";
			return Arrays.asList(chosenModel, finalErrorRate, finalRequestedMetric, onlinedata);
		}
		finally {
			System.out.println("KILLING PROCESS!");
			ProcessUtil.killProcess(p);
		}
	}

	public static void simulateAlgorithm(final int openmlid, final String algo) throws ExperimentEvaluationFailedException, ExperimentFailurePredictionException, InterruptedException {
		Map<String, String> exp = new HashMap<>();

		exp.put("openmlid", "" + openmlid);
		exp.put("seed", "0");
		exp.put("timeout", "120");
		exp.put("algorithm", algo);

		new NaiveAutoMLExperimentRunner().evaluate(new ExperimentDBEntry(0, new Experiment(8000, 1, exp)), p -> {});

	}

	public static void main(final String[] args) throws ExperimentDBInteractionFailedException, InterruptedException, ExperimentEvaluationFailedException, ExperimentFailurePredictionException {

		String databaseconf = args[0];
		String jobInfo = args[1];

		/* setup experimenter frontend */
		ExperimenterFrontend fe = new ExperimenterFrontend().withEvaluator(new NaiveAutoMLExperimentRunner()).withExperimentsConfig(new File("conf/experiments.conf")).withDatabaseConfig(new File(databaseconf));
		fe.setLoggerName("frontend");
		fe.withExecutorInfo(jobInfo);

		long startTime = System.currentTimeMillis();
		long elapsedTime = 0;
		long totalAvailableTime = 60 * 20;
		do {
			logger.info("Elapsed time: {}/{}. Conducting next experiment. Currently used memory is {}MB. Free memory is {}MB.", elapsedTime, totalAvailableTime,
					(Runtime.getRuntime().maxMemory() - Runtime.getRuntime().freeMemory()) / (1024 * 1024.0), Runtime.getRuntime().freeMemory() / (1024 * 1024.0));
			fe.randomlyConductExperiments(1);
			elapsedTime = (System.currentTimeMillis() - startTime) / 60000;
		} while (elapsedTime + 90 <= totalAvailableTime && fe.mightHaveMoreExperiments());
		logger.info("Finishing, no more experiments!");
	}
}
