package naiveautoml.legitimacy;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.stream.Collectors;

import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledInstance;
import org.moeaframework.util.io.FileUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import ai.libs.jaicore.basic.FileUtil;
import ai.libs.jaicore.experiments.ExperimentDBEntry;
import ai.libs.jaicore.experiments.ExperimenterFrontend;
import ai.libs.jaicore.experiments.IExperimentIntermediateResultProcessor;
import ai.libs.jaicore.experiments.IExperimentSetEvaluator;
import ai.libs.jaicore.experiments.exceptions.ExperimentDBInteractionFailedException;
import ai.libs.jaicore.experiments.exceptions.ExperimentEvaluationFailedException;
import ai.libs.jaicore.experiments.exceptions.ExperimentFailurePredictionException;
import ai.libs.jaicore.ml.core.dataset.serialization.CSVDatasetAdapter;
import ai.libs.jaicore.ml.core.dataset.serialization.OpenMLDatasetReader;
import ai.libs.jaicore.ml.core.filter.SplitterUtil;
import ai.libs.jaicore.ml.weka.dataset.WekaInstances;
import ai.libs.jaicore.processes.ProcessIDNotRetrievableException;
import ai.libs.jaicore.processes.ProcessUtil;

public class LegitimacyExperimentRunner implements IExperimentSetEvaluator {

	private static final Logger logger = LoggerFactory.getLogger("experimenter");

	@Override
	public void evaluate(final ExperimentDBEntry experimentEntry, final IExperimentIntermediateResultProcessor processor) throws ExperimentEvaluationFailedException, ExperimentFailurePredictionException, InterruptedException {
		try {

			/* extract infos */
			Map<String, String> keys = experimentEntry.getExperiment().getValuesOfKeyFields();
			int openmlid = Integer.parseInt(keys.get("openmlid"));
			int seed = Integer.parseInt(keys.get("seed"));
			String dataPreProcessor = keys.get("datapreprocessor");
			String featurePreProcessor = keys.get("featurepreprocessor");
			String predictor = keys.get("predictor");
			boolean hpo = Boolean.parseBoolean(keys.get("hpo"));

			/* create split */
			OpenMLDatasetReader reader = new OpenMLDatasetReader();
			reader.setLoggerName(logger.getName() + ".openmlreader");
			ILabeledDataset<ILabeledInstance> dsOrig = reader.deserializeDataset(openmlid);
			logger.info("Applying approach to dataset with dimensions {} x {}. Now creating instances object.", dsOrig.size(), dsOrig.getNumAttributes());
			logger.info("Done. Creating stratified split.");
			List<WekaInstances> trainTestSplit = SplitterUtil.getLabelStratifiedTrainTestSplit(dsOrig, seed, 0.9).stream().map(i -> new WekaInstances(i)).collect(Collectors.toList());
			dsOrig = null; // make memory
			logger.info("Instances admitted for training: {}", trainTestSplit.get(0).size());

			/* conduct experiment */
			JsonNode evalResults = this.runPythonExperiment(trainTestSplit, dataPreProcessor, featurePreProcessor, predictor, hpo);

			/* compile and send result map */
			Map<String, Object> results = new HashMap<>();
			results.put("chosendatapreprocessor", evalResults.asText("chosenparameters_datapreprocessor"));
			results.put("chosenfeaturepreprocessor", evalResults.asText("chosenparameters_featurepreprocessor"));
			results.put("chosenpredictor", evalResults.asText("chosenparameters_predictor"));
			//			results.put("errorrate", evalResults.get("errorrate").asDouble());
			results.put("metric", evalResults.get("metric").asDouble());
			results.put("history", evalResults.get("history"));
			processor.processResults(results);
			System.out.println("Results written:" + evalResults);
		}
		catch (Exception e) {
			throw new ExperimentEvaluationFailedException(e);
		}
	}

	public JsonNode runPythonExperiment(final List<WekaInstances> trainTestSplit, final String dataPreProcessor, final String featurePreProcessor, final String predictor, final boolean hpo) throws InterruptedException, IOException, ProcessIDNotRetrievableException {

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

		String labelAttribute = trainTestSplit.get(0).getLabelAttribute().getName();
		trainTestSplit.clear(); // make space for memory

		String singularityImage = "test.simg";
		String file = "evalpipeline.py";
		System.out.println("Executing " + new File(workingDirectory + File.separator + file) + " in singularity.");
		String options = dataPreProcessor + " " + featurePreProcessor + " " + predictor + " " + hpo;
		System.out.println("Options: " + options);
		List<String> cmd = Arrays.asList("singularity", "exec", singularityImage, "bash", "-c", "python3 " + file + " \"" + options + "\"" + " " + trainFile + " " + testFile + " " + labelAttribute);
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

			return new ObjectMapper().readTree(FileUtil.readFileAsString(new File(folder + File.separator + "results.json")));
		}
		finally {
			System.out.println("KILLING PROCESS!");
			ProcessUtil.killProcess(p);
		}
	}

	public static void main(final String[] args) throws ExperimentDBInteractionFailedException, InterruptedException, ExperimentEvaluationFailedException, ExperimentFailurePredictionException {

		String databaseconf = args[0];
		String jobInfo = args[1];

		/* setup experimenter frontend */
		ExperimenterFrontend fe = new ExperimenterFrontend().withEvaluator(new LegitimacyExperimentRunner()).withExperimentsConfig(new File("conf/experiments-legitimacy.conf")).withDatabaseConfig(new File(databaseconf));
		fe.setLoggerName("frontend");
		fe.withExecutorInfo(jobInfo);
		for (int i = 0; i < 1000000; i++) {
			fe.randomlyConductExperiments(1);
		}
	}
}
