package naiveautoml.experiments;

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

import org.moeaframework.util.io.FileUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.libs.jaicore.basic.FileUtil;
import ai.libs.jaicore.experiments.ExperimentDBEntry;
import ai.libs.jaicore.experiments.ExperimenterFrontend;
import ai.libs.jaicore.experiments.IExperimentIntermediateResultProcessor;
import ai.libs.jaicore.experiments.IExperimentSetEvaluator;
import ai.libs.jaicore.experiments.exceptions.ExperimentDBInteractionFailedException;
import ai.libs.jaicore.experiments.exceptions.ExperimentEvaluationFailedException;
import ai.libs.jaicore.experiments.exceptions.ExperimentFailurePredictionException;
import ai.libs.jaicore.processes.ProcessIDNotRetrievableException;
import ai.libs.jaicore.processes.ProcessUtil;

public class NaiveAutoMLExperimentRunner implements IExperimentSetEvaluator {

	private static final Logger logger = LoggerFactory.getLogger("experimenter");

	@Override
	public void evaluate(final ExperimentDBEntry experimentEntry, final IExperimentIntermediateResultProcessor processor) throws ExperimentEvaluationFailedException, ExperimentFailurePredictionException, InterruptedException {
		try {

			/* prepare variables for experiment */
			Map<String, String> keys = experimentEntry.getExperiment().getValuesOfKeyFields();
			int openmlid = Integer.parseInt(keys.get("openmlid"));
			int seed = Integer.parseInt(keys.get("seed"));
			String algo = keys.get("algorithm");
			int timeoutGlobal = Integer.parseInt(keys.get("timeout_total"));
			int timeoutEval = Integer.parseInt(keys.get("timeout_evaluation"));

			/* execute experiment */
			Map<String, Object> results = this.runPythonExperiment(openmlid, algo, seed, timeoutGlobal, timeoutEval);

			/* write results */
			processor.processResults(results);
			logger.info("Results written");
		}
		catch (Exception e) {
			throw new ExperimentEvaluationFailedException(e);
		}
	}

	public Map<String, Object> runPythonExperiment(final int openmlid, final String algorithm, final int seed, final int timeoutGlobal, final int timeoutSingleEval) throws InterruptedException, IOException, ProcessIDNotRetrievableException {
		StringBuilder optionBuilder = new StringBuilder();
		optionBuilder.append("--dataset_id=" + openmlid);
		optionBuilder.append(" --timeout_total=" + timeoutGlobal);
		optionBuilder.append(" --timeout_per_eval=" + timeoutSingleEval);
		optionBuilder.append(" --algorithm=" + algorithm);
		optionBuilder.append(" --seed=" + seed);

		String id = UUID.randomUUID().toString();

		/* serialize files */
		File workingDirectory = new File("singularity");
		File folder = new File(workingDirectory.getAbsolutePath() + File.separator + "tmp/" + id);
		File file = new File("runexperiment.py");
		FileUtils.mkdir(new File(workingDirectory.getAbsolutePath() + File.separator + "tmp/" + id));
		optionBuilder.append(" --folder=" + folder);
		System.out.println("Current memory usage is " + ((Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / 1024 / 1024) + "MB");

		String singularityImage = "test.simg";
		System.out.println("Executing " + new File(workingDirectory + File.separator + file) + " in singularity.");
		String options = optionBuilder.toString();
		System.out.println("Options: " + options);
		List<String> cmd = Arrays.asList("singularity", "exec", singularityImage, "bash", "-c", "python3.8 " + file +  " " + options);
		System.out.println("Running " + cmd.stream().collect(Collectors.joining(" ")));
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
			String onlineData =  onlineDataFile.exists() ? FileUtil.readFileAsString(onlineDataFile) : "[]";

			/* compile result map */
			Map<String, Object> results = new HashMap<>();
			results.put("chosenmodel", chosenModel);
			results.put("errorrate", finalErrorRate);
			results.put("metric", finalRequestedMetric);
			if (onlineData != null) {
				results.put("onlinedata", onlineData.toString());
			}
			return results;
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
