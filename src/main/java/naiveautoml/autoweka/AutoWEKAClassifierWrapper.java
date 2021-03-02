package naiveautoml.autoweka;
import java.io.File;
import java.io.IOException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ai.libs.jaicore.basic.FileUtil;
import weka.classifiers.Classifier;
import weka.classifiers.meta.AutoWEKAClassifier;

public class AutoWEKAClassifierWrapper extends AutoWEKAClassifier {

	private String paramDir;

	public AutoWEKAClassifierWrapper() {
		super();
		this.paramDir = autoweka.Util.getAutoWekaDistributionPath() + File.separator + "params";
		System.out.println("Expecting param file in " + this.paramDir);
	}

	public Classifier getChosenModel() {
		return this.classifier;
	}

	private List<String> getLogFileContent() throws IOException {
		return FileUtil.readFileAsList(new File(msExperimentPaths[0] + File.separator + "Auto-WEKA/out/autoweka/log-run123.txt"));
	}

	public Map<Integer, Double> getHistory() throws IOException, ParseException {
		List<String> lines = this.getLogFileContent();
		SimpleDateFormat sdf = new SimpleDateFormat("HH:mm:ss");
		long timestamp_start = -1;
		long timestamp_last = 0;
		Map<String, Long> lastTimestampsPerModel = new HashMap<>();
		Map<String, Set<Double>> scoresPerModel = new HashMap<>();
		for (String line : lines) {
			String[] logParts = line.split(" ");

			/* parse a time stamp */
			if (logParts[0].length() == 12) {
				try {
					Date date = sdf.parse(logParts[0].substring(0, 10));
					long millis = date.getTime();
					if (timestamp_start < 0) {
						timestamp_start = millis;
					}
					timestamp_last = millis;
				}
				catch (ParseException e) {

				}
			}
			if (line.contains("autoweka.ClassifierRunner._run")) {
				String[] subParts = line.split("autoweka.ClassifierRunner._run \\(autoweka.ClassifierRunner:318\\)] - ");
				if (subParts.length > 1) {
					String rightPart = subParts[1];
					String[] resultParts = rightPart.split(";");
					String model = resultParts[0] + ";" + resultParts[1] + ";" +resultParts[2] + ";" + resultParts[3] + ";" + resultParts[4] + ";" + resultParts[5];
					double score = Double.valueOf(resultParts[resultParts.length - 1]);
					//				System.out.println(model + ": "  + score);
					lastTimestampsPerModel.put(model, timestamp_last);
					scoresPerModel.computeIfAbsent(model, m -> new HashSet<>()).add(score);
				}
			}
			//			System.out.println(line);
		}

		Map<Integer, Double> history = new HashMap<>();
		for (String key : scoresPerModel.keySet()) {
			int timeToSolution = (int)(lastTimestampsPerModel.get(key) - timestamp_start);
			double meanScore = scoresPerModel.get(key).stream().mapToDouble(e -> e).average().getAsDouble();
			history.put(timeToSolution, meanScore);
		}
		return history;
	}

	public static void main(final String[] args) {
		AutoWEKAClassifierWrapper w = new AutoWEKAClassifierWrapper();
		try {
			w.getHistory();
		} catch (Exception e) {
			e.printStackTrace();
		}

	}
}
