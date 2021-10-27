package naiveautoml;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

public class EvaluationReports extends ArrayList<EvaluationReport> {

	public Set<Integer> getCoveredSizes() {
		return this.stream().map(r -> r.getNumInstances()).collect(Collectors.toSet());
	}

	public double getAverageScorePerSize(final int size) {
		return this.getReportsForTrainSize(size).stream().mapToDouble(r -> r.getScore()).average().getAsDouble();
	}

	public double getBestAverageScoreSeen() {
		double best = 1;
		for (int size : this.getCoveredSizes()) {
			best = Math.min(this.getAverageScorePerSize(size), best);
		}
		return best;
	}

	public int getBiggestSuccessfulTrainSize() {
		int max = 0;
		for (EvaluationReport r : this) {
			max = Math.max(max, r.getNumInstances());
		}
		return max;
	}

	public boolean hasEntryForTrainSize(final int trainSize) {
		return this.stream().anyMatch(r -> r.getNumInstances() == trainSize);
	}

	public EvaluationReports getReportsForTrainSize(final int trainSize) {
		EvaluationReports filteredReports = new EvaluationReports();
		filteredReports.addAll(this.stream().filter(r -> r.getNumInstances() == trainSize).collect(Collectors.toList()));
		return filteredReports;
	}

	public boolean hasFailedReportForSmallerSize(final int size) {
		return this.stream().anyMatch(r -> r.getNumInstances() <= size && r.isFailed());
	}

	public Map<Integer, Double> getLearningCurveAsMap() {
		Map<Integer, Double> map = new HashMap<>();
		for (int size : this.getCoveredSizes()) {
			map.put(size, this.getAverageScorePerSize(size));
		}
		return map;
	}
}
