package naiveautoml;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

public class FilteringResult {

	private final Map<String, int[]> attributeRankings;
	private final Map<String, Integer> locallyBestNumberOfFeaturesPerPreprocessor;
	private final Map<String, List<Double>> pilotPerformanceHistories;
	private final Map<String, Double> bestPilotPerformancePerPreProcessor;

	public FilteringResult(final Map<String, int[]> attributeRankings, final Map<String, Integer> locallyBestNumberOfFeaturesPerPreprocessor, final Map<String, List<Double>> pilotPerformanceHistories,
			final Map<String, Double> bestPilotPerformancePerPreProcessor) {
		super();
		this.attributeRankings = attributeRankings;
		this.locallyBestNumberOfFeaturesPerPreprocessor = locallyBestNumberOfFeaturesPerPreprocessor;
		this.pilotPerformanceHistories = pilotPerformanceHistories;
		this.bestPilotPerformancePerPreProcessor = bestPilotPerformancePerPreProcessor;
	}

	public Map<String, int[]> getAttributeRankings() {
		return this.attributeRankings;
	}

	public Map<String, Integer> getLocallyBestNumberOfFeaturesPerPreprocessor() {
		return this.locallyBestNumberOfFeaturesPerPreprocessor;
	}

	public Map<String, Double> getPilotPerformancePerPreProcessor() {
		return this.bestPilotPerformancePerPreProcessor;
	}

	public Map<String, List<Double>> getPilotPerformanceHistories() {
		return this.pilotPerformanceHistories;
	}

	public Map<String, List<Integer>> getAttributesChosenByFilters() {
		Map<String, List<Integer>> map = new HashMap<>();
		for (Entry<String, int[]> entry : this.attributeRankings.entrySet()) {
			map.put(entry.getKey(), Util.getAttributeList(entry.getValue(), this.locallyBestNumberOfFeaturesPerPreprocessor.get(entry.getKey())));
		}
		return map;
	}

	public List<Integer> getAttributSetWithBestScore() {

		/* get default attribute set */
		String chosenPreprocessor = this.bestPilotPerformancePerPreProcessor.entrySet().stream().min((e1, e2) -> Double.compare(e1.getValue(), e2.getValue())).get().getKey();
		return Util.getAttributeList(this.attributeRankings.get(chosenPreprocessor), this.locallyBestNumberOfFeaturesPerPreprocessor.get(chosenPreprocessor));
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((this.attributeRankings == null) ? 0 : this.attributeRankings.hashCode());
		result = prime * result + ((this.locallyBestNumberOfFeaturesPerPreprocessor == null) ? 0 : this.locallyBestNumberOfFeaturesPerPreprocessor.hashCode());
		result = prime * result + ((this.bestPilotPerformancePerPreProcessor == null) ? 0 : this.bestPilotPerformancePerPreProcessor.hashCode());
		return result;
	}

	@Override
	public boolean equals(final Object obj) {
		if (this == obj) {
			return true;
		}
		if (obj == null) {
			return false;
		}
		if (this.getClass() != obj.getClass()) {
			return false;
		}
		FilteringResult other = (FilteringResult) obj;
		if (this.attributeRankings == null) {
			if (other.attributeRankings != null) {
				return false;
			}
		} else if (!this.attributeRankings.equals(other.attributeRankings)) {
			return false;
		}
		if (this.locallyBestNumberOfFeaturesPerPreprocessor == null) {
			if (other.locallyBestNumberOfFeaturesPerPreprocessor != null) {
				return false;
			}
		} else if (!this.locallyBestNumberOfFeaturesPerPreprocessor.equals(other.locallyBestNumberOfFeaturesPerPreprocessor)) {
			return false;
		}
		if (this.bestPilotPerformancePerPreProcessor == null) {
			if (other.bestPilotPerformancePerPreProcessor != null) {
				return false;
			}
		} else if (!this.bestPilotPerformancePerPreProcessor.equals(other.bestPilotPerformancePerPreProcessor)) {
			return false;
		}
		return true;
	}

}
