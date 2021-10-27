package naiveautoml;

import java.util.Map;

public class FinalResult {

	private final Map<String, Integer> runtimesPerStage;
	private final Map<Long, Double> resultHistory;
	private final PipelineDescription finalChoice;

	private final double estimatedOutOfSampleError;

	public FinalResult(final PipelineDescription finalChoice, final double estimatedOutOfSampleError, final Map<String, Integer> runtimesPerStage, final Map<Long, Double> resultHistory) {
		super();
		this.finalChoice = finalChoice;
		this.estimatedOutOfSampleError = estimatedOutOfSampleError;
		this.runtimesPerStage = runtimesPerStage;
		this.resultHistory = resultHistory;
	}

	public PipelineDescription getFinalChoice() {
		return this.finalChoice;
	}

	public double getEstimatedOutOfSampleError() {
		return this.estimatedOutOfSampleError;
	}

	public Map<String, Integer> getRuntimesPerStage() {
		return this.runtimesPerStage;
	}

	public Map<Long, Double> getResultHistory() {
		return this.resultHistory;
	}

}
