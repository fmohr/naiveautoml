package naiveautoml;

public class EvaluationReport {
	private final int numInstances;
	private final double score;
	private final int runtime;
	private final boolean failed;

	public EvaluationReport(final int numInstances, final double score, final int runtime, final boolean failed) {
		super();
		this.numInstances = numInstances;
		this.score = score;
		this.runtime = runtime;
		this.failed = failed;
	}

	public int getNumInstances() {
		return this.numInstances;
	}

	public double getScore() {
		return this.score;
	}

	public int getRuntime() {
		return this.runtime;
	}

	public boolean isFailed() {
		return this.failed;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + (this.failed ? 1231 : 1237);
		result = prime * result + this.numInstances;
		result = prime * result + this.runtime;
		long temp;
		temp = Double.doubleToLongBits(this.score);
		result = prime * result + (int) (temp ^ (temp >>> 32));
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
		EvaluationReport other = (EvaluationReport) obj;
		if (this.failed != other.failed) {
			return false;
		}
		if (this.numInstances != other.numInstances) {
			return false;
		}
		if (this.runtime != other.runtime) {
			return false;
		}
		if (Double.doubleToLongBits(this.score) != Double.doubleToLongBits(other.score)) {
			return false;
		}
		return true;
	}
}
