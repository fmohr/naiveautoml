package naiveautoml;

import java.util.Arrays;
import java.util.Collection;

public class PipelineDescription {

	private final boolean cheapVersion;
	private final Collection<Integer> attributes;
	private final String baseLearner;
	private final String[] baseLearnerParams;
	private final String metaLearner;
	private final String[] metaLearnerParams;

	public PipelineDescription(final String baseLearner) {
		this(null, baseLearner);
	}

	public PipelineDescription(final Collection<Integer> attributes, final String baseLearner) {
		this(attributes, baseLearner, true);
	}

	public PipelineDescription(final Collection<Integer> attributes, final String baseLearner, final boolean cheapVersion) {
		this(attributes, baseLearner, null, cheapVersion, null, null);
	}

	public PipelineDescription(final Collection<Integer> attributes, final String baseLearner, final String[] baseLearnerParams, final String metaLearner, final String[] metaLearnerParams) {
		this(attributes, baseLearner, baseLearnerParams, false, metaLearner, metaLearnerParams);
	}

	private PipelineDescription(final Collection<Integer> attributes, final String baseLearner, final String[] baseLearnerParams, final boolean cheapVersion, final String metaLearner, final String[] metaLearnerParams) {
		super();
		this.attributes = attributes;
		this.baseLearner = baseLearner;
		this.baseLearnerParams = baseLearnerParams;
		this.cheapVersion = cheapVersion;
		this.metaLearner = metaLearner;
		this.metaLearnerParams = metaLearnerParams;
	}

	public Collection<Integer> getAttributes() {
		return this.attributes;
	}

	public String getBaseLearner() {
		return this.baseLearner;
	}

	public String getMetaLearner() {
		return this.metaLearner;
	}

	public String[] getBaseLearnerParams() {
		return this.baseLearnerParams;
	}

	public String[] getMetaLearnerParams() {
		return this.metaLearnerParams;
	}

	public boolean isCheapVersion() {
		return this.cheapVersion;
	}

	public PipelineDescription getCheapVersion() {
		return new PipelineDescription(this.attributes, this.baseLearner, this.baseLearnerParams, true, this.metaLearner, this.metaLearnerParams);
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((this.attributes == null) ? 0 : this.attributes.hashCode());
		result = prime * result + ((this.baseLearner == null) ? 0 : this.baseLearner.hashCode());
		result = prime * result + Arrays.hashCode(this.baseLearnerParams);
		result = prime * result + (this.cheapVersion ? 1231 : 1237);
		result = prime * result + ((this.metaLearner == null) ? 0 : this.metaLearner.hashCode());
		result = prime * result + Arrays.hashCode(this.metaLearnerParams);
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
		PipelineDescription other = (PipelineDescription) obj;
		if (this.attributes == null) {
			if (other.attributes != null) {
				return false;
			}
		} else if (!this.attributes.equals(other.attributes)) {
			return false;
		}
		if (this.baseLearner == null) {
			if (other.baseLearner != null) {
				return false;
			}
		} else if (!this.baseLearner.equals(other.baseLearner)) {
			return false;
		}
		if (!Arrays.equals(this.baseLearnerParams, other.baseLearnerParams)) {
			return false;
		}
		if (this.cheapVersion != other.cheapVersion) {
			return false;
		}
		if (this.metaLearner == null) {
			if (other.metaLearner != null) {
				return false;
			}
		} else if (!this.metaLearner.equals(other.metaLearner)) {
			return false;
		}
		if (!Arrays.equals(this.metaLearnerParams, other.metaLearnerParams)) {
			return false;
		}
		return true;
	}

	@Override
	public String toString() {
		return "PipelineDescription [cheapVersion=" + this.cheapVersion + ", attributes=" + this.attributes + ", baseLearner=" + this.baseLearner + ", baseLearnerParams=" + Arrays.toString(this.baseLearnerParams) + ", metaLearner="
				+ this.metaLearner + ", metaLearnerParams=" + Arrays.toString(this.metaLearnerParams) + "]";
	}
}
