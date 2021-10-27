package naiveautoml;

public class NaiveAutoMLBuilder {

	private boolean preprocessing;
	private boolean consolidation;
	private boolean metalearners;
	private boolean wrapping;
	private boolean parametertuning;
	private boolean finalselection;
	private boolean iterativeEvaluation;

	public NaiveAutoMLBuilder withPreprocessing() {
		this.preprocessing = true;
		return this;
	}

	public NaiveAutoMLBuilder withConsolidationPhase() {
		this.consolidation = true;
		return this;
	}

	public NaiveAutoMLBuilder withMetaLearners() {
		this.metalearners = true;
		return this;
	}

	public NaiveAutoMLBuilder withParameterTuning() {
		this.parametertuning = true;
		return this;
	}

	public NaiveAutoMLBuilder withWrapping() {
		this.wrapping = true;
		return this;
	}

	public NaiveAutoMLBuilder withFinalSelectionPhase() {
		this.finalselection = true;
		return this;
	}

	public NaiveAutoMLBuilder withIterativeEvaluations() {
		this.iterativeEvaluation = true;
		return this;
	}

	public NaiveAutoML build() {
		if (this.wrapping && !this.preprocessing) {
			throw new IllegalStateException("Wrapping has been enabled but pre-processing not. This is not supported currently.");
		}
		return new NaiveAutoML(this.preprocessing, this.consolidation, this.metalearners, this.wrapping, this.parametertuning, this.finalselection, this.iterativeEvaluation);
	}
}
