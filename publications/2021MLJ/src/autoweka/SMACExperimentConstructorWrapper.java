package naiveautoml.autoweka;

import autoweka.Experiment;
import autoweka.smac.SMACExperimentConstructor;

public class SMACExperimentConstructorWrapper extends SMACExperimentConstructor {

	public String getBaseDir() {
		return this.mParamBaseDir;
	}

	public void setExperiment(final Experiment e) {
		this.mExperiment = e;
	}
}
