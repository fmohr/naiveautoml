package naiveautoml.legitimacy;

import java.io.File;

import org.api4.java.algorithm.exceptions.AlgorithmExecutionCanceledException;
import org.api4.java.algorithm.exceptions.AlgorithmTimeoutedException;

import ai.libs.jaicore.experiments.ExperimenterFrontend;
import ai.libs.jaicore.experiments.exceptions.ExperimentAlreadyExistsInDatabaseException;
import ai.libs.jaicore.experiments.exceptions.ExperimentDBInteractionFailedException;
import ai.libs.jaicore.experiments.exceptions.IllegalExperimentSetupException;

public class LegitimacyDatabaseSetup {

	public static void main(final String[] args) throws AlgorithmTimeoutedException, ExperimentDBInteractionFailedException, IllegalExperimentSetupException, ExperimentAlreadyExistsInDatabaseException, InterruptedException, AlgorithmExecutionCanceledException {
		/* setup experimenter frontend */
		ExperimenterFrontend fe = new ExperimenterFrontend().withEvaluator(new LegitimacyExperimentRunner()).withExperimentsConfig(new File("conf/experiments-legitimacy.conf")).withDatabaseConfig(new File("conf/database-local-legitimacy.conf"));
		fe.setLoggerName("frontend");
		fe.synchronizeDatabase();
	}
}
