package naiveautoml;

import java.util.concurrent.TimeUnit;

import org.api4.java.algorithm.Timeout;

import ai.libs.jaicore.ml.core.dataset.serialization.OpenMLDatasetReader;
import ai.libs.jaicore.ml.weka.dataset.WekaInstances;
import ai.libs.jaicore.timing.TimedComputation;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.functions.supportVector.Puk;
import weka.core.Instances;

public class RFTest {

	public static void main(final String[] args) throws Exception {

		int openmlid = 4541; //4538;// GesturePhaseSegmentationProcessed
		Instances dsOrig = new WekaInstances(OpenMLDatasetReader.deserializeDataset(openmlid)).getInstances();


		SMO smo = new SMO();

		smo.setOptions(new String[] {"-C", "1.0", "-L", "0.001", "-P", "1.0E-12", "-N", "0", "-V", "-1", "-W", "1", "-do-not-check-capabilities"});
		Logistic l = new Logistic();
		l.setOptions(new String[] {"-R", "1.0E-8", "-M", "-1", "-num-decimal-places", "4"});
		smo.setCalibrator(l);
		Kernel k = new Puk();
		k.setOptions(new String[] {"-O", "1.0", "-S", "1.0", "-C", "250007"});
		smo.setKernel(k);
		TimedComputation.compute(() -> {
			smo.buildClassifier(dsOrig);
			return null;
		}, new Timeout(1, TimeUnit.MINUTES), "msg");
	}
}
