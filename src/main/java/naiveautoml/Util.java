package naiveautoml;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import ai.libs.jaicore.components.api.IComponentInstance;
import ai.libs.jaicore.components.api.IComponentRepository;
import ai.libs.jaicore.components.model.ComponentUtil;
import ai.libs.jaicore.ml.core.dataset.serialization.OpenMLDatasetReader;
import ai.libs.jaicore.ml.weka.classification.learner.IWekaClassifier;
import ai.libs.jaicore.ml.weka.classification.learner.WekaClassifier;
import ai.libs.mlplan.weka.MLPlanWekaBuilder;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.SingleClassifierEnhancer;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.trees.RandomForest;

public class Util {

	private static IComponentRepository components;

	static {
		try {
			components = MLPlanWekaBuilder.forClassification().withDataset(OpenMLDatasetReader.deserializeDataset(30)).build().getHASCO().getInput().getComponents();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public static IWekaClassifier getClassifierFromDescription(final PipelineDescription desc) {
		/* compile solution classifier */
		try {
			Classifier baseClassifier = getClassifier(desc.getBaseLearner(), desc.getBaseLearnerParams()).getClassifier();
			SingleClassifierEnhancer metaClassifier = desc.getMetaLearner() != null ? (SingleClassifierEnhancer)AbstractClassifier.forName(desc.getMetaLearner(), desc.getMetaLearnerParams()) : null;
			if (metaClassifier != null) {
				metaClassifier.setClassifier(baseClassifier);
			}
			return new WekaClassifier(metaClassifier != null ? metaClassifier : baseClassifier);
		}
		catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	public static PipelineDescription getDescriptionFromComponentInstance(final IComponentInstance ci) {
		return new PipelineDescription(null, ci.getComponent().getName(), false);
	}

	public static IComponentInstance getComponentInstanceFromDescription(final PipelineDescription pl) {
		return ComponentUtil.getDefaultParameterizationOfComponent(components.getComponent(pl.getBaseLearner()));
	}

	public static boolean hasCheapDefaultClassifier(final String name) throws Exception {
		if (name.equals("weka.classifiers.trees.RandomForest")) {
			return true;
		}
		if (name.startsWith(SMO.class.getName())) {
			return true;
		}
		if (name.startsWith(MultilayerPerceptron.class.getName())) {
			return true;
		}
		return false;
	}

	public static WekaClassifier getCheapDefaultClassifier(final String name) throws Exception {

		/* for random forests, use a lower number of trees */
		if (name.equals("weka.classifiers.trees.RandomForest")) {
			RandomForest rf = new RandomForest();
			rf.setOptions(new String[] {"-I", "10"});
			return new WekaClassifier(rf);
		}

		/* for SVMs, factorize the kernel and the complexity parameter */
		if (name.startsWith("weka.classifiers.functions.SMO")) {
			SMO smo = new SMO();
			String[] vals = name.split("-");
			smo.setOptions(new String[] {"-C", vals[2], "-L", "0.1", "-P", "0.1"});
			smo.setKernel(Kernel.forName(vals[1], null));
			return new WekaClassifier(smo);
		}

		if (name.startsWith(MultilayerPerceptron.class.getName())) {
			MultilayerPerceptron ann = new MultilayerPerceptron();
			ann.setOptions(new String[] {"-N", "50"});
			return new WekaClassifier(ann);
		}

		return new WekaClassifier(AbstractClassifier.forName(name, null));
	}

	public static WekaClassifier getDefaultClassifier(final String name) throws Exception {

		/* for SVMs, factorize the kernel and the complexity parameter */
		if (name.startsWith("weka.classifiers.functions.SMO")) {
			SMO smo = new SMO();
			String[] vals = name.split("-");
			smo.setOptions(new String[] {"-C", vals[2]});
			smo.setKernel(Kernel.forName(vals[1], null));
			return new WekaClassifier(smo);
		}

		if (name.startsWith(MultilayerPerceptron.class.getName())) {
			MultilayerPerceptron ann = new MultilayerPerceptron();
			ann.setOptions(new String[] {"-N", "50"});
			return new WekaClassifier(ann);
		}

		return new WekaClassifier(AbstractClassifier.forName(name, null));
	}

	public static WekaClassifier getClassifier(final String name, final String[] options) throws Exception {

		String[] optionsCopy = options == null ? null : Arrays.copyOf(options, options.length);

		/* for SVMs, factorize the kernel and the complexity parameter */
		if (name.startsWith("weka.classifiers.functions.SMO") && !name.equals("weka.classifiers.functions.SMO")) {
			SMO smo = new SMO();
			String[] vals = name.split("-");
			smo.setOptions(optionsCopy );
			smo.setKernel(Kernel.forName(vals[1], null));
			return new WekaClassifier(smo);
		}

		return new WekaClassifier(AbstractClassifier.forName(name, optionsCopy));
	}

	public static double getBestExpectedScoreAtPointUnderConvexityAssumption(final Map<Integer, Double> learningCurveEvaluations, final int pointOfInterest, final double leastAssumedImprovementPerInstance) {
		List<Integer> anchorPoints = new ArrayList<>();
		List<Double> scores = new ArrayList<>();
		learningCurveEvaluations.keySet().stream().sorted(Integer::compare).forEach(k -> {
			anchorPoints.add(k);
			scores.add(learningCurveEvaluations.get(k));
		});
		return getBestExpectedScoreAtPointUnderConvexityAssumption(anchorPoints, scores, pointOfInterest, leastAssumedImprovementPerInstance);
	}

	public static double getBestExpectedScoreAtPointUnderConvexityAssumption(final List<Integer> anchorPoints, final List<Double> learningCurveEvaluations, final int pointOfInterest, final double leastAssumedImprovementPerInstance) {

		int v = anchorPoints.get(learningCurveEvaluations.size() - 1);
		if (pointOfInterest < v) {
			throw new IllegalArgumentException("Cannot estimate score for anchor point "  + pointOfInterest + " smaller than the biggest known anchor point " + v);
		}
		//		double b = pointOfInterest * 1.0 / v;
		double fv = learningCurveEvaluations.get(learningCurveEvaluations.size() - 1);
		double biggestSlope = -1;
		for (int i = 0; i < learningCurveEvaluations.size() - 1; i++) {
			for (int j = 0; j < i; j++) {
				double trueSlopeBetweenPoints = (learningCurveEvaluations.get(i) - learningCurveEvaluations.get(j)) / (anchorPoints.get(i) - anchorPoints.get(j));
				double slope = Math.min(-1 * leastAssumedImprovementPerInstance, trueSlopeBetweenPoints);
				biggestSlope = Math.max(biggestSlope, slope);
			}
		}
		double bestBoundOnImprovement = biggestSlope * (pointOfInterest - v) * -1;
		double highestLowerBoundForFOfPointOfInterest = fv - bestBoundOnImprovement;
		//		double highestLowerBoundForFOfPointOfInterest = 0;
		//		for (int i = 0; i < learningCurveEvaluations.size() - 1; i++) {
		//			int u = ANCHOR_POINTS.get(i);
		//			double a = v * 1.0 / u;
		//			double fu = learningCurveEvaluations.get(i);
		//			double delta = Math.max(0.01, fu - fv); // at least on percentage point improvement assumed
		//			double expectedImprovementOverFV = (delta * a * (b - 1) / (a - 1));
		//			double lowerBoundOnfOfPointOfInterest =  fv - expectedImprovementOverFV;
		//			logger.info("a = " + a + ", b = " + b + ", delta = " + delta + ", ei = " + expectedImprovementOverFV + ", lower bound = " + lowerBoundOnfOfPointOfInterest);
		//			highestLowerBoundForFOfPointOfInterest = Math.max(highestLowerBoundForFOfPointOfInterest, lowerBoundOnfOfPointOfInterest);
		//		}
		return highestLowerBoundForFOfPointOfInterest;
	}

	public static List<Integer> getAttributeList(final int[] attributeRanking, final int n) {
		List<Integer> attributes = new ArrayList<>();
		for (int i = 0; i < n; i++) {
			attributes.add(attributeRanking[i]);
		}
		return attributes;
	}

	public static List<String> blackListedParameters(final String baseLearner) {
		if (baseLearner.startsWith(SMO.class.getName())) {
			return Arrays.asList("C");
		}
		return Arrays.asList();
	}
}
