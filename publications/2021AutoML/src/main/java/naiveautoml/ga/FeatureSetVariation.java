package naiveautoml.ga;

import java.util.Random;

import org.moeaframework.core.Solution;
import org.moeaframework.core.Variation;
import org.moeaframework.core.variable.BinaryVariable;

public class FeatureSetVariation implements Variation {

	private final Random random;

	public FeatureSetVariation(final Random random) {
		super();
		this.random = random;
	}

	@Override
	public int getArity() {
		return 2;
	}

	@Override
	public Solution[] evolve(final Solution[] parents) {
		int n = parents[0].getNumberOfVariables();
		System.out.println("Evolve " + parents.length + " parents.");

		/* create probability vector */
		double[] probabilities = new double[n];
		for (int i = 0; i < n; i++) {
			BinaryVariable v1 = (BinaryVariable) parents[0].getVariable(i);
			BinaryVariable v2 = (BinaryVariable) parents[1].getVariable(i);
			if (v1.get(0) && v2.get(0)) {
				probabilities[i] = 0.9;
			}
			if (v1.get(0) ^ v2.get(0)) { // exactly one of the two is true
				probabilities[i] = 0.5;
			} else {
				probabilities[i] = 0.05;
			}
		}

		/* now draw children based on the probability vector */
		int numKids = 5;
		Solution[] children = new Solution[numKids];
		children[0] = parents[0];
		children[1] = parents[1];
		for (int i = 2; i < numKids; i++) {
			Solution s = new Solution(n, 1);
			for (int j = 0; j < n; j++) {
				BinaryVariable v = new BinaryVariable(1);
				v.set(0, this.random.nextDouble() <= probabilities[j]);
				s.setVariable(j, v);
			}
			children[i] = s;
		}
		return children;
	}

}
