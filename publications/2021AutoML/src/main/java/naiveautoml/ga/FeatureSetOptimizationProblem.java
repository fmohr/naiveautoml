package naiveautoml.ga;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Future;

import org.moeaframework.core.Solution;
import org.moeaframework.core.variable.BinaryVariable;
import org.moeaframework.problem.AbstractProblem;

import naiveautoml.EvaluationReports;
import naiveautoml.PipelineDescription;
import naiveautoml.RacePool;

public class FeatureSetOptimizationProblem extends AbstractProblem {

	private final PipelineDescription pl;
	private final int numAttributes;
	private final RacePool pool;
	private final Map<List<Integer>, Double> cache = new HashMap<>();

	public FeatureSetOptimizationProblem(final PipelineDescription pl, final int numAttributes, final RacePool pool) {
		super(numAttributes, 1);
		this.pl = pl;
		this.numAttributes = numAttributes;
		this.pool = pool;
	}

	public List<Integer> getPositiveEntries(final Solution sol) {
		List<Integer> active = new ArrayList<>();
		int n = sol.getNumberOfVariables();
		for (int i = 0; i < n; i++) {
			if (((BinaryVariable) sol.getVariable(i)).get(0)) {
				active.add(i);
			}
		}
		return active;
	}

	@Override
	public void evaluate(final Solution solution) {
		try {
			List<Integer> attributes = this.getPositiveEntries(solution);
			if (!this.cache.containsKey(attributes)) {
				PipelineDescription candidate = new PipelineDescription(attributes, this.pl.getBaseLearner(), this.pl.getBaseLearnerParams(), this.pl.getMetaLearner(), this.pl.getMetaLearnerParams());
				Future<EvaluationReports> f = this.pool.submitCandidate(candidate);
				double score = f.get().getBestAverageScoreSeen();
				this.cache.put(attributes, score);
			}
			double score = this.cache.get(attributes);
			solution.setObjective(0, score);
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	@Override
	public Solution newSolution() {
		Solution s = new Solution(this.numAttributes, 1);
		for (int i = 0; i < this.numAttributes; i++) {
			s.setVariable(i, new BinaryVariable(1));
		}
		return s;
	}

	public Solution getSolutionFromEncoding(final int... code) {
		List<Integer> l = new ArrayList<>();
		for (int i : code) {
			l.add(i);
		}
		return this.getSolutionFromEncoding(l);
	}

	public Solution getSolutionFromEncoding(final List<Integer> code) {
		Solution s = this.newSolution();
		for (int i : code) {
			((BinaryVariable) s.getVariable(i)).set(0, true);
		}
		return s;
	}

}
