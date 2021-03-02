package naiveautoml.ga;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map.Entry;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import org.api4.java.algorithm.Timeout;
import org.moeaframework.algorithm.single.GeneticAlgorithm;
import org.moeaframework.algorithm.single.MinMaxDominanceComparator;
import org.moeaframework.core.Initialization;
import org.moeaframework.core.Solution;
import org.moeaframework.core.operator.CompoundVariation;
import org.moeaframework.core.operator.InjectedInitialization;
import org.moeaframework.core.operator.OnePointCrossover;
import org.moeaframework.core.operator.TournamentSelection;
import org.moeaframework.core.operator.binary.BitFlip;

import ai.libs.jaicore.ml.core.dataset.serialization.OpenMLDatasetReader;
import ai.libs.jaicore.ml.weka.dataset.IWekaInstances;
import ai.libs.jaicore.ml.weka.dataset.WekaInstances;
import naiveautoml.EvaluationReports;
import naiveautoml.FilteringResult;
import naiveautoml.NaiveAutoMLBuilder;
import naiveautoml.PipelineDescription;
import naiveautoml.RacePool;
import weka.classifiers.lazy.IBk;

public class FeatureSetOptimizer {

	private final IWekaInstances data;
	private final FilteringResult filteringResult;

	public FeatureSetOptimizer(final IWekaInstances data, final FilteringResult filteringResult) {
		super();
		this.data = data;
		this.filteringResult = filteringResult;
	}

	public List<Entry<PipelineDescription, EvaluationReports>> wrapperOptimization(final PipelineDescription basis, final Random random, final Timeout to) throws InterruptedException {
		RacePool pool = new RacePool(this.data, Arrays.asList(100, 200, 500, 1000));
		FeatureSetOptimizationProblem problem = new FeatureSetOptimizationProblem(basis, this.data.getNumAttributes(), pool);
		int populationSize = 20;
		List<Solution> initPopulation = new ArrayList<>();
		long deadline = System.currentTimeMillis() + to.milliseconds();
		for (List<Integer> attributeSet : this.filteringResult.getAttributesChosenByFilters().values()) {
			initPopulation.add(problem.getSolutionFromEncoding(attributeSet));
		}
		for (int i = 0; i < populationSize; i ++) {
			List<Integer> cand = new ArrayList<>();
			for (int j = 0; j < 10; j++) {
				cand.add(random.nextInt(problem.getNumberOfVariables()));
			}
			initPopulation.add(problem.getSolutionFromEncoding(cand));
		}
		Initialization initialization = new InjectedInitialization(problem, populationSize, initPopulation);
		GeneticAlgorithm ga = new GeneticAlgorithm(problem, new MinMaxDominanceComparator(), initialization, new TournamentSelection(populationSize), new CompoundVariation(new OnePointCrossover(1), new BitFlip(1.0 / problem.getNumberOfVariables())));
		int iterationsSinceLastImprovement = 0;
		double bestScore = 1;
		int i = 0;
		while (iterationsSinceLastImprovement < 20 && System.currentTimeMillis() < deadline) {
			System.out.println("ITERATION " + (++i));
			ga.step();
			double score = ga.getResult().get(0).getObjective(0);
			if (score < bestScore) {
				System.out.println("new best score: " + score);
				bestScore = score;
				iterationsSinceLastImprovement = 0;
			}
			else {
				iterationsSinceLastImprovement ++;
			}
		}
		pool.closeAndAwaitTermination();
		return pool.getBestCandidates(5);
	}




	public static void main(final String[] args) throws Exception {

		Random random = new Random(0);
		IWekaInstances instances = new WekaInstances(OpenMLDatasetReader.deserializeDataset(1485));



		PipelineDescription pl = new PipelineDescription(IBk.class.getName());

		FeatureSetOptimizer opt = new FeatureSetOptimizer(instances, new NaiveAutoMLBuilder().build().getFilterRankings(instances, random));
		opt.wrapperOptimization(pl, random, new Timeout(1, TimeUnit.MINUTES)).forEach(e -> System.out.println(e.getKey() + ": " + e.getValue().getBestAverageScoreSeen()));


	}
}
