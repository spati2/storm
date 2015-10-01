package uk.ac.essex.csp.algorithms.mo.ea;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.commons.math.random.RandomData;
import org.apache.commons.math.random.RandomDataImpl;
import org.apache.commons.math.random.RandomGenerator;
import org.apache.commons.pool.PoolableObjectFactory;
import org.apache.commons.pool.impl.GenericObjectPool;

import uk.ac.essex.csp.algorithms.mo.DominationDeterminer;
import uk.ac.essex.csp.algorithms.mo.MultiObjectiveProblem;
import uk.ac.essex.csp.algorithms.mo.MultiObjectiveSolver;
import uk.ac.essex.csp.algorithms.random.RanMT;

public abstract class MoeaGenotype implements MultiObjectiveSolver {

	protected GenotypeEventManager eventManager = new GenotypeEventManager();

	public static final String Property_Population_Size = "population_size";

	protected MultiObjectiveProblem mop;

	protected GenericObjectPool chromosomePool;

	protected RandomDataImpl randomData;
	protected RandomGenerator randomGenerator;

	protected int EvalCounter = 0;

	// The iteration counter
	public int ItrCounter;

	protected Configurator config;

	protected int numObjectives;
	protected int numParameters;

	// indicate terminated by another thread in the middle of process.
	protected boolean stoped = false;

	public Set<MoChromosome> archive = new HashSet<MoChromosome>();

	public RandomData getRandomData() {
		return randomData;
	}

	public RandomGenerator getRandomGenerator() {
		return randomGenerator;
	}

	public void reSeedRandom(long long1) {
		this.randomData.reSeed(long1);
	}

	public Configurator getConfiguration() {
		return config;
	}

	/**
	 * To terminate the processing after the current generation.
	 */
	public void stop() {
		this.stoped = true;
	}

	/**
	 * To indicate if the processing has been stopped.
	 * 
	 * @return
	 */
	public boolean stopped() {
		return this.stoped;
	}

	public GenotypeEventManager getEventManager() {
		return this.eventManager;
	}

	public MoeaGenotype() {
		config = createConfigurator();
		// this.randomGenerator = new JDKRandomGenerator();
		// this.randomGenerator = new CustomRandomGenerator();
		this.randomGenerator = new RanMT();
		this.randomData = new RandomDataImpl(this.randomGenerator);
		chromosomePool = new GenericObjectPool();
		chromosomePool
				.setWhenExhaustedAction(GenericObjectPool.WHEN_EXHAUSTED_GROW);
		chromosomePool.setFactory(new MoChromosomePoolFactory());
	}

	protected Configurator createConfigurator() {
		return new Configurator();
	}

	public MultiObjectiveProblem getMultiObjectiveProblem() {
		if (mop == null)
			throw new IllegalStateException(
					"A Multi-Objective Problem has not been set for the Genotype yet.");
		else
			return mop;
	}

	public void setMultiObjectiveProblem(MultiObjectiveProblem problem) {
		this.mop = problem;
	}

	public MoPopulation createPopulation(int size) {
		MoPopulation pop = new MoPopulation(this, size);
		return pop;
	}

	public MoChromosome createChromosome() {
		Object object = null;
		try {
			object = this.chromosomePool.borrowObject();
			return (MoChromosome) object;
		} catch (Exception e) {
			e.printStackTrace();
			return generateRandomChromosome();
		}
	}

	public void destroyChromosome(MoChromosome chromosome) {
		try {
			this.chromosomePool.returnObject(chromosome);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private class MoChromosomePoolFactory implements PoolableObjectFactory {

		public void activateObject(Object obj) throws Exception {
		}

		public void destroyObject(Object obj) throws Exception {
			MoChromosome chromosome = (MoChromosome) obj;
			chromosome.randomize(randomData);
		}

		public Object makeObject() throws Exception {
			return mop.createRandomMoChromosome(randomData);
		}

		public void passivateObject(Object obj) throws Exception {
		}

		public boolean validateObject(Object obj) {
			return true;
		}
	}

	private MoChromosome generateRandomChromosome() {
		return this.mop.createRandomMoChromosome(randomData);
	}

	public void evaluate(MoChromosome chromosmoe) {
		MultiObjectiveProblem multiObjectiveProblem = this
				.getMultiObjectiveProblem();
		multiObjectiveProblem.evaluate(chromosmoe);
		this.EvalCounter++;
	};

	public void evaluate(MoPopulation pop) {
		int size = pop.size();
		for (int i = 0; i < size; i++) {
			MoChromosome chromosome = pop.getChromosome(i);
			evaluate(chromosome);
		}
	}

	public final void solve(MultiObjectiveProblem problem) {
		this.setMultiObjectiveProblem(problem);
		this.numObjectives = problem.getObjectiveSpaceDimension();
		this.numParameters = problem.getParameterSpaceDimension();
		this.doSolve();
		this.stoped = true;
	}

	abstract protected void doSolve();

	public void reset() {
		this.stoped = false;
		this.ItrCounter = 0;
		this.EvalCounter = 0;
		randomData.reSeed();
	}

	// ========== Utility Methods ================================

	// public static double[] levelConvert(int[] p, int levels,
	// SquareDimentionalSpace domain) {
	// double[] result = new double[p.length];
	// for (int i = 0; i < result.length; i++) {
	// DoubleRange range = domain.getDimensionRange(i);
	// result[i] = SLHD.levelMaping(levels, p[i],
	// range.getMinimumDouble(), range.getMaximumDouble());
	// }
	// return result;
	// }

	public static void binaryTournamentSelection(MoPopulation pop,
			MoChromosome[] selected, RandomData random) {
		int size = pop.size();
		for (int i = 0; i < selected.length; i++) {
			int k = random.nextInt(0, size - 1);
			int j = random.nextInt(0, size - 1);
			MoChromosome chromosome1 = pop.getChromosome(k);
			MoChromosome chromosome2 = pop.getChromosome(j);
			if (chromosome2.fitnessValue > chromosome1.fitnessValue)
				selected[i] = chromosome2;
			else
				selected[i] = chromosome1;
		}
	}

	public static double distance(double[] weight1, double[] weight2) {
		double sum = 0;
		for (int i = 0; i < weight1.length; i++) {
			sum += Math.pow((weight1[i] - weight2[i]), 2);
		}
		return Math.sqrt(sum);
	}

	/**
	 * Non-domination Selection procedure to filter out the dominated
	 * individuals from the given collection.
	 * 
	 * @param collection
	 *            the collection of MoChromosome object need to be filtered.
	 * @return the reference copy of the non-dominating individuals
	 */
	public static List<MoChromosome> nonDominatingSelect(
			List<MoChromosome> collection, boolean toMin) {
		List<MoChromosome> result = new ArrayList<MoChromosome>();
		if (collection.size() == 0)
			return result;

		result.add(collection.get(0));
		int counter = 1;
		out: while (counter < collection.size()) {
			int jj = 0;
			MoChromosome chromosome2 = collection.get(counter);
			int resultsize = result.size();
			boolean remove[] = new boolean[resultsize];
			while (jj < resultsize) {
				MoChromosome chromosome1 = result.get(jj);
				if (DominationDeterminer.dominate(chromosome2.objectivesValue,
						chromosome1.objectivesValue, toMin)) {
					remove[jj] = true;
					// result.remove(jj);
				} else if (DominationDeterminer.dominate(
						chromosome1.objectivesValue,
						chromosome2.objectivesValue, toMin)) {
					counter++;
					continue out;
				}
				jj++;
			}
			for (int i = remove.length - 1; i >= 0; i--) {
				if (remove[i])
					result.remove(i);
			}
			result.add(chromosome2);
			counter++;
		}
		return result;
	}

	/**
	 * Determine if one MoChromosome dominates another.
	 * 
	 * @param chmo1
	 *            the first MoChromosome to compare with another.
	 * @param chmo2
	 *            the MoChromosome need to be compared.
	 * @param toMin
	 *            whether minimal is better or not.
	 * @return
	 */
	public static boolean dominate(MoChromosome chmo1, MoChromosome chmo2,
			boolean toMin) {
		return dominate(chmo1.objectivesValue, chmo2.objectivesValue, toMin);
	}

	public static boolean dominate(double[] oa, double[] ob, boolean toMin) {
		boolean a_is_worse = false;
		boolean equals = true;
		for (int i = 0; i < oa.length && !a_is_worse; i++) {
			double fitnessValueA = oa[i];
			double fitnessValueB = ob[i];
			a_is_worse = toMin ? (fitnessValueA > fitnessValueB)
					: (fitnessValueA < fitnessValueB);
			equals = (fitnessValueA == fitnessValueB) && equals;
		}
		return (!equals && !a_is_worse);
	}

	public static boolean equals(double[] oa, double[] ob) {
		boolean equals = true;
		for (int i = 0; i < oa.length; i++) {
			equals = (oa[i] == ob[i]) && equals;
		}
		return equals;
	}
}
