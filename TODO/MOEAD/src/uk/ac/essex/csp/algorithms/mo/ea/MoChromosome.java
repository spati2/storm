package uk.ac.essex.csp.algorithms.mo.ea;

import org.apache.commons.math.random.RandomData;
import org.apache.commons.math.random.RandomGenerator;

public abstract class MoChromosome {
	
	public static final double EPS = 1.2e-7;

	private static final long serialVersionUID = 1L;

	public double[][] domainInfo;

	public int parDimension;

	public int objectDimension;

	public double[] objectivesValue;

	public double[] objectivesEI;

	public double[] estimatedObjectiveDevitation;

	public double[] estimatedObjectiveValue;

	public double fitnessValue;

	public boolean selected;

	public int liveGeneration;

	public boolean evaluated;

	public double crdistance;

	protected Object clone() throws CloneNotSupportedException {
		throw new CloneNotSupportedException(
				"Chromosome cannot be directlyCannot be cloned, it must be created from a pool");
	}

	public void copyTo(MoChromosome copyto) {
		copyto.domainInfo = this.domainInfo;
		copyto.evaluated = this.evaluated;
		copyto.fitnessValue = this.fitnessValue;
		System.arraycopy(this.objectivesValue, 0, copyto.objectivesValue, 0,
				this.objectivesValue.length);
	}

	public void randomize(RandomData randomGenerator) {
		randomizeParameter(randomGenerator);
		for (int i = 0; i < objectDimension; i++) {
			objectivesValue[i] = 0;
			objectivesEI[i] = 0;
			estimatedObjectiveDevitation[i] = 0;
			estimatedObjectiveValue[i] = 0;
		}
		crdistance = 0;
		evaluated = false;
		fitnessValue = 0;
		liveGeneration = 0;
		selected = false;
	};

	protected abstract void randomizeParameter(RandomData randomGenerator);
	
	public abstract double parameterDistance(MoChromosome another);

	public static double objectiveDistance(MoChromosome ch1, MoChromosome ch2) {
		double sum = 0;
		for (int i = 0; i < ch1.objectivesValue.length; i++) {
			sum += Math.pow(ch1.objectivesValue[i] - ch2.objectivesValue[i], 2);
		}
		return Math.sqrt(sum);
	}
	
	public abstract void mutate(RandomGenerator rg, double mutationrate);

	public abstract void diff_xover(MoChromosome ind0, MoChromosome ind1,
			MoChromosome ind2, RandomData randomData);

	public abstract void crossover(MoChromosome ind0, MoChromosome ind1,
			RandomGenerator randomData);

	public abstract String vectorString();
	
	public abstract String getParameterString();
}
