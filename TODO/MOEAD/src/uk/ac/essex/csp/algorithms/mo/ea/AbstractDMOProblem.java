package uk.ac.essex.csp.algorithms.mo.ea;

import org.apache.commons.math.random.RandomData;

public abstract class AbstractDMOProblem extends AbstractMOP {

	final protected int[] getRandomParameterPoint(RandomData randomObject) {
		double[][] domain = this.getDomain();
		int[] value = new int[this.parDimension];
		for (int i = 0; i < this.parDimension; i++) {
			value[i] = randomObject.nextInt((int) domain[i][0],
					(int) domain[i][1]);
		}
		return value;
	}

	public MoChromosome createRandomMoChromosome(RandomData randomObject) {
		int[] randomParameterPoint = getRandomParameterPoint(randomObject);
		DMoChromosome chrom = (DMoChromosome)(this.createMoChromosomeInstance());
		chrom.objectDimension = this.getObjectiveSpaceDimension();
		chrom.parDimension = this.getParameterSpaceDimension();
		chrom.domainInfo = this.getDomain();
		chrom.intGenes = randomParameterPoint;
		chrom.objectivesValue = new double[chrom.objectDimension];
		return chrom;
	}

	public void evaluate(double[] sp, double[] obj) {
		throw new UnsupportedOperationException(
				"this method is not intended for the use of continious problem");
	}

	public void evaluate(MoChromosome chromosome) {
		this.evaluate(((DMoChromosome) chromosome).intGenes,
				chromosome.objectivesValue);
		// update the range.
		// double[][] range2 = this.getRange();
		// for (int i = 0; i < chromosome.objectivesValue.length; i++) {
		// if (chromosome.objectivesValue[i] < range2[i][0])
		// range2[i][0] = chromosome.objectivesValue[i];
		// else if (chromosome.objectivesValue[i] < range2[i][1])
		// range2[i][1] = chromosome.objectivesValue[i];
		// else
		//				;
		//		}
	}
}
