package uk.ac.essex.csp.algorithms.mo.ea;

import org.apache.commons.math.random.RandomData;

public abstract class AbstractCMOProblem extends AbstractMOP {

	final protected double[] getRandomParameterPoint(RandomData randomObject) {
		double[][] domain = this.getDomain();
		double[] value = new double[this.parDimension];
		for (int i = 0; i < this.parDimension; i++) {
			value[i] = randomObject.nextUniform(domain[i][0], domain[i][1]);
		}
		return value;
	}

	public MoChromosome createRandomMoChromosome(RandomData randomObject) {
		double[] randomParameterPoint = getRandomParameterPoint(randomObject);
		CMoChromosome chrom = (CMoChromosome) this.createMoChromosomeInstance();
		chrom.objectDimension = this.getObjectiveSpaceDimension();
		chrom.parDimension = this.getParameterSpaceDimension();
		chrom.domainInfo = this.getDomain();
		chrom.realGenes = randomParameterPoint;
		chrom.objectivesValue = new double[chrom.objectDimension];
		chrom.objectivesEI = new double[chrom.objectDimension];
		chrom.estimatedObjectiveValue = new double[chrom.objectDimension];
		chrom.estimatedObjectiveDevitation = new double[chrom.objectDimension];
		return chrom;
	}

	@Override
	protected MoChromosome createMoChromosomeInstance() {
		return new CMoChromosome();
	}

	public void evaluate(int[] sp, double[] obj) {
		throw new UnsupportedOperationException(
				"this method is not intended for the use of continious problem");
	}

	public void evaluate(MoChromosome chromosome) {
		this.evaluate(((CMoChromosome) chromosome).realGenes,
				chromosome.objectivesValue);
		// update the range.
		double[][] range2 = this.getRange();
		for (int i = 0; i < chromosome.objectivesValue.length; i++) {
			if (chromosome.objectivesValue[i] < range2[i][0])
				range2[i][0] = chromosome.objectivesValue[i];
			else if (chromosome.objectivesValue[i] < range2[i][1])
				range2[i][1] = chromosome.objectivesValue[i];
			else
				;
		}
	}
}
