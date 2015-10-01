package uk.ac.essex.csp.algorithms.mo.ea;

import java.util.Arrays;

import org.apache.commons.lang.builder.ToStringBuilder;
import org.apache.commons.lang.builder.ToStringStyle;
import org.apache.commons.math.random.RandomData;
import org.apache.commons.math.random.RandomGenerator;

public class DMoChromosome extends MoChromosome {
	public int[] intGenes;

	@Override
	public void randomizeParameter(RandomData randomGenerator) {
		for (int i = 0; i < intGenes.length; i++) {
			intGenes[i] = randomGenerator.nextInt((int) domainInfo[i][0],
					(int) domainInfo[i][1]);
		}
	}
	
	@Override
	public String getParameterString() {
		String string = Arrays.toString(this.intGenes);
		return string.substring(1, string.length() - 1);
	}

	@Override
	public boolean equals(Object obj) {
		if (obj instanceof CMoChromosome == false)
			return false;
		if (this == obj)
			return true;
		CMoChromosome another = (CMoChromosome) obj;
		boolean equals = true;
		for (int i = 0; i < intGenes.length; i++) {
			if (Math.abs(another.realGenes[i] - this.intGenes[i]) > 1000 * Double.MIN_VALUE)
				equals = false;
		}
		return equals;
	}

	@Override
	public String toString() {
		return new ToStringBuilder(this, ToStringStyle.SIMPLE_STYLE).append(
				this.intGenes).append(this.objectivesValue).append(
				this.fitnessValue).toString();
	}

	public void copyTo(MoChromosome copyto) {
		super.copyTo(copyto);
		System.arraycopy(this.intGenes, 0, ((DMoChromosome) copyto).intGenes,
				0, this.intGenes.length);
	}

	public void copyFrom(int[] genevalue) {
		if (intGenes == null)
			intGenes = new int[domainInfo.length];
		System.arraycopy(genevalue, 0, intGenes, 0, intGenes.length);
	}

	@Override
	public void mutate(RandomGenerator rg, double rate) {
		for (int i = 0; i < intGenes.length; i++) {
			if (rg.nextDouble() < rate) {// mutate.
				int span = (int) domainInfo[i][1] - (int) domainInfo[i][0];
				intGenes[i] = rg.nextInt(span) + (int) domainInfo[i][0];
			}
		}
	};

	@Override
	public void crossover(MoChromosome ind0, MoChromosome ind1,
			RandomGenerator randomData) {
		int nvar = ind0.parDimension;
		// int idx_rnd = this.randomGenerator.nextInt(nvar);
		int i = randomData.nextInt(nvar);
		int j = randomData.nextInt(nvar);
		int min = Math.min(i, j);
		int max = Math.max(i, j);
		for (int n = 0; n < nvar; n++) {
			if (n < min)
				intGenes[n] = ((DMoChromosome) ind0).intGenes[n];
			else if (n > max)
				intGenes[n] = ((DMoChromosome) ind1).intGenes[n];
			else
				intGenes[n] = ((DMoChromosome) ind0).intGenes[n];
		}
	}

	@Override
	public void diff_xover(MoChromosome ind0, MoChromosome ind1,
			MoChromosome ind2, RandomData randomData) {
		int nvar = ind0.parDimension;
		// int idx_rnd = this.randomGenerator.nextInt(nvar);
		int i = randomData.nextInt(0, nvar - 1);
		int j = randomData.nextInt(0, nvar - 1);
		int min = Math.min(i, j);
		int max = Math.max(i, j);
		for (int n = 0; n < nvar; n++) {
			if (n < min)
				intGenes[n] = ((DMoChromosome) ind0).intGenes[n];
			else if (n > max)
				intGenes[n] = ((DMoChromosome) ind1).intGenes[n];
			else
				intGenes[n] = ((DMoChromosome) ind2).intGenes[n];
		}
	}
	
	public String vectorString() {return this.toString();} 
	
	@Override
	public double parameterDistance(MoChromosome another) {
		//TODO
		return 0;
	}
}
