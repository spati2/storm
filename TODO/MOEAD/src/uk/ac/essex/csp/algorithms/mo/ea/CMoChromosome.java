package uk.ac.essex.csp.algorithms.mo.ea;

import java.util.Arrays;

import org.apache.commons.lang.builder.ToStringBuilder;
import org.apache.commons.lang.builder.ToStringStyle;
import org.apache.commons.math.random.RandomData;
import org.apache.commons.math.random.RandomGenerator;

public class CMoChromosome extends MoChromosome {
	public static final int id_cx = 20;
	public static final int id_mu = 20;

	public double[] realGenes;

	protected void randomizeParameter(RandomData randomGenerator) {
		for (int i = 0; i < realGenes.length; i++) {
			realGenes[i] = randomGenerator.nextUniform(domainInfo[i][0],
					domainInfo[i][1]);
		}
	}

	@Override
	public boolean equals(Object obj) {
		if (obj instanceof CMoChromosome == false)
			return false;
		if (this == obj)
			return true;
		CMoChromosome another = (CMoChromosome) obj;
		boolean equals = true;
		for (int i = 0; i < realGenes.length; i++) {
			if (Math.abs(another.realGenes[i] - this.realGenes[i]) > 1000 * Double.MIN_VALUE)
				equals = false;
		}
		return equals;
	}

	@Override
	public String toString() {
		return new ToStringBuilder(this, ToStringStyle.SIMPLE_STYLE).append(
				this.realGenes).append(this.objectivesValue).append(
				this.fitnessValue).toString();
	}

	public void copyTo(MoChromosome copyto) {
		super.copyTo(copyto);
		System.arraycopy(this.objectivesEI, 0, copyto.objectivesEI, 0,
				this.objectivesEI.length);
		System.arraycopy(this.estimatedObjectiveDevitation, 0,
				copyto.estimatedObjectiveDevitation, 0,
				this.estimatedObjectiveDevitation.length);
		System.arraycopy(this.estimatedObjectiveValue, 0,
				copyto.estimatedObjectiveValue, 0,
				this.estimatedObjectiveValue.length);
		System.arraycopy(this.realGenes, 0, ((CMoChromosome) copyto).realGenes,
				0, this.realGenes.length);
	}

	public void copyFrom(double[] genevalue) {
		if (realGenes == null)
			realGenes = new double[parDimension];
		System.arraycopy(genevalue, 0, realGenes, 0, parDimension);
	}

	public String vectorString() {
		String string1 = Arrays.toString(realGenes);
		String string2 = Arrays.toString(this.objectivesValue);
		String string3 = Arrays.toString(this.estimatedObjectiveValue);
		String string4 = Arrays.toString(this.estimatedObjectiveDevitation);
		return string1.substring(1, string1.length() - 1) + "; "
				+ string2.substring(1, string2.length() - 1) + "; "
				+ string3.substring(1, string3.length() - 1) + "; "
				+ string4.substring(1, string4.length() - 1);
	}

	public void mutate(RandomGenerator rg, double rate) {
		double rnd, delta1, delta2, mut_pow, deltaq;
		double y, yl, yu, val, xy;
		double eta_m = 20;

		for (int j = 0; j < parDimension; j++) {
			if (rg.nextDouble() <= rate) {
				y = realGenes[j];
				yl = domainInfo[j][0];
				yu = domainInfo[j][1];

				delta1 = (y - yl) / (yu - yl);
				delta2 = (yu - y) / (yu - yl);

				rnd = rg.nextDouble();
				mut_pow = 1.0 / (eta_m + 1.0);
				if (rnd <= 0.5) {
					xy = 1.0 - delta1;
					val = 2.0 * rnd + (1.0 - 2.0 * rnd)
							* (Math.pow(xy, (eta_m + 1.0)));
					deltaq = Math.pow(val, mut_pow) - 1.0;
				} else {
					xy = 1.0 - delta2;
					val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5)
							* (Math.pow(xy, (eta_m + 1.0)));
					deltaq = 1.0 - (Math.pow(val, mut_pow));
				}
				y = y + deltaq * (yu - yl);
				if (y < yl)
					y = yl;
				if (y > yu)
					y = yu;
				realGenes[j] = y;
			}
		}
		return;
	};

	public void diff_xover(MoChromosome ind0, MoChromosome ind1,
			MoChromosome ind2, RandomData randomData) {
		int nvar = ind0.domainInfo.length;
		// int idx_rnd = this.randomGenerator.nextInt(nvar);
		double rate = 0.5;
		for (int n = 0; n < nvar; n++) {
			/* Selected Two Parents */
			double lowBound = ind0.domainInfo[n][0];
			double upBound = ind0.domainInfo[n][1];
			realGenes[n] = ((CMoChromosome) ind0).realGenes[n]
					+ rate
					* (((CMoChromosome) ind2).realGenes[n] - ((CMoChromosome) ind1).realGenes[n]);

			if (realGenes[n] < lowBound) {
				realGenes[n] = randomData.nextUniform(lowBound, upBound);
			}
			if (realGenes[n] > upBound) {
				realGenes[n] = randomData.nextUniform(lowBound, upBound);
			}
		}
	}

	@Override
	public void crossover(MoChromosome p1, MoChromosome p2, RandomGenerator rg) {
		double rand;
		double y1, y2, yl, yu;
		double c1, c2;
		double alpha, beta, betaq;
		double eta_c = id_cx;

		CMoChromosome parent1 = (CMoChromosome) p1;
		CMoChromosome parent2 = (CMoChromosome) p2;
		int numVariables = p1.domainInfo.length;
		if (rg.nextDouble() <= 1.0) {
			for (int i = 0; i < numVariables; i++) {
				if (rg.nextDouble() <= 0.5) {
					if (Math.abs(parent1.realGenes[i] - parent2.realGenes[i]) > EPS) {
						if (parent1.realGenes[i] < parent2.realGenes[i]) {
							y1 = parent1.realGenes[i];
							y2 = parent2.realGenes[i];
						} else {
							y1 = parent2.realGenes[i];
							y2 = parent1.realGenes[i];
						}
						yl = p1.domainInfo[i][0];
						yu = p1.domainInfo[i][1];
						rand = rg.nextDouble();
						beta = 1.0 + (2.0 * (y1 - yl) / (y2 - y1));
						alpha = 2.0 - Math.pow(beta, -(eta_c + 1.0));
						if (rand <= (1.0 / alpha)) {
							betaq = Math.pow((rand * alpha),
									(1.0 / (eta_c + 1.0)));
						} else {
							betaq = Math.pow((1.0 / (2.0 - rand * alpha)),
									(1.0 / (eta_c + 1.0)));
						}
						c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1));
						beta = 1.0 + (2.0 * (yu - y2) / (y2 - y1));
						alpha = 2.0 - Math.pow(beta, -(eta_c + 1.0));
						if (rand <= (1.0 / alpha)) {
							betaq = Math.pow((rand * alpha),
									(1.0 / (eta_c + 1.0)));
						} else {
							betaq = Math.pow((1.0 / (2.0 - rand * alpha)),
									(1.0 / (eta_c + 1.0)));
						}
						c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1));
						if (c1 < yl)
							c1 = yl;
						if (c2 < yl)
							c2 = yl;
						if (c1 > yu)
							c1 = yu;
						if (c2 > yu)
							c2 = yu;
						if (rg.nextDouble() <= 0.5) {
							realGenes[i] = c2;
						} else {
							realGenes[i] = c1;
						}
					} else {
						realGenes[i] = parent1.realGenes[i];
					}
				} else {
					realGenes[i] = parent1.realGenes[i];
				}
			}
		} else {
			for (int i = 0; i < numVariables; i++) {
				realGenes[i] = parent1.realGenes[i];
			}
		}
	}

	@Override
	public String getParameterString() {
		String string = Arrays.toString(this.realGenes);
		return string.substring(1, string.length() - 1);
	}

	@Override
	public double parameterDistance(MoChromosome another) {
		return euclideanDistance(this.realGenes,
				((CMoChromosome) another).realGenes);
	}

	public static double euclideanDistance(double[] a, double[] b) {
		double sum = 0;
		for (int i = 0; i < a.length; i++) {
			sum += (a[i] - b[i]) * (a[i] - b[i]);
		}
		return Math.sqrt(sum);
	}
}
