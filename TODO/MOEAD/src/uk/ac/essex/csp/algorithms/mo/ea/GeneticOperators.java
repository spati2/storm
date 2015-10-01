package uk.ac.essex.csp.algorithms.mo.ea;

import org.apache.commons.math.random.RandomGenerator;

public class GeneticOperators {

	public static final int id_cx = 20;

	public static final int id_mu = 20;

	public static final double EPS = 1.2e-7;

	// public static void realCrossover2(MoChromosome parent1,
	// MoChromosome parent2, MoChromosome child1, MoChromosome child2,
	// SquareDimentionalSpace domain) {
	// double rand;
	// double y1, y2, yl, yu;
	// double c1, c2;
	// double alpha, beta, betaq;
	// double eta_c = id_cx;
	//
	// int numVariables = parent1.getDimension();
	// if (Math.random() <= 1.0) {
	// for (int i = 0; i < numVariables; i++) {
	// if (Math.random() <= 0.5) {
	// if (Math.abs(parent1.realGenes[i] - parent2.realGenes[i]) > EPS) {
	// if (parent1.realGenes[i] < parent2.realGenes[i]) {
	// y1 = parent1.realGenes[i];
	// y2 = parent2.realGenes[i];
	// } else {
	// y1 = parent2.realGenes[i];
	// y2 = parent1.realGenes[i];
	// }
	// DoubleRange dimensionRange = domain
	// .getDimensionRange(i);
	// yl = dimensionRange.getMinimumDouble();
	// yu = dimensionRange.getMaximumDouble();
	// rand = Math.random();
	// beta = 1.0 + (2.0 * (y1 - yl) / (y2 - y1));
	// alpha = 2.0 - Math.pow(beta, -(eta_c + 1.0));
	// if (rand <= (1.0 / alpha)) {
	// betaq = Math.pow((rand * alpha),
	// (1.0 / (eta_c + 1.0)));
	// } else {
	// betaq = Math.pow((1.0 / (2.0 - rand * alpha)),
	// (1.0 / (eta_c + 1.0)));
	// }
	// c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1));
	// beta = 1.0 + (2.0 * (yu - y2) / (y2 - y1));
	// alpha = 2.0 - Math.pow(beta, -(eta_c + 1.0));
	// if (rand <= (1.0 / alpha)) {
	// betaq = Math.pow((rand * alpha),
	// (1.0 / (eta_c + 1.0)));
	// } else {
	// betaq = Math.pow((1.0 / (2.0 - rand * alpha)),
	// (1.0 / (eta_c + 1.0)));
	// }
	// c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1));
	// if (c1 < yl)
	// c1 = yl;
	// if (c2 < yl)
	// c2 = yl;
	// if (c1 > yu)
	// c1 = yu;
	// if (c2 > yu)
	// c2 = yu;
	// if (Math.random() <= 0.5) {
	// child1.realGenes[i] = c2;
	// child2.realGenes[i] = c1;
	// } else {
	// child1.realGenes[i] = c1;
	// child2.realGenes[i] = c2;
	// }
	// } else {
	// child1.realGenes[i] = parent1.realGenes[i];
	// child2.realGenes[i] = parent2.realGenes[i];
	// }
	// } else {
	// child1.realGenes[i] = parent1.realGenes[i];
	// child2.realGenes[i] = parent2.realGenes[i];
	// }
	// }
	// } else {
	// for (int i = 0; i < numVariables; i++) {
	// child1.realGenes[i] = parent1.realGenes[i];
	// child2.realGenes[i] = parent2.realGenes[i];
	// }
	// }
	// return;
	// }

	/* Routine for real polynomial mutation of an T */
	public static void realmutation(double[] ind, double rate,
			RandomGenerator rg) {
		int dimensionNumber = ind.length;

		double rnd, delta1, delta2, mut_pow, deltaq;
		double y, yl, yu, val, xy;
		double eta_m = 20;

		for (int j = 0; j < dimensionNumber; j++) {
			if (rg.nextDouble() <= rate) {
				y = ind[j];
				yl = 0;
				yu = 1;

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
				ind[j] = y;
			}
		}
		return;
	}
}
