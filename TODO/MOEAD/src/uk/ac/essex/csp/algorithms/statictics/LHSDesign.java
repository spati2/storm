package uk.ac.essex.csp.algorithms.statictics;

import org.apache.commons.math.random.RandomGenerator;
import org.apache.commons.math.stat.StatUtils;

import uk.ac.essex.csp.algorithms.Sorting;

/**
 * Algorithms copy and translated from matlab stats toolbox lhsdesign.m.
 * 
 * @author wliui
 * 
 */
public class LHSDesign {
	public static final int MAXITR = 50000;

	/**
	 * Get a random LHS Design
	 * 
	 * @param n
	 * @param p
	 * @param dosmooth
	 * @return
	 */
	public static double[][] getlhssample(int number, int dimension,
			boolean dosmooth, RandomGenerator rg) {
		double[][] result = new double[number][dimension];
		lhssample(number, dimension, dosmooth, result, rg);
		return result;
	}

	public static void lhssample(int number, int dimension, boolean dosmooth,
			double[][] sample, RandomGenerator rg) {
		for (int i = 0; i < number; i++)
			for (int j = 0; j < dimension; j++)
				sample[i][j] = rg.nextDouble();
		double[] column = new double[number];
		for (int i = 0; i < dimension; i++) {
			copycolumn(sample, column, i);
			int[] rank = rank(column);
			for (int j = 0; j < number; j++)
				sample[j][i] = rank[j];
		}
		for (int i = 0; i < number; i++)
			for (int j = 0; j < dimension; j++) {
				if (dosmooth)
					sample[i][j] -= rg.nextDouble();
				else
					sample[i][j] -= 0.5;
				sample[i][j] /= number;
			}
	}

	public static double score(double[][] design, int dimension) {
		double[][] coef = Correlation.corrcoef(design, dimension);
		double sum = 0;
		for (int i = 0; i < dimension; i++) {
			for (int j = i + 1; j < dimension; j++)
				sum += coef[i][j] * coef[i][j];
		}
		return -sum; // the larger , the better.
	}

	public static double[][] betterDesign(int number, int dimension, RandomGenerator rg) {
		double[][] design = new double[number][dimension];
		goodDesign(number, dimension, design, rg);
		double score = score(design, dimension);

		double[][] betterdesign = new double[number][dimension];
		for (int i = 0; i < 200; i++) {
			goodDesign(number, dimension, betterdesign, rg);
			double s = score(betterdesign, dimension);
			if (s > score) {// better desgin
				score = s;
				for (int j = 0; j < number; j++) {
					System.arraycopy(betterdesign[j], 0, design[j], 0, dimension);
				}
			}
		}
		return design;
	}

	public static void goodDesign(int number, int dimension, double[][] sample, RandomGenerator rg) {
		lhssample(number, dimension, false, sample, rg);

		double bestscore = score(sample, dimension);

		double[] columnj = new double[number];
		double[] columnk = new double[number];
		double[] z = new double[number];

		for (int itr = 2; itr < MAXITR; itr++) {
			// Forward ranked Gram-Schmidt step:
			for (int j = 1; j < dimension; j++) {
				copycolumn(sample, columnj, j);
				for (int k = 0; k <= j - 1; k++) {
					copycolumn(sample, columnk, k);
					takeout(columnj, columnk, z);
					int[] r = rank(z);
					for (int row = 0; row < number; row++)
						sample[row][k] = (r[row] - 0.5) / number;
				}
			}
			// Backward ranked Gram-Schmidt step:
			for (int j = dimension - 2; j >= 0; j--) {
				copycolumn(sample, columnj, j);
				for (int k = dimension - 1; k >= j + 1; k--) {
					copycolumn(sample, columnk, k);
					takeout(columnj, columnk, z);
					int[] r = rank(z);
					for (int row = 0; row < number; row++)
						sample[row][k] = (r[row] - 0.5) / number;
				}
			}
			// Check for convergence
			double newscore = score(sample, dimension);
			if (newscore <= bestscore)
				break;
			else
				bestscore = newscore;
		}
	}

	public static double[][] goodDesign(int number, int dimension) {
		double[][] sample = new double[number][dimension];
		return sample;
	}

	public static double[] takeout(double[] x, double[] y) {
		// Remove from y its projection onto x, ignoring constant terms
		double[] z = new double[x.length];
		takeout(x, y, z);
		return z;
	}

	public static void takeout(double[] x, double[] y, double[] z) {
		// Remove from y its projection onto x, ignoring constant terms
		double[] xc = new double[x.length];
		double[] yc = new double[y.length];
		double[] xx = new double[x.length];
		double[] yy = new double[x.length];

		double xmean = StatUtils.mean(x);
		double ymean = StatUtils.mean(y);

		for (int i = 0; i < xc.length; i++) {
			xc[i] = x[i] - xmean;
			yc[i] = y[i] - ymean;
		}

		double xcmean = StatUtils.mean(xc);
		double ycmean = StatUtils.mean(yc);
		for (int i = 0; i < xc.length; i++) {
			xx[i] = xc[i] - xcmean;
			yy[i] = yc[i] - ycmean;
		}

		double b = leftdevide(xx, yy);
		for (int i = 0; i < z.length; i++)
			z[i] = y[i] - b * xc[i];
	}

	public static int[] rank(double[] x) {
		int[] rowinx = Sorting.sorting(x);
		int[] r = new int[x.length];
		for (int i = 0; i < x.length; i++)
			r[rowinx[i]] = i + 1;
		return r;
	}

	public static double leftdevide(double[] a, double b[]) {
		double above = 0, below = 0;
		for (int i = 0; i < a.length; i++) {
			above += a[i] * b[i];
			below += a[i] * a[i];
		}
		return above / below;
	}

	public static void copycolumn(double[][] matrix, double[] column, int index) {
		for (int i = 0; i < matrix.length; i++) {
			column[i] = matrix[i][index];
		}
	}

}
