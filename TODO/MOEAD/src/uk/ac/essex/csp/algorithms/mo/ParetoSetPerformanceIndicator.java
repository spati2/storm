package uk.ac.essex.csp.algorithms.mo;

import java.util.Collection;
import java.util.Iterator;
import java.util.Stack;

public class ParetoSetPerformanceIndicator {

	/**
	 * Compute the max spread on a given asix.
	 * 
	 * @param points
	 * @param index
	 * @return
	 */
	public static double maxspread(double[][] points, int index) {
		double max = Double.NEGATIVE_INFINITY;
		double min = Double.POSITIVE_INFINITY;
		for (int j = 0; j < points.length; j++) {
			if (points[j][index] < min)
				min = points[j][index];
			if (points[j][index] > max)
				max = points[j][index];
		}
		return max - min;
	}

	/**
	 * Compute the evenly distribution on a given asix.
	 * 
	 * @param points
	 * @param index
	 * @return
	 */
	public static double eventDistribution(double[][] points, int index) {
		double total = 0;
		for (int i = 0; i < points.length; i++) {
			total += points[i][index];
		}
		double average = total / points.length;
		total = 0;
		for (int i = 0; i < points.length; i++) {
			total += Math.pow((average - points[i][index]), 2);
		}
		return (double) Math.sqrt(total / points.length);
	}

	/**
	 * compute the hypervolumn increase when a new point is added.
	 * 
	 * @param points
	 * @param newpoints
	 * @param badpoint
	 * @return
	 */
	public static double hypervolumn(double[][] points, double[] newpoints,
			double[] badpoint) {
		double result = 0;
		int len = points.length;

		Stack<double[]> stackZ = new Stack<double[]>();
		for (int i = 0; i < len; i++) {
			stackZ.push(points[i]);
		}

		stackZ.push(newpoints);

		while (stackZ.size() > len) {
			double lopOffVol = 1;
			double[] y = stackZ.pop();
			for (int i = 0; i < 4; i++) {
				double vi = getBoundValue(i, y, stackZ, badpoint);
				lopOffVol *= (vi - y[i]);
				double[] s = spawnVector(y, i, vi);
				// if stackZ is not dominating S, and s[i] != the least bound
				// value.
				if (non_dominate(stackZ, s) && s[i] != badpoint[i]) { // TODO
					stackZ.push(s);
				}
			}
			result += lopOffVol;
		}
		return result;
	}

	/**
	 * Because we got a maximum problem, here we should compute the close higher
	 * bound. This method return the nearest value to the ith component of y
	 * from a list containing the ith component of the higher bounding point and
	 * of all the vectors in solutionset.
	 * 
	 * @return
	 */
	private static double getBoundValue(int i, double[] solution,
			Collection<double[]> solutions, double[] boundpoint) {
		assert i < 4 && i >= 0;
		double minDef = Double.MAX_VALUE;
		double result = 0;

		Iterator<double[]> name = solutions.iterator();
		while (name.hasNext()) {
			double[] managerUtility2 = name.next();
			double def = managerUtility2[i] - solution[i];
			if (def > 0 && def < minDef) {
				minDef = def;
				result = managerUtility2[i];
			}
		}
		double def = boundpoint[i] - solution[i];
		if (def > 0 && def < minDef) {
			result = boundpoint[i];
		}

		return result;
	}

	/**
	 * determine if the pointset dominate the given point. when there exist one
	 * point the the collection, that dominate point, then it return true. if
	 * there is no point in the collection, then return true..
	 * 
	 * @param pointset
	 * @param point
	 * @return
	 */
	private static boolean non_dominate(Collection<double[]> pointset,
			double point[]) {
		Iterator<double[]> name = pointset.iterator();
		while (name.hasNext()) {
			double[] fs = name.next();
			if (dominate(fs, point))
				return false;
		}
		return true;
	}

	/**
	 * determine if one point dominate another.
	 * 
	 * @param a
	 * @param b
	 * @return
	 */
	public static boolean dominate(double[] a, double[] b) {
		assert a.length == b.length;
		for (int i = 0; i < 4; i++) {
			if (a[i] > b[i]) { // worse in one element.
				return false;
			}
		}
		return true;
	}

	/**
	 * this method return the vetor y with its ith component replaced by v.
	 * 
	 * @param y
	 * @param i
	 * @param v
	 * @return
	 */
	private static double[] spawnVector(double[] y, int i, double v) {
		double[] result = new double[4];
		System.arraycopy(y, 0, result, 0, 4);
		result[i] = v;
		return result;
	}
}
