package uk.ac.essex.csp.algorithms.statictics;

import org.apache.commons.math.DimensionMismatchException;
import org.apache.commons.math.linear.RealMatrixImpl;
import org.apache.commons.math.stat.descriptive.moment.VectorialCovariance;

public class Correlation {

	/**
	 * Return the Correlation coefficients of the samples.
	 * 
	 * @return
	 */
	public static double[][] corrcoef(double[][] samples, int dimension) {
		VectorialCovariance vc = new VectorialCovariance(dimension, true);
		for (int i = 0; i < samples.length; i++) {
			try {
				vc.increment(samples[i]);
			} catch (DimensionMismatchException e) {
				e.printStackTrace();
			}
		}
		RealMatrixImpl covmatrix = (RealMatrixImpl) vc.getResult();
		double[][] dataRef = covmatrix.getDataRef();
		for (int i = 0; i < dimension; i++) {
			for (int j = i + 1; j < dimension; j++) {
				dataRef[i][j] = dataRef[i][j]
						/ Math.sqrt(dataRef[i][i] * dataRef[j][j]);
				dataRef[j][i] = dataRef[i][j];
			}
			dataRef[i][i] = 1;
		}
		return dataRef;
	}
}
