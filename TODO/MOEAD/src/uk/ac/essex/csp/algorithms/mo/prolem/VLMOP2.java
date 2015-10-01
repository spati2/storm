package uk.ac.essex.csp.algorithms.mo.prolem;

import static java.lang.Math.exp;
import static java.lang.Math.pow;
import static java.lang.Math.sqrt;
import uk.ac.essex.csp.algorithms.mo.ea.AbstractCMOProblem;

/**
 * Adpoted from the parGEO's paper on TEC.
 * 
 * @author wudong
 * 
 */
public class VLMOP2 extends AbstractCMOProblem {

	public VLMOP2() {
		init();
	}

	@Override
	protected void init() {
		this.parDimension = 2;
		this.domain = new double[this.parDimension][2];
		for (int i = 0; i < parDimension; i++) {
			domain[i][0] = -2;
			domain[i][1] = 2;
		}
		this.objDimension = 2;
		this.range = new double[objDimension][2];
		this.idealpoint = new double[] { 0, 0 };

	}

	public void evaluate(double[] x, double[] y) {
		double sum1 = 0;
		double sum2 = 0;

		for (int i = 0; i < 2; i++) {
			sum1 += pow(x[i] - (1 / sqrt(2.0)), 2);
			sum2 += pow(x[i] + (1 / sqrt(2.0)), 2);
		}

		y[0] = 1 - exp(-sum1);
		y[1] = 1 - exp(-sum2);
	}

	public static final VLMOP2 getInstance() {
		if (instance == null) {
			instance = new VLMOP2();
			instance.name = "VLMOP2";
		}
		return instance;
	}

	private static VLMOP2 instance;
}
