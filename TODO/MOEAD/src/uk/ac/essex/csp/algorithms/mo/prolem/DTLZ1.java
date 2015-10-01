package uk.ac.essex.csp.algorithms.mo.prolem;

import static java.lang.Math.PI;
import static java.lang.Math.cos;
import uk.ac.essex.csp.algorithms.mo.ea.AbstractCMOProblem;

/**
 * Adpoted from the parGEO's paper on TEC.
 * 
 * @author wudong
 * 
 */
public class DTLZ1 extends AbstractCMOProblem {

	public DTLZ1() {
		init();
	}

	@Override
	protected void init() {
		parDimension = 6;
		this.domain = new double[this.parDimension][2];
		for (int i = 0; i < parDimension; i++) {
			domain[i][0] = 0;
			domain[i][1] = 1;
		}
		this.objDimension = 2;
		this.range = new double[objDimension][2];
		// this.idealpoint = new double[] { 0, 0 };
	}

	// @Override
	// protected SquareDimentionalSpace createProblemDomain() {
	// double[][] domain = { { 0, 1 }, { 0, 1 } ,{ 0, 1 }, { 0, 1 },{ 0, 1 }, {
	// 0, 1 }};
	// return new SquareDimentionalSpace(domain);
	// }

	public void evaluate(double[] x, double[] y) {
		double g = 0.0;
		for (int i = 1; i < parDimension; i++)
			g += (x[i] - 0.5) * (x[i] - 0.5) - cos(2 * PI * (x[i] - 0.5));
		// Note this is 20*PI in Deb's dtlz1 func
		g += parDimension - 1;
		g *= 100;

		y[0] = 0.5 * x[0] * (1 + g);
		y[1] = 0.5 * (1 - x[0]) * (1 + g);
	}

	public static final DTLZ1 getInstance() {
		if (instance == null) {
			instance = new DTLZ1();
			instance.name = "DTLZ1";
		}
		return instance;
	}

	private static DTLZ1 instance;

}
