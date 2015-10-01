package uk.ac.essex.csp.algorithms.mo.prolem;

import static java.lang.Math.PI;
import static java.lang.Math.cos;
import static java.lang.Math.sin;
import uk.ac.essex.csp.algorithms.mo.ea.AbstractCMOProblem;

/**
 * Adpoted from the parGEO's paper on TEC.
 * 
 * @author wudong
 * 
 */
public class KNO1 extends AbstractCMOProblem {
	//
	// @Override
	// protected SquareDimentionalSpace createProblemDomain() {
	// double[][] domain = { { 0, 3 }, { 0, 3 } };
	// return new SquareDimentionalSpace(domain);
	// }

	public void evaluate(double[] x, double[] y) {
		// double r = r(sp);
		// double t = t(sp);
		// obj[0] = 20-r*cos(t);
		// obj[1] = 20-r*sin(t);

		double f;
		double g;
		double c;

		c = x[0] + x[1];

		f = 20 - (11 + 3 * sin((5 * c) * (0.5 * c)) + 3 * sin(4 * c) + 5 * sin(2 * c + 2));
		// f = 20*(1-(myabs(c-3.0)/3.0));

		g = (PI / 2.0) * (x[0] - x[1] + 3.0) / 6.0;

		y[0] = 20 - (f * cos(g));
		y[1] = 20 - (f * sin(g));
	}

	@Override
	protected void init() {
		this.parDimension = 2;
		this.domain = new double[this.parDimension][2];
		for (int i = 0; i < parDimension; i++) {
			domain[i][0] = 0;
			domain[i][1] = 3;
		}
		this.objDimension = 2;
		this.range = new double[objDimension][2];
		this.idealpoint = new double[] { 0, 0 };
	}

	public KNO1() {
		init();
	}

	public static final KNO1 getInstance() {
		if (instance == null) {
			instance = new KNO1();
			instance.name = "KNO1";
		}
		return instance;
	}

	// @Override
	// public double[][] getRange() {
	// if (range == null) {
	// range = new double[][] { { 0, 20 }, { 0, 20 } };
	// };
	// return range;
	// }

	// private double[][] range;
	//
	// private double[] idealpoint;

	private static KNO1 instance;
}
