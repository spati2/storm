package uk.ac.essex.csp.algorithms.mo.prolem;

import uk.ac.essex.csp.algorithms.mo.ea.AbstractCMOProblem;

public class ZDT4 extends AbstractCMOProblem {

	private static ZDT4 instance;

	public ZDT4(int pd) {
		parDimension = pd;
		init();
	}

	public static final ZDT4 getInstance(int pd) {
		if (instance == null) {
			instance = new ZDT4(pd);
			instance.name = "ZDT4_" + pd;
		}
		return instance;
	}

	private double g(double[] point) {
		double sum = 0;
		for (int i = 1; i < parDimension; i++)
			sum += (Math.pow(point[i], 2) - (10.0 * Math.cos(4 * Math.PI
					* point[i])));
		return 1.0 + (10.0 * (parDimension - 1.0)) + sum;
	}

	public void evaluate(double[] sp, double[] obj) {
		obj[0] = sp[0];

		double g = g(sp);
		double h = 1.0 - Math.sqrt(sp[0] / g);
		obj[1] = g * h;
	}

	@Override
	protected void init() {
		this.domain = new double[this.parDimension][2];
		domain[0][0] = 0;
		domain[0][1] = 1;
		for (int i = 1; i < parDimension; i++) {
			domain[i][0] = -5;
			domain[i][1] = 5;
		}
		this.objDimension = 2;
		this.range = new double[objDimension][2];
		this.idealpoint = new double[] { 0, 0 };
	}
}
