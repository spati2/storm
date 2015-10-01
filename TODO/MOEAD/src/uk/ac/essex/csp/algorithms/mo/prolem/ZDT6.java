package uk.ac.essex.csp.algorithms.mo.prolem;

import uk.ac.essex.csp.algorithms.mo.ea.AbstractCMOProblem;

public class ZDT6 extends AbstractCMOProblem {

	private static ZDT6 instance;

	public ZDT6(int pd) {
		parDimension = pd;
		init();
	}

	public static final ZDT6 getInstance(int pd) {
		if (instance == null) {
			instance = new ZDT6(pd);
			instance.name = "ZDT6_" + pd;
		}
		return instance;
	}

	private double g(double[] point) {
		double sum = 0;
		for (int i = 1; i < parDimension; i++)
			sum += point[i];
		sum /= (parDimension - 1);

		return 1 + 9 * Math.pow(sum, 0.25);
	}

	public void evaluate(double[] sp, double[] obj) {
		obj[0] = (1.0 - Math.exp(-4.0 * sp[0])
				* Math.pow(Math.sin(6.0 * Math.PI * sp[0]), 6));

		double g = 0, h = 0;
		g = g(sp);
		h = 1.0 - ((obj[0] / g) * (obj[0] / g));

		obj[1] = g * h;
	}

	@Override
	protected void init() {
		this.domain = new double[this.parDimension][2];
		for (int i = 0; i < parDimension; i++) {
			domain[i][0] = 0;
			domain[i][1] = 1;
		}
		this.objDimension = 2;
		this.range = new double[objDimension][2];
		this.idealpoint = new double[] { 0, 0 };
	}
}
