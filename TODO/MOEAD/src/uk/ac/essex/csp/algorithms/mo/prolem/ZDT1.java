package uk.ac.essex.csp.algorithms.mo.prolem;

import uk.ac.essex.csp.algorithms.mo.ea.AbstractCMOProblem;

public class ZDT1 extends AbstractCMOProblem {

	private static ZDT1 instance;

	private ZDT1(int pd) {
		this.parDimension = pd;
		init();
	}

	public static final ZDT1 getInstance(int pd) {
		if (instance == null) {
			instance = new ZDT1(pd);
			instance.name = "ZDT1_" + pd;
		}
		return instance;
	}

	public void evaluate(double[] sp, double[] obj) {
		obj[0] = sp[0];
		double g = g(sp);
		double part2 = (1 - Math.sqrt(obj[0] / g));
		obj[1] = g * part2;
		// obj[1] = 50*obj[1];//test for normalization.
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

	private double g(double[] point) {
		double sum = 0;
		for (int i = 1; i < parDimension; i++)
			sum += point[i];
		return 1 + 9 * sum / (parDimension - 1);
	}
}
