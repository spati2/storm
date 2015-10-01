package uk.ac.essex.csp.algorithms.mo.prolem;

import uk.ac.essex.csp.algorithms.mo.ea.AbstractCMOProblem;

public class ZDT3 extends AbstractCMOProblem {

	private static ZDT3 instance;

	public ZDT3(int pd) {
		parDimension = pd;
		init();
	}

	public static final ZDT3 getInstance(int pd) {
		if (instance == null) {
			instance = new ZDT3(pd);
			instance.name = "ZDT3_" + pd;
		}
		return instance;
	}

	private double g(double[] point) {
		double sum = 0;
		for (int i = 1; i < parDimension; i++)
			sum += point[i];
		return 1 + 9 * sum / (parDimension - 1);
	}

	public void evaluate(double[] sp, double[] obj) {
		obj[0] = sp[0];
		double g = g(sp);
		double part = sp[0] / g;
		double part2 = (1 - Math.sqrt(part) - part
				* Math.sin(10 * Math.PI * sp[0]));
		obj[1] = g * part2;
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
		this.idealpoint = new double[] { 0, -1 };
	}

	// @Override
	// protected SquareDimentionalSpace createProblemDomain() {
	// double[][] domain = new double[ParameterNumber][2];
	// for (int i = 0; i < ParameterNumber; i++) {
	// domain[i][0] = 0;
	// domain[i][1] = 1;
	// }
	// return new SquareDimentionalSpace(domain);
	// }
}
