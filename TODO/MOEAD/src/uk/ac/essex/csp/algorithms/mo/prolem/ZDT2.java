package uk.ac.essex.csp.algorithms.mo.prolem;

import uk.ac.essex.csp.algorithms.mo.ea.AbstractCMOProblem;

public class ZDT2 extends AbstractCMOProblem {

	private static ZDT2 instance;

	private ZDT2(int pd) {
		this.parDimension = pd;
		init();
	}

	public static final ZDT2 getInstance(int pd) {
		if (instance == null) {
			instance = new ZDT2(pd);
			instance.name = "ZDT2_"+pd;
		}
		return instance;
	}

	public void evaluate(double[] sp, double[] obj) {
		obj[0]=sp[0];
		double g = g(sp);
		double part2 = (1 - Math.pow(sp[0] / g, 2));
		obj[1]= g * part2;
	}
	
	private double g(double[] point) {
		double sum = 0;
		for (int i = 1; i < parDimension; i++)
			sum += point[i];
		return 1 + 9 * sum / (parDimension - 1);
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
