package uk.ac.essex.csp.algorithms.mo.prolem;

public class LZF1 extends LZFSuite {

	// @Override
	// protected SquareDimentionalSpace createProblemDomain() {
	// double[][] domain = new double[ParameterNumber][2];
	// for (int i = 0; i < ParameterNumber; i++) {
	// domain[i][0] = 0;
	// domain[i][1] = 1;
	// }
	// return new SquareDimentionalSpace(domain);
	// }

//	@Override
//	protected void init() {
//		this.domain = new double[this.parDimension][2];
//		for (int i = 0; i < parDimension; i++) {
//			domain[i][0] = 0;
//			domain[i][1] = 1;
//		}
//		this.objDimension = 2;
//		this.range = new double[objDimension][2];
//		this.idealpoint = new double[] { 0, 0 };
//	}

	public void evaluate(double[] sp, double[] obj) {
		super.objective(sp, obj, 21, 1, 21);
	}

	private LZF1(int pd) {
		this.parDimension = pd;
		init();
	}

	public static final LZF1 getInstance(int pd) {
		if (instance == null) {
			instance = new LZF1(pd);
			instance.name = "LZF1_" + pd;
		}
		return instance;
	}

	private static LZF1 instance;

}
