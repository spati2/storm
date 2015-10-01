package uk.ac.essex.csp.algorithms.mo.prolem;

public class LZF2 extends LZFSuite {

	// @Override
	// protected SquareDimentionalSpace createProblemDomain() {
	// double[][] domain = new double[ParameterNumber][2];
	// domain[0][0]=0;
	// domain[0][1]=1;
	//		
	// for (int i = 1; i < ParameterNumber; i++) {
	// //domain[i][0] = -1;
	// domain[i][0] = 0;
	// domain[i][1] = 1;
	// }
	// return new SquareDimentionalSpace(domain);
	// }

	public void evaluate(double[] sp, double[] obj) {
		super.objective(sp, obj, 22, 1, 21);
	}

	private LZF2(int pd) {
		this.parDimension = pd;
		init();
	}

	public static final LZF2 getInstance(int pd) {
		if (instance == null) {
			instance = new LZF2(pd);
			instance.name = "LZF2_" + pd;
		}
		return instance;
	}

	private static LZF2 instance;

}
