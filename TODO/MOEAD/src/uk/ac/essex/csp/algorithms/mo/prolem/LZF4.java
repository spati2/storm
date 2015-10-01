package uk.ac.essex.csp.algorithms.mo.prolem;


public class LZF4 extends LZFSuite {

	public void evaluate(double[] sp, double[] obj) {
		super.objective(sp, obj, 24, 1, 21);
	}

	private LZF4(int pd) {
		this.parDimension = pd;
		init();
	}

	public static final LZF4 getInstance(int pd) {
		if (instance == null) {
			instance = new LZF4(pd);
			instance.name = "LZF4_" + pd;
		}
		return instance;
	}

	private static LZF4 instance;

}
