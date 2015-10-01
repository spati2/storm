package uk.ac.essex.csp.algorithms.mo.prolem;

public class LZF5 extends LZFSuite {

	public void evaluate(double[] sp, double[] obj) {
		super.objective(sp, obj, 26, 1, 21);
	}

	private LZF5(int pd) {
		this.parDimension = pd;
		init();
	}

	public static final LZF5 getInstance(int pd) {
		if (instance == null) {
			instance = new LZF5(pd);
			instance.name = "LZF5_" + pd;
		}
		return instance;
	}

	private static LZF5 instance;

}
