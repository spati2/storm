package uk.ac.essex.csp.algorithms.mo.prolem;

import static java.lang.Math.PI;
import static java.lang.Math.cos;
import uk.ac.essex.csp.algorithms.mo.ea.AbstractCMOProblem;

/**
 * The original oka1 problem from Okabe thesis, page 138.
 * 
 * @author wudong
 * 
 */
public class OKA1O extends AbstractCMOProblem {

	public OKA1O() {
		init();
	}

	@Override
	protected void init() {
		this.parDimension = 2;
		this.domain = new double[][] { { -PI, PI }, { -5, 5 } };
		this.objDimension = 2;
		this.range = new double[objDimension][2];
		this.idealpoint = new double[] { 0, 0 };
	}

	public void evaluate(double[] sp, double[] obj) {
		obj[0] = sp[0];
		obj[1] = PI - sp[0] + Math.abs(sp[1] - 5 * cos(sp[0]));
	}

	public int getObjectiveSpaceDimension() {
		return 2;
	}

	public int getParameterSpaceDimension() {
		return 2;
	}

	public static final OKA1O getInstance() {
		if (instance == null) {
			instance = new OKA1O();
			instance.name = "OKA1O";
		}
		return instance;
	}

	private static OKA1O instance;
}
