package uk.ac.essex.csp.algorithms.mo.prolem;

import static java.lang.Math.PI;
import static java.lang.Math.abs;
import static java.lang.Math.cos;
import static java.lang.Math.pow;
import static java.lang.Math.sin;
import static java.lang.Math.sqrt;
import uk.ac.essex.csp.algorithms.mo.ea.AbstractCMOProblem;

public class OKA1 extends AbstractCMOProblem {

	public void evaluate(double[] sp, double[] obj) {
		// double x1=x1(sp);
		// double x2=x2(sp);
		// obj[0]=x1;
		// obj[1]=sqrt(2*PI)-sqrt(abs(x1))+2*pow(abs(x2-3*cos(x1)-3),1/3d);

		double x1p = cos(PI / 12.0) * sp[0] - sin(PI / 12.0) * sp[1];
		double x2p = sin(PI / 12.0) * sp[0] + cos(PI / 12.0) * sp[1];

		obj[0] = x1p;
		obj[1] = sqrt(2 * PI) - sqrt(abs(x1p)) + 2
				* pow(abs(x2p - 3 * cos(x1p) - 3), 0.33333333);
	}

	public OKA1() {
		init();
	}

	@Override
	protected void init() {
		this.parDimension = 2;
		this.domain = new double[][] {
				{ 6 * sin(PI / 12.0),
						6 * sin(PI / 12.0) + 2 * PI * cos(PI / 12.0) },
				{ -2 * PI * sin(PI / 12.0), 6 * cos(PI / 12.0) } };
		this.objDimension = 2;
		this.range = new double[objDimension][2];
		this.idealpoint = new double[] { 0, 0 };
	}

	public static final OKA1 getInstance() {
		if (instance == null) {
			instance = new OKA1();
			instance.name = "OKA1";
		}
		return instance;
	}

	private static OKA1 instance;
}
