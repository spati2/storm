package uk.ac.essex.csp.algorithms.random;

import org.apache.commons.math.random.AbstractRandomGenerator;

public class CustomRandomGenerator extends AbstractRandomGenerator {

	private static final long IM1 = 2147483563;
	private static final long IM2 = 2147483399;
	private static final double AM = (1.0 / IM1);
	private static final long IMM1 = (IM1 - 1);
	private static final long IA1 = 40014;
	private static final long IA2 = 40692;
	private static final long IQ1 = 53668;
	private static final long IQ2 = 52774;
	private static final long IR1 = 12211;
	private static final long IR2 = 3791;
	private static final int NTAB = 32;
	private static final long NDIV = (1 + IMM1 / NTAB);
	private static final double EPS = 1.2e-7;
	private static final double RNMX = (1.0 - EPS);

	private long seed;
	private long idum2 = 123456789;
	private long iy = 0;
	private long iv[] = new long[32];

	public CustomRandomGenerator() {
		this.setSeed(System.currentTimeMillis());
	}

	@Override
	public void setSeed(long arg0) {
		this.seed = arg0;
	}

	// the random generator in [0,1)
	public double nextDouble() {
		int j;
		long k;
		double temp;

		if (seed <= 0) {
			if (-(seed) < 1)
				seed = 1;
			else
				seed = -(seed);
			idum2 = (seed);
			for (j = NTAB + 7; j >= 0; j--) {
				k = (seed) / IQ1;
				seed = IA1 * (seed - k * IQ1) - k * IR1;
				if (seed < 0)
					seed += IM1;
				if (j < NTAB)
					iv[j] = seed;
			}
			iy = iv[0];
		}

		k = (seed) / IQ1;
		seed = IA1 * (seed - k * IQ1) - k * IR1;
		if (seed < 0)
			seed += IM1;
		k = idum2 / IQ2;
		idum2 = IA2 * (idum2 - k * IQ2) - k * IR2;
		if (idum2 < 0)
			idum2 += IM2;
		j = (int) (iy / NDIV);
		iy = iv[j] - idum2;
		iv[j] = seed;
		if (iy < 1)
			iy += IMM1;
		if ((temp = AM * iy) > RNMX)
			return RNMX;
		else
			return temp;
	}
}