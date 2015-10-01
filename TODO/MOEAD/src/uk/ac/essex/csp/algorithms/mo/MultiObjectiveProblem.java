package uk.ac.essex.csp.algorithms.mo;

import org.apache.commons.math.random.RandomData;

import uk.ac.essex.csp.algorithms.mo.ea.MoChromosome;

/**
 * An OO modeled MultiObjective Problem.
 * 
 * @author Wudong
 * 
 */
public interface MultiObjectiveProblem {
	public int getObjectiveSpaceDimension();

	public int getParameterSpaceDimension();

	// public ObjectiveFunction[] getObjectiveFunctions();

	// public double[] getRandomParameterPoint(RandomData randomObject);

	public MoChromosome createRandomMoChromosome(RandomData randomObject);

	// public SquareDimentionalSpace getDomain();

	public double[] getIdealPoint();

	public double[][] getRange();

	public double[][] getDomain();

	public void evaluate(MoChromosome chromosome);

	public void evaluate(double[] sp, double[] obj);

	public void evaluate(int[] sp, double[] obj);

	public String getName();
}
