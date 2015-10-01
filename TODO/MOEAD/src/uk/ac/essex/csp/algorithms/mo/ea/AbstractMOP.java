package uk.ac.essex.csp.algorithms.mo.ea;

import uk.ac.essex.csp.algorithms.mo.MultiObjectiveProblem;

public abstract class AbstractMOP implements MultiObjectiveProblem {
	protected String name;
	protected double[][] domain;
	protected double[][] range;
	protected double[] idealpoint;
	protected int objDimension;
	protected int parDimension;

	public String getName() {
		return name;
	}

	public double[] getIdealPoint() {
		return idealpoint;
	}

	public int getObjectiveSpaceDimension() {
		return objDimension;
	}

	public int getParameterSpaceDimension() {
		return parDimension;
	}

	public double[][] getRange() {
		return range;
	}

	public double[][] getDomain() {
		return domain;
	}
	
	protected abstract void init();
	
	protected abstract MoChromosome createMoChromosomeInstance();

}
