package uk.ac.essex.csp.algorithms.moead;

import uk.ac.essex.csp.algorithms.mo.ea.MoChromosome;

/**
 * 
 * 
 * <p>
 * 2.11.07
 * <p>
 * Subproblem is modified to maintain two different population, the one for
 * searching and the one for selection. The idea here is: to maintain the
 * diversity of the searching population, not all better individual will be
 * updated. But for the purpose of selection, the best value need to be used.
 * 
 * 
 * @author wudong
 * 
 */
public class Subproblem {
	/**
	 * The index of this subproblem in the main population.
	 */
	public int mainpopIndex;

	/**
	 * The weight of this subproblem.
	 */
	public double[] weight;

	/**
	 * The current individual of this subproblem.
	 */
	public MoChromosome currentIndividual;

	/**
	 * The current objective value of this subproblem computed by the current
	 * individual.
	 */
	public double currentValue;

//	/**
//	 * The current best individual of this subproblem that encountered in the
//	 * searching process. Note that to maintain diversity of the population, the
//	 * current individual, which is used for searching, will not necessary be
//	 * updated when come cross a better individual, but the currentBest is
//	 * always updated with the newly better individual.
//	 * 
//	 */
//	public MoChromosome currentBestIndividual;
//	
//	/**
//	 * The current Best value of this subproblem computed by the current best
//	 * individual.
//	 */
//	public double currentBestValue;

	/**
	 * The neighbour of this subproblem.
	 */
	public Subproblem[] neighbours;
	
	public double subobjective(MoChromosome chromosome) {
		return 0;
	}
	
}
