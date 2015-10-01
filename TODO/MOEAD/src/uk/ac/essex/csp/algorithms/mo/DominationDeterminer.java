package uk.ac.essex.csp.algorithms.mo;

import java.util.ArrayList;
import java.util.List;

/**
 * To determine if one object dominate another based on a set of evaluator for
 * them. New evaluater can be added then used to evaluate the value of the
 * object.
 * 
 * @author Wudong
 * 
 * @param <T>
 */
public class DominationDeterminer<T> {
	private List<Evaluator<T>> evaluators = new ArrayList<Evaluator<T>>();

	public void addEvaluator(Evaluator<T> e) {
		evaluators.add(e);
	}

	/**
	 * Determine if one object dominate another given a set of evaluators. For
	 * one object to dominate another, it's better in the sense of every
	 * evaluator.
	 * 
	 * @param a
	 * @param b
	 * @return
	 */
	public boolean dominate(T a, T b) {
		boolean a_is_worse = false;
		boolean equals = true;
		for (int i = 0; i < evaluators.size() && !a_is_worse; i++) {
			Evaluator<T> ff = evaluators.get(i);
			double fitnessValueA = ff.evaluate(a);
			double fitnessValueB = ff.evaluate(b);
			a_is_worse = ff.isToMinimize() ? (fitnessValueA > fitnessValueB)
					: (fitnessValueA < fitnessValueB);
			equals = (fitnessValueA == fitnessValueB) && equals;
		}
		return (!equals && !a_is_worse);
	};

	/**
	 * Determine if two object are same according to the given set of evaluator.
	 * For two objects to be same, the results from every evaluator of the two
	 * objects should be same.
	 * 
	 * @param a
	 * @param b
	 * @return ture if two objects are consider to be equal.
	 */
	public boolean equals(T a, T b) {
		boolean equals = true;
		for (int i = 0; i < evaluators.size(); i++) {
			Evaluator<T> ff = evaluators.get(i);
			equals = (ff.evaluate(b) == ff.evaluate(a)) && equals;
		}
		return equals;
	}
	
	public int getDimension() {
		return evaluators.size();
	}

	/**
	 * The interface is used internally for the DominationCaculator. It's
	 * responsible to evaluate the vaule of the object to a doulbe result.
	 * 
	 * @author Wudong
	 * 
	 * @param <T>
	 */
	public static interface Evaluator<T> {
		/**
		 * 
		 * @param t
		 * @return
		 */
		double evaluate(T t);

		/**
		 * determine if min value is better or max value is better for the
		 * evaluator.
		 * 
		 * @return true if min value is better.
		 */
		boolean isToMinimize();
	}
	
	public static boolean dominate(double[] oa, double[] ob, boolean toMin) {
		boolean a_is_worse = false;
		boolean equals = true;
		for (int i = 0; i < oa.length && !a_is_worse; i++) {
			double fitnessValueA = oa[i];
			double fitnessValueB = ob[i];
			a_is_worse = toMin ? (fitnessValueA > fitnessValueB)
					: (fitnessValueA < fitnessValueB);
			equals = (fitnessValueA == fitnessValueB) && equals;
		}
		return (!equals && !a_is_worse);
	}
	
	public static boolean equals(double[] oa, double[] ob) {
		boolean equals = true;
		for (int i = 0; i < oa.length; i++) {
			equals = (oa[i] == ob[i]) && equals;
		}
		return equals;
	}
}
