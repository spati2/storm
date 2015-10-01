package uk.ac.essex.csp.algorithms.mo;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

import uk.ac.essex.csp.algorithms.mo.ea.MoChromosome;

/**
 * This the implementation of the algorithms for choosing non-dominate Set.
 * 
 * 
 * Algorithms, form the "Multi-Objective Optimization using Evolutionary
 * Algorithms", by Kalyanmoy Deb, Page 37.
 * 
 * 
 * @param <T>
 */
public class NonDominationSelector<T> {
	private DominationDeterminer<T> dominator;

	private List<T> collection;

	/**
	 * Set the DominationDetermine object for the non-domination selection.
	 * 
	 * @param d
	 */
	public void setDominator(DominationDeterminer<T> d) {
		this.dominator = d;
	}

	/**
	 * Set the colection of object to be select from.
	 * 
	 * @param col a collection of objects.
	 */
	public void setChoosingSet(Collection<T> col) {
		if (collection==null) {
			collection = new LinkedList<T>();
		}else {
			collection.clear();
		}
		
		Iterator<T> name = col.iterator();
		while(name.hasNext()) {
			collection.add(name.next());
		}
	}

	/**
	 * Select the non-domination subset from the given selection.
	 * @return the selection result, a non-domination 
	 */
	public List<T> select() {
		assert dominator != null;
		assert collection != null;

		List<T> result = new ArrayList<T>();
		if (collection.size() == 0)
			return result;
		result.add(collection.get(0));
		int counter = 1;
		out: while (counter < collection.size()) {
			int jj = 0;
			T chromosome2 = collection.get(counter);
			int resultsize = result.size();
			boolean remove[] = new boolean[resultsize];
			while (jj < resultsize) {
				T chromosome1 = result.get(jj);
				if (dominator.dominate(chromosome2, chromosome1)) {
					remove[jj] = true;
					// result.remove(jj);
				} else if (dominator.dominate(chromosome1, chromosome2)) {
					counter++;
					break out;
				}
				jj++;
			}
			for (int i = remove.length - 1; i >= 0; i--) {
				if (remove[i])
					result.remove(i);
			}
			result.add(chromosome2);
			counter++;
		}
		return result;
	}
}
