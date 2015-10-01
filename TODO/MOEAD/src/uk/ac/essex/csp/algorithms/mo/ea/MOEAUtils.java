package uk.ac.essex.csp.algorithms.mo.ea;

import java.util.Comparator;
import java.util.List;

import uk.ac.essex.csp.algorithms.Sorting;

public class MOEAUtils {

	public static void AssignNSGA2CrowdingDistance(List<MoChromosome> chroms) {
		final class MoChromosomeObjectiveComparator implements
				Comparator<MoChromosome> {
			private int i;

			MoChromosomeObjectiveComparator(int i) {
				this.i = i;
			}

			public int compare(MoChromosome o1, MoChromosome o2) {
				if (o1.objectivesValue[i] == o2.objectivesValue[i])
					return 0;
				return o1.objectivesValue[i] > o2.objectivesValue[i] ? 1 : -1;
			}
		}

		int size = chroms.size();
		MoChromosome moChromosome = chroms.get(0);
		int ojbdimension = moChromosome.objectDimension;

		for (MoChromosome chrom : chroms)
			chrom.crdistance = 0;

		for (int i = 0; i < ojbdimension; i++) {
			int[] sorting = Sorting.sorting(chroms,
					new MoChromosomeObjectiveComparator(i));
			MoChromosome chromfirst = chroms.get(sorting[0]);
			chromfirst.crdistance = Double.MAX_VALUE;
			double lowest = chromfirst.objectivesValue[i];

			MoChromosome chromlast = chroms.get(sorting[size - 1]);
			chromlast.crdistance = Double.MAX_VALUE;
			double highest = chromlast.objectivesValue[i];

			for (int j = 1; j < size - 1; j++) {
				chroms.get(sorting[j]).crdistance += (chroms
						.get(sorting[j + 1]).objectivesValue[i] - chroms
						.get(sorting[j - 1]).objectivesValue[i])
						/ (highest - lowest);
			}
		}
	}
}
