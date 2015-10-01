package uk.ac.essex.csp.algorithms.statictics;

import java.util.Arrays;
import java.util.Random;

/**
 * SLHD is described in
 * 
 * <pre>
 *            K.Q.Ye, W.Li, and A.Sudjianto, 
 *            &quot;Algorithmic construction of optimal symmetric latin hypercube design&quot;
 *            J. Statis. Planning Inference, vol. 90, pp.145-159, 2000.
 * </pre>
 * 
 * Characteristics(Same as regular LHD):
 * <ul>
 * <li>The user can specify the number of design points.
 * <li>If project the design points onto any single dimension, the result is a
 * regular grid in one dimension.
 * </ul>
 * 
 * 
 * 
 */
public class SLHD {

	/**
	 * The procedure is to constructing randomly generated SLHDs of size m in a
	 * hypercube of d dimensions. It's abstract from paper Regis2004.
	 * 
	 * However, this is just a randomly generated SLHD, nothing more is doing to
	 * gaurantee its quality such as point distance, and entropy.
	 * 
	 * This algorithms generate a set of levels for each dimension. On every
	 * dimension, the space is diveded into levels. and the design points is the
	 * crossing of all that levels on every dimension.
	 * 
	 * TODO, as noted, this is just a randomly genearte SLHD, algorithms for one
	 * with good quality should also be implementation. This is can be simplely
	 * generatd by randomly generate largely quantity of SLHD's and choose the
	 * best one according to criteria.
	 * 
	 * @param levels
	 *            the number of levels to divides the dimension, also is the
	 *            number of points of the design to be generated.
	 * @param dimension
	 *            the dimension of the space.
	 * @return the resulting design points that is a SLHD.
	 */
	public static int[][] randomslhd(int levels, int dimension) {
		int[][] M = new int[levels][dimension];
		// if size is old. the middle one should never be exchanged.
		if ((levels % 2) == 0) {
			int middle = (levels + 1) / 2;
			for (int j = 1; j <= dimension; j++) {
				M[middle - 1][j - 1] = middle;
			}
		}
		int middle = (levels) / 2;
		// for every dimension, generate a permutation from 1 to middle.
		int[][] permutes = Permutation.randompermutate(dimension, middle);
		Random random = new Random();
		for (int i = 1; i <= middle; i++) {
			for (int j = 1; j <= dimension; j++) {
				double d = random.nextDouble();
				// swap one element with another, and also the corresponding
				// one.
				if (d <= 0.5) {
					M[i - 1][j - 1] = permutes[j - 1][i - 1];
					M[levels + 1 - i - 1][j - 1] = levels + 1
							- permutes[j - 1][i - 1];
				} else {
					M[levels + 1 - i - 1][j - 1] = permutes[j - 1][i - 1];
					M[i - 1][j - 1] = levels + 1 - permutes[j - 1][i - 1];
				}
			}
		}
		return M;
	}
	
	public static boolean isSLHD(int[][] matrix){
		int midsize = matrix.length / 2;
		int[][] ks = invertMatrix(matrix);
		int oppositePosition = -1;
		for (int i=0; i<midsize; i++){
			int[] row = matrix[i];
			for (int j=0; j<row.length; j++){
				int k=-2;
				//Find the opposite position.
				for (int m=0; m<ks[j].length; m++)
					if (ks[j][m]==matrix.length+1-row[j])
						k = m;
				if (k==-2)return false;
				
				if (oppositePosition==-1){
					oppositePosition = k;
				}else{
					if (k!=oppositePosition)
						return false;
				}
			}
			oppositePosition = -1;
		}
		return true;
	}
	
	public static int[][] invertMatrix(int[][] matrix){
		int rowsize = matrix.length;
		int columnsize = matrix[0].length;
		
		int[][] invertMatrix = new int[columnsize][rowsize];
		for (int i=0; i<columnsize; i++){
			for (int j=0; j<rowsize; j++){
				invertMatrix[i][j] = matrix[j][i];
			}
		}
		return invertMatrix;
	}
	
	public static boolean isLHD(int[][] matrix){
		return false;
	}
	
	public static double levelMaping(int levels, int level, double lowEnd,
			double highEnd) {
		double averageLevel = (highEnd - lowEnd) / (levels + 1);
		return lowEnd + (level) * averageLevel;
	}
}
