package uk.ac.essex.csp.algorithms.statictics;

import java.util.Random;

/**
 * http://www.techuser.net/randpermgen.html
 * 
 *
 */
public class Permutation {
	/**
	 * 
	 * @param size
	 * @param length
	 * @return
	 */
	public static int[][] randompermutate(int size, int length){
		int [][] points = new int[size][length];
		for (int i=0; i<size; i++){
			points[i] = randompermutate(length, null);
		}
		return points;
	}
	
	/**
	 * Determine if the given array is a permutation.
	 */
	public static boolean isPermutation(int[] permutation){
		return false;
	}
	
	public static boolean samePermutation(int[] p1, int[] p2){
		if (p1.length !=p2.length)return false;
		for (int i=0; i<p1.length; i++){
			if (p1[i]!=p2[i])return false;
		}
		return true;
	}
	
	/**
	 * 
	 * @param length
	 * @return
	 */
	public static int[] randompermutate(int length, int[] permutateseed){
		int[] permutate = new int[length];
		if (permutateseed!=null){
			if (permutateseed.length==length){
				System.arraycopy(permutateseed, 0, permutate, 0, length);
			}
		}else{
			for (int i=1; i<=length; i++){
				permutate[i-1] = i;
			}
		}
		Random random = new Random(); 
		//for (i=1 to (n-1)) swap(ar[i], ar[Random(i,n)]);
		for (int i=0; i<length-1; i++){
			int valueSwap = permutate[i];
			int j = random.nextInt(length-i)+i;
			permutate[i]=permutate[j];
			permutate[j]=valueSwap;
		}
		return permutate;
	}
}
