package uk.ac.essex.csp.algorithms.mo.ea;

import java.util.List;

import uk.ac.essex.csp.algorithms.moead.wa.Subproblem;

public interface IGenotypeListener {
	public void generationBegin(List<Subproblem> pop, int generation);
	public void generationEnd(List<Subproblem> pop, int generation);
	public void initFinish(List<Subproblem> subproblems);
}
