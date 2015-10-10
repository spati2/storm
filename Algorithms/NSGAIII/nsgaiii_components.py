import os, sys, inspect



cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe()))[0], "../..")))
print cmd_subfolder
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

from jmoo_algorithms import *
from jmoo_individual import *
from binary_crossover import sbx_crossover
from polynomial_mutation import pmutation

def nsgaiii_selector(problem, population):
    return population, 0

def nsgaiii_sbx(problem, population):
    from random import choice
    mutants = []
    for _ in xrange(len(population)/2):
        father = choice(population)
        while True:
            mother = choice(population)
            if father != mother: break
        child1, child2 = sbx_crossover(problem, father, mother)
        mchild1 = pmutation(problem, child1)
        mchild2 = pmutation(problem, child2)

        mutants.append(mchild1)
        mutants.append(mchild2)
    assert(len(mutants) == len(population)), "Length of the offspring population should be equal to the parent population"
    return mutants, 0


def nsgaiii_recombine(problem, population, selectees, k):
    evaluate_no = 0
    assert(len(population+selectees) == 2 * len(population)), "The recombination population should be 2 * len(population)"
    # Evaluate any new guys
    for individual in population+selectees:
            if not individual.valid:
                individual.evaluate()
                evaluate_no += 1
    # Format a population Data structure usable by DEAP's package
    dIndividuals = jmoo_algorithms.deap_format(problem, population+selectees)
    # Combine
    population = tools.selNSGA3(problem, dIndividuals, k)
    return population, evaluate_no