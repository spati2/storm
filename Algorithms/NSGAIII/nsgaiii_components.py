import os, sys, inspect
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe()))[0], "../DEAP")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

from deap import base
from deap import creator
from deap import tools

from jmoo_individual import *
from binary_crossover import sbx_crossover
from polynomial_mutation import pmutation

cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe()))[0], "..")))
# grep print cmd_subfolder
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

from jmoo_algorithms import *


def nsgaiii_selector(problem, population):
    return population, 0

def nsgaiii_sbx(problem, population):
    from random import choice
    mutants = []
    for _ in xrange((len(population)/2)+1):
        father = choice(population)
        while True:
            mother = choice(population)
            if father != mother: break
        child1, child2 = sbx_crossover(problem, father, mother)
        mchild1 = pmutation(problem, child1)
        mchild2 = pmutation(problem, child2)

        # print father.decisionValues
        # print mother.decisionValues
        # print
        # print child1.decisionValues
        # print child2.decisionValues
        # print
        # print mchild1.decisionValues
        # print mchild2.decisionValues
        # exit()

        mutants.append(mchild1)
        mutants.append(mchild2)

    return mutants, 0








def nsgaiii_recombine(problem, population, selectees, k):
    evaluate_no = 0


    # Evaluate any new guys
    for individual in population+selectees:
        try:
            if not individual.valid:
                individual.evaluate()
                evaluate_no += 1
        except:
            print individual
            import pdb
            pdb.set_trace()

    # Format a population data structure usable by DEAP's package
    dIndividuals = jmoo_algorithms.deap_format(problem, population+selectees)

    # Combine
    population = tools.selNSGA3(problem, dIndividuals, k)

    return population, evaluate_no