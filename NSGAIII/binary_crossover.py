from __future__ import division
import random

import os, sys, inspect
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe()))[0], "../..")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

from jmoo_individual import jmoo_individual


EPS = 1.0e-14

def get_betaq(rand, alpha, eta=30):
    betaq = 0.0
    if rand <= (1.0/alpha):
        betaq = (rand * alpha) ** (1.0/(eta+1.0))
    else:
        betaq = (1.0/(2.0 - rand*alpha)) ** (1.0/(eta+1.0))
    return betaq

def sbx_crossover(problem, parent1, parent2, cr=1, eta=30):
    assert(len(parent1.decisionValues) == len(parent2.decisionValues)), "Parents are sick"

    from copy import deepcopy
    child1 = [0 for _ in xrange(len(parent1.decisionValues))]
    child2 = [0 for _ in xrange(len(parent1.decisionValues))]

    if random.random() > cr: return parent1, parent2
    for index in xrange(len(parent1.decisionValues)):

        # import pdb
        # pdb.set_trace()

        # Should these variables be considered for crossover
        if random.random() > 0.5:
            child1[index] = parent1.decisionValues[index]
            child2[index] = parent2.decisionValues[index]
            continue

        # Are these variable the same
        # print parent1.decisionValues[index], parent2.decisionValues[index], parent1.decisionValues[index] - parent2.decisionValues[index]
        if parent1.decisionValues[index] - parent2.decisionValues[index] <= EPS:
            child1[index] = parent1.decisionValues[index]
            child2[index] = parent2.decisionValues[index]
            continue

        lower_bound = problem.decisions[index].low
        upper_bound = problem.decisions[index].up

        y1 = min(parent1.decisionValues[index], parent2.decisionValues[index])
        y2 = max(parent1.decisionValues[index], parent2.decisionValues[index])
        random_no = random.random()

        # child 1
        beta = 1.0 + (2.0 * (y1 - lower_bound)/(y2 - y1))
        alpha = 2.0 - beta ** (-(eta+1.0))
        betaq = get_betaq(random_no, alpha, eta)

        child1[index] = 0.5 * ((y1 + y2) - betaq * (y2 - y1))

        # child 2
        beta = 1.0 + (2.0 * (upper_bound-y2)/(y2-y1))
        alpha = 2.0 - beta ** -(eta+1.0)
        betaq = get_betaq(random_no, alpha, eta)

        child2[index] = 0.5 * ((y1 + y2) + betaq * (y2 - y1))

        child1[index] = max(lower_bound, min(child1[index], upper_bound))
        child2[index] = max(lower_bound, min(child2[index], upper_bound))


    return jmoo_individual(problem, child1), jmoo_individual(problem, child2)


