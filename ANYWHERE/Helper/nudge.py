from mutator import trim

import os, inspect, sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from jmoo_individual import jmoo_individual

def nudge(problem, individual, east, increment):
    if individual.anyscore == 1e32:
        return individual
    temp = []
    for i, decision in enumerate(problem.decisions):
        up = decision.up
        low = decision.low
        mutation = increment * (east.decisionValues[i] - individual.decisionValues[i])
        temp.append(trim(individual.decisionValues[i] + mutation, low, up))
    return jmoo_individual(problem, temp, None)