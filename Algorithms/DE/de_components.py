import os
import sys
import inspect
import random

from jmoo_individual import *
from jmoo_algorithms import *
from jmoo_stats_box import *


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import jmoo_properties
from jmoo_preprocessor import CULLING, NEW_DE


def three_others(individuals, one):
    seen = [one]

    def other():
        while True:
            random_selection = random.randint(0, len(individuals) - 1)
            if individuals[random_selection] not in seen:
                seen.append(individuals[random_selection])
                break
        return individuals[random_selection]

    return other(), other(), other()


def trim(mutated, low, up):
    return max(low, min(mutated, up))

def crossover(problem, candidate_a, candidate_b):
    assert(len(candidate_a) == len(candidate_b)), "Candidate length are not the same"
    crossover_point = random.randrange(1, len(candidate_a), 1)
    assert(crossover_point < len(candidate_a)), "Crossover point has gone overboard"
    mutant = list(candidate_a[:crossover_point])
    mutant.extend(list(candidate_b[crossover_point:]))
    assert(len(mutant) == len(candidate_a)), "Mutant created doesn't have the same length as candidates"
    return mutant

def extrapolate(problem, individuals, one, f, cf):
    # #print "Extrapolate"
    two, three, four = three_others(individuals, one)
    # #print two,three,four
    solution = []
    for d, decision in enumerate(problem.decisions):
        assert isinstance(two, jmoo_individual)
        x, y, z = two.decisionValues[d], three.decisionValues[d], four.decisionValues[d]
        if random.random() < cf:
            mutated = x + f * (y - z)
            solution.append(trim(mutated, decision.low, decision.up))
        else:
            solution.append(one.decisionValues[d])

    if NEW_DE is True:
        if random.random() < cf:
            solution = crossover(problem, one.decisionValues, solution)


    return jmoo_individual(problem, [float(d) for d in solution], None)

def better(problem,individual,mutant):
    if len(individual.fitness.fitness) > 1:
        # From Joe: Score the poles
        # print "### Individual: ", individual.fitness.fitness
        # print "### Mutant: ", mutant.fitness.fitness
        n = len(problem.decisions)
        weights = []
        for obj in problem.objectives:
            # w is negative when we are maximizing that objective
            if obj.lismore:
                weights.append(+1)
            else:
                weights.append(-1)
        weighted_individual = [c*w for c,w in zip(individual.fitness.fitness, weights)]
        weighted_mutant = [c*w for c,w in zip(mutant.fitness.fitness, weights)]
        individual_loss = loss(weighted_individual, weighted_mutant, mins = [obj.low for obj in problem.objectives], maxs = [obj.up for obj in problem.objectives])
        mutant_loss = loss(weighted_mutant, weighted_individual, mins = [obj.low for obj in problem.objectives], maxs = [obj.up for obj in problem.objectives])

        if individual_loss < mutant_loss:
            # print ">>", mutant.fitness.fitness, individual.fitness.fitness
            return mutant
        else:
            # print ">>", individual.fitness.fitness, mutant.fitness.fitness
            return individual  # otherwise
    else:
        # print "Binary"
        assert(len(individual.fitness.fitness) == len(mutant.fitness.fitness)), "length of the objectives are not equal"
        # print  problem.objectives[-1].lismore
        if problem.objectives[-1].lismore:
            indi = 100 - individual.fitness.fitness[-1]
            mut = 100 - mutant.fitness.fitness[-1]
        else:
            indi = individual.fitness.fitness[-1]
            mut = mutant.fitness.fitness[-1]
        if indi >= mut:
            # print ">>", mutant.fitness.fitness, individual.fitness.fitness
            return individual
        else:
            # print ">>", individual.fitness.fitness, mutant.fitness.fitness
            return mutant




def de_selector(problem, individuals):
    #print "selector"
    newer_generation = []
    for individual in individuals:
        if not individual.valid:
            individual.evaluate()
    no_evals = 0
    #print "Length of population: ", len(individuals)
    #print "F: ", jmoo_properties.F
    #print "CF: ", jmoo_properties.CF
    for individual in individuals:
        #print "Old Decision: ",individual.decisionValues
        #print "Old Score: ", individual.fitness.fitness
        if CULLING is True:
            count = 0
            while True:
                mutant = extrapolate(problem, individuals, individual, jmoo_properties.F, jmoo_properties.CF)
                mutant.evaluate()
                temp = mutant.fitness.fitness
                count += 1
                if (temp[0] > 66 and temp[1] < 33) or count > 10:
                    if count < 10:
                        print "SUCCESS"
                    no_evals += count
                    break
        else:
            mutant = extrapolate(problem, individuals, individual, jmoo_properties.F, jmoo_properties.CF)
            mutant.evaluate()
            #print mutant.fitness.fitness
            no_evals += 1

        ##print "Mutant Decisions: ",mutant.decisionValues
        #print "New Score: ", mutant.fitness.fitness
        newer_generation.append(better(problem, individual, mutant))

    # print len(newer_generation)
    return newer_generation, no_evals

#Vivek: This is just a stub
def de_mutate(problem, population):
    #print "mutate"
    # print "mutated: Length of the Population: ",len(population)
    #for i,p in enumerate(population):
    #    print ">>", i, sum(p.fitness.fitness)
    #print "Minimum in population: ", min([sum(p.fitness.fitness) for p in population])

    return population, 0

#Vivek: This is just a stub
def de_recombine(problem, unusedSlot, mutants, MU):
    # print "recombine: Length of population: ", len(mutants)
    return mutants, 0