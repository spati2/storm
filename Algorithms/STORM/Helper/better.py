import os, sys, inspect
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe()))[0],"../../")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

from utility import loss

class Poles:
    def __init__(self, i, east, west):
        self.id = i
        self.east = east
        self.west = west

    def __str__(self):
        return str(self.__dict__)

def better(problem,individual,mutant):

    if len(individual.fitness.fitness) > 1:
        weights = []
        for obj in problem.objectives:
            # w is negative when we are maximizing that objective
            if obj.lismore:
                weights.append(+1)
            else:
                weights.append(-1)
        weighted_individual = [c*w for c,w in zip(individual.fitness.fitness, weights)]
        weighted_mutant = [c*w for c,w in zip(mutant.fitness.fitness, weights)]
        individual_loss = loss(weighted_individual, weighted_mutant, mins=[obj.low for obj in problem.objectives], maxs = [obj.up for obj in problem.objectives])
        mutant_loss = loss(weighted_mutant, weighted_individual, mins=[obj.low for obj in problem.objectives], maxs = [obj.up for obj in problem.objectives])

        if individual_loss < mutant_loss:
            return mutant, individual
        else:
            return individual, mutant  # otherwise
    else:
        assert(len(individual.fitness.fitness) == len(mutant.fitness.fitness)), "length of the objectives are not equal"
        if problem.objectives[-1].lismore:
            indi = 1e10 - individual.fitness.fitness[-1]
            mut = 1e10 - mutant.fitness.fitness[-1]
        else:
            indi = individual.fitness.fitness[-1]
            mut = mutant.fitness.fitness[-1]
        if indi >= mut:
            return individual, mutant
        else:
            return mutant, individual

def rearrange(problem, midpoint, stars):
    evaluated_stars = []
    unique_data = [list(x) for x in set(tuple(x) for x in [p.decisionValues for p in stars])]
    print "Length of stars: ", len(unique_data)
    stars = [[midpoint, point] for point in stars]



    for i, (east, west) in enumerate(stars):
        # print east.fitness.fitness, west.fitness.fitness
        if east.fitness.fitness is None: east.evaluate()
        if west.fitness.fitness is None: west.evaluate()
        east, west = better(problem, east, west)  # east is better than west
        evaluated_stars.append(Poles(i, east, west))
    return evaluated_stars

