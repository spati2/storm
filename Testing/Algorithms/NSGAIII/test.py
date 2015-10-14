import os, sys, inspect
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe()))[0], "../../..")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

from jmoo_core import *

def readpf(problem):
    filename = "./Testing/PF/" + problem.name.split("_")[0] + "(" + str(len(problem.objectives)) + ")-PF.txt"
    return [[float(num) for num in line.split()] for line in open(filename, "r").readlines()]


def convert_jmoo(pareto_fronts):
    tpopulation = []
    for front_no, front in enumerate(pareto_fronts[:1]):
        for i, dIndividual in enumerate(front):
            cells = []
            for j in xrange(len(dIndividual)):
                cells.append(dIndividual[j])
            tpopulation.append(jmoo_individual(problem, cells, dIndividual.fitness.values))
    return tpopulation

from Techniques.IGD_Calculation import IGD
algorithms = [jmoo_NSGAIII_New()]
problems = [dtlz1(7, 3)]
os.chdir("../../..")  # Since the this file is nested so the working directory has to be changed

from Algorithms.DEAP.tools.emo import sortNondominated

# Wrap the tests in the jmoo core framework
tests = jmoo_test(problems, algorithms)
IGD_Results = []
random.seed(20)
for problem in tests.problems:
    for algorithm in tests.algorithms:
        for repeat in xrange(repeats):
            initialPopulation(problem, MU)
            statBox = jmoo_evo(problem, algorithm, repeat)

            # Individuals = jmoo_algorithms.deap_format(problem, statBox.box[-1].population)
            # pareto_fronts = sortNondominated(Individuals, jmoo_properties.MU)
            # result_population = convert_jmoo(pareto_fronts)

            resulting_pf = [[float(f) for f in individual.fitness.fitness] for individual in statBox.box[-1].population]
            IGD_Results.append(IGD(resulting_pf, readpf(problem)))
            print IGD(resulting_pf, readpf(problem))
        IGD_Results = sorted(IGD_Results)
        print "Problem Name: ", problem.name
        print "Algorithm Name: ", algorithm.name
        print "- Generated New Population"
        print "- Ran the algorithm for ", repeats
        print "- The SBX crossover and mutation parameters are correct"
        print "Best: ", IGD_Results[0]
        print "Worst: ", IGD_Results[-1]
        print "Median: ", IGD_Results[int(len(IGD_Results)/2)]


# jmetal = [[0.292,0.042,0.167],
# [0,0.208,0.292],
# [0.125,0.043,0.333],
# [0.042,0.334,0.125],
# [0.334,0.042,0.125],
# [0.334,0.125,0.042],
# [0.083,0.125,0.292],
# [0.125,0.334,0.042],
# [0.167,0.334,0],
# [0.042,0.167,0.292],
# [0,0.459,0.042],
# [0.168,0.126,0.207],
# [0.25,0.126,0.125],
# [0.208,0,0.292],
# [0.292,0,0.209],
# [0.042,0,0.459],
# [0.167,0.209,0.126],
# [0.209,0.125,0.167],
# [0,0.041,0.46],
# [0.083,0.083,0.334],
# [0.042,0.376,0.083],
# [0.249,0.085,0.167],
# [0.25,0.251,0],
# [0,0.125,0.375],
# [0.376,0,0.125],
# [0.209,0.041,0.25],
# [0.083,0.209,0.209],
# [0.042,0.042,0.417],
# [0,0.25,0.25],
# [0.167,0.083,0.251],
# [0.376,0.084,0.042],
# [0.125,0.125,0.25],
# [0.126,0,0.375],
# [0,0,0.501],
# [0.125,0.084,0.292],
# [0,0.292,0.209],
# [0.042,0.251,0.209],
# [0.125,0.209,0.167],
# [0.167,0,0.334],
# [0.417,0.042,0.042],
# [0,0.167,0.334],
# [0.209,0.209,0.083],
# [0,0.501,0],
# [0,0.334,0.167],
# [0.083,0.042,0.376],
# [0.083,0.376,0.042],
# [0.043,0.417,0.042],
# [0.167,0.292,0.042],
# [0.25,0.04,0.21],
# [0.124,0.168,0.209],
# [0.292,0.209,0],
# [0.042,0.292,0.167],
# [0.083,0.167,0.251],
# [0.21,0.083,0.208],
# [0.459,0.042,0],
# [0.125,0.251,0.125],
# [0.167,0.042,0.292],
# [0.125,0.375,0],
# [0.167,0.25,0.084],
# [0.042,0.084,0.375],
# [0.376,0.125,0],
# [0.209,0.292,0],
# [0.209,0.251,0.041],
# [0.208,0.168,0.125],
# [0.417,0,0.083],
# [0.334,0.167,0],
# [0.25,0.209,0.042],
# [0.041,0.209,0.251],
# [0.334,0.083,0.084],
# [0.041,0.459,0],
# [0.167,0.167,0.167],
# [0.292,0.167,0.042],
# [0,0.376,0.125],
# [0.292,0.084,0.125],
# [0.334,0.001,0.167],
# [0.251,0,0.251],
# [0,0.084,0.417],
# [0.084,0.418,0],
# [0.292,0.126,0.083],
# [0.251,0.167,0.084],
# [0.501,0,0],
# [0.376,0.042,0.083],
# [0.083,0.334,0.084],
# [0,0.417,0.083],
# [0.417,0.084,0],
# [0.459,0,0.042],
# [0.084,0.25,0.167],
# [0.083,0.292,0.125],
# [0.042,0.125,0.334],
# [0.125,0.292,0.083],
# [0.083,0,0.417],
# [0.235,0.217,0.048]]
#
#
#
# print IGD(jmetal, readpf(dtlz1(7, 3)))
