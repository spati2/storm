import os, sys, inspect
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe()))[0], "../../..")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
from jmoo_properties import *
from jmoo_core import *


def readpf(problem):
    filename = "./Testing/PF/" + problem.name.split("_")[0] + "(" + str(len(problem.objectives)) + ")-PF.txt"
    print filename
    for line in open(filename, "r").readlines():
        print line.split()
    exit()
    # true_PF = [[float(x) for x in line.split(",")] for line in open(filename, "r").readlines()]
    return true_PF



algorithms = [jmoo_NSGAIII()]
problems =[dtlz1(9, 5)]
os.chdir("../../..")  # Since the this file is nested so the working directory has to be changed
for l in readpf(problems[0]):
    print l
    exit()

from Techniques.IGD_Calculation import IGD


# Wrap the tests in the jmoo core framework
tests = jmoo_test(problems, algorithms)
for problem in tests.problems:
    initialPopulation(problem, MU)
    for algorithm in tests.algorithms:
        for repeat in range(repeats):
            statBox = jmoo_evo(problem, algorithm, repeat)
        print "Problem Name: ", problem.__name__
        print "Algorithm Name: ", algorithm.__name__
        print "- Generated New Population"
        print "- Ran the algorithm for ", repeat

