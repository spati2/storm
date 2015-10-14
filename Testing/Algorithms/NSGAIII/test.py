import os, sys, inspect
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe()))[0], "../../..")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

from jmoo_core import *

def readpf(problem):
    filename = "./Testing/PF/" + problem.name.split("_")[0] + "(" + str(len(problem.objectives)) + ")-PF.txt"
    return [[float(num) for num in line.split()] for line in open(filename, "r").readlines()]

from Techniques.IGD_Calculation import IGD
algorithms = [jmoo_NSGAIII()]
Configurations = {
    "Universal": {
        "Repeats" : 10,
        "Population_Size" : 136,
        "No_of_Generations" : 1500
    },
    "NSGAIII": {
        "SBX_Probability": 1,
        "ETA_C_DEFAULT_" : 30,
        "ETA_M_DEFAULT_" : 20
    }
}

def problems_runner(list_args):
    problems = [list_args[0]]
    Configurations["Universal"]["Population_Size"] = list_args[1]
    Configurations["Universal"]["No_of_Generations"] = list_args[2]


    os.chdir("../../..")  # Since the this file is nested so the working directory has to be changed
    # Wrap the tests in the jmoo core framework
    tests = jmoo_test(problems, algorithms)
    IGD_Results = []
    random.seed(20)
    for problem in tests.problems:
        print problem.name, " ",
        for algorithm in tests.algorithms:
            for repeat in xrange(Configurations["Universal"]["Repeats"]):
                print repeat, " ",
                import sys
                sys.stdout.flush()
                initialPopulation(problem, Configurations["Universal"]["Population_Size"])
                statBox = jmoo_evo(problem, algorithm, Configurations)

                resulting_pf = [[float(f) for f in individual.fitness.fitness] for individual in statBox.box[-1].population]
                IGD_Results.append(IGD(resulting_pf, readpf(problem)))
                print IGD(resulting_pf, readpf(problem))
            IGD_Results = sorted(IGD_Results)

            results_string = ""
            results_string += "Problem Name: ", problem.name
            results_string += "Algorithm Name: ", algorithm.name
            results_string += "- Generated New Population"
            results_string += "- Ran the algorithm for ", Configurations["Universal"]["Repeats"]
            results_string += "- The SBX crossover and mutation parameters are correct"
            results_string += "Best: ", IGD_Results[0]
            results_string += "Worst: ", IGD_Results[-1]
            results_string += "Median: ", IGD_Results[int(len(IGD_Results)/2)]

            filename = "./Results/" + problem.name + ".txt"
            f = open(filename, "w")
            f.write(results_string)
            f.close()

def dtlz_7_3():
    Configurations["Universal"]["Population_Size"] = 92
    Configurations["Universal"]["No_of_Generations"] = 400

    problems = [dtlz1(7, 3)]
    os.chdir("../../..")  # Since the this file is nested so the working directory has to be changed
    # Wrap the tests in the jmoo core framework
    tests = jmoo_test(problems, algorithms)
    IGD_Results = []
    random.seed(20)
    for problem in tests.problems:
        for algorithm in tests.algorithms:
            for repeat in xrange(Configurations["Universal"]["Repeats"]):
                initialPopulation(problem, Configurations["Universal"]["Population_Size"])
                statBox = jmoo_evo(problem, algorithm, Configurations)

                resulting_pf = [[float(f) for f in individual.fitness.fitness] for individual in statBox.box[-1].population]
                IGD_Results.append(IGD(resulting_pf, readpf(problem)))
                print IGD(resulting_pf, readpf(problem))
            IGD_Results = sorted(IGD_Results)

            results_string = ""
            results_string += "Problem Name: ", problem.name
            results_string += "Algorithm Name: ", algorithm.name
            results_string += "- Generated New Population"
            results_string += "- Ran the algorithm for ", Configurations["Universal"]["Repeats"]
            results_string += "- The SBX crossover and mutation parameters are correct"
            results_string += "Best: ", IGD_Results[0]
            results_string += "Worst: ", IGD_Results[-1]
            results_string += "Median: ", IGD_Results[int(len(IGD_Results)/2)]

            filename = "./Results/" + problem.name + ".txt"
            f = open(filename, "w")
            f.write(results_string)
            f.close()


def dtlz1_9_5():
    Configurations["Universal"]["Population_Size"] = 212
    Configurations["Universal"]["No_of_Generations"] = 600

    problems = [dtlz1(9, 5)]
    os.chdir("../../..")  # Since the this file is nested so the working directory has to be changed
    # Wrap the tests in the jmoo core framework
    tests = jmoo_test(problems, algorithms)
    IGD_Results = []
    random.seed(20)
    for problem in tests.problems:
        for algorithm in tests.algorithms:
            for repeat in xrange(Configurations["Universal"]["Repeats"]):
                initialPopulation(problem, Configurations["Universal"]["Population_Size"])
                statBox = jmoo_evo(problem, algorithm, Configurations)

                resulting_pf = [[float(f) for f in individual.fitness.fitness] for individual in statBox.box[-1].population]
                IGD_Results.append(IGD(resulting_pf, readpf(problem)))
                print IGD(resulting_pf, readpf(problem))
            IGD_Results = sorted(IGD_Results)

            results_string = ""
            results_string += "Problem Name: ", problem.name
            results_string += "Algorithm Name: ", algorithm.name
            results_string += "- Generated New Population"
            results_string += "- Ran the algorithm for ", Configurations["Universal"]["Repeats"]
            results_string += "- The SBX crossover and mutation parameters are correct"
            results_string += "Best: ", IGD_Results[0]
            results_string += "Worst: ", IGD_Results[-1]
            results_string += "Median: ", IGD_Results[int(len(IGD_Results)/2)]

            filename = "./Results/" + problem.name + ".txt"
            f = open(filename, "w")
            f.write(results_string)
            f.close()

def dtlz1_12_8():
    Configurations["Universal"]["Population_Size"] = 156
    Configurations["Universal"]["No_of_Generations"] = 750

    problems = [dtlz1(12, 8)]
    os.chdir("../../..")  # Since the this file is nested so the working directory has to be changed
    # Wrap the tests in the jmoo core framework
    tests = jmoo_test(problems, algorithms)
    IGD_Results = []
    random.seed(20)
    for problem in tests.problems:
        for algorithm in tests.algorithms:
            for repeat in xrange(Configurations["Universal"]["Repeats"]):
                initialPopulation(problem, Configurations["Universal"]["Population_Size"])
                statBox = jmoo_evo(problem, algorithm, Configurations)

                resulting_pf = [[float(f) for f in individual.fitness.fitness] for individual in statBox.box[-1].population]
                IGD_Results.append(IGD(resulting_pf, readpf(problem)))
                print IGD(resulting_pf, readpf(problem))
            IGD_Results = sorted(IGD_Results)

            results_string = ""
            results_string += "Problem Name: ", problem.name
            results_string += "Algorithm Name: ", algorithm.name
            results_string += "- Generated New Population"
            results_string += "- Ran the algorithm for ", Configurations["Universal"]["Repeats"]
            results_string += "- The SBX crossover and mutation parameters are correct"
            results_string += "Best: ", IGD_Results[0]
            results_string += "Worst: ", IGD_Results[-1]
            results_string += "Median: ", IGD_Results[int(len(IGD_Results)/2)]

            filename = "./Results/" + problem.name + ".txt"
            f = open(filename, "w")
            f.write(results_string)
            f.close()


def dtlz1_14_10():
    Configurations["Universal"]["Population_Size"] = 276
    Configurations["Universal"]["No_of_Generations"] = 1000

    problems = [ dtlz1(14, 10)]
    os.chdir("../../..")  # Since the this file is nested so the working directory has to be changed
    # Wrap the tests in the jmoo core framework
    tests = jmoo_test(problems, algorithms)
    IGD_Results = []
    random.seed(20)
    for problem in tests.problems:
        for algorithm in tests.algorithms:
            for repeat in xrange(Configurations["Universal"]["Repeats"]):
                initialPopulation(problem, Configurations["Universal"]["Population_Size"])
                statBox = jmoo_evo(problem, algorithm, Configurations)

                resulting_pf = [[float(f) for f in individual.fitness.fitness] for individual in statBox.box[-1].population]
                IGD_Results.append(IGD(resulting_pf, readpf(problem)))
                print IGD(resulting_pf, readpf(problem))
            IGD_Results = sorted(IGD_Results)

            results_string = ""
            results_string += "Problem Name: ", problem.name
            results_string += "Algorithm Name: ", algorithm.name
            results_string += "- Generated New Population"
            results_string += "- Ran the algorithm for ", Configurations["Universal"]["Repeats"]
            results_string += "- The SBX crossover and mutation parameters are correct"
            results_string += "Best: ", IGD_Results[0]
            results_string += "Worst: ", IGD_Results[-1]
            results_string += "Median: ", IGD_Results[int(len(IGD_Results)/2)]

            filename = "./Results/" + problem.name + ".txt"
            f = open(filename, "w")
            f.write(results_string)
            f.close()


# dtlz1_14_10()
problems = [
    [dtlz1(7,3), 92, 400], [dtlz1(9, 5), 212, 600], [dtlz1(12, 8), 156, 750], [dtlz1(14, 10), 276, 1000], [dtlz1(19, 15), 136, 1500],
    [dtlz2(12, 3), 92, 250], [dtlz2(14, 5), 212, 350], [dtlz2(17, 8), 156, 500], [dtlz2(19, 10), 276, 750], [dtlz2(24, 15), 136, 1000],
    [dtlz3(12, 3), 92, 1000], [dtlz3(14, 5), 212, 1000], [dtlz3(17, 8), 156, 1000], [dtlz3(19, 10), 276, 1500], [dtlz3(24, 15), 136, 2000],
    [dtlz4(12, 3), 92, 600], [dtlz4(14, 5), 212, 1000], [dtlz4(17, 8), 156, 1250], [dtlz4(19, 10), 276, 2000], [dtlz4(24, 15), 136, 3000],
        ]

for problem in problems:
    problems_runner(problem)