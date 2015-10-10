from __future__ import division
from os.path import realpath, abspath, split, join
from inspect import currentframe, getfile
from sys import path


cmd_subfolder = realpath(abspath(join(split(getfile(currentframe()))[0], "..")))
if cmd_subfolder not in path: path.insert(0, cmd_subfolder)
print "2new", cmd_subfolder
from DEAP.tools.emo import sortNondominated


cmd_subfolder = realpath(abspath(join(split(getfile(currentframe()))[0], "../..")))
if cmd_subfolder not in path: path.insert(0, cmd_subfolder)
print "new", cmd_subfolder
from jmoo_individual import *
from jmoo_properties import *
from jmoo_algorithms import *
from random import random


# ---------------------- Helper --------------------------------

class Node(object):
    def __init__(self, data=-1, level=0, parent=None):
        self.data = data
        self.level = level
        self.children = []
        self.parent = parent

    def add_child(self, obj):
        self.children.append(obj)


def tree(node, n, p, level=0):
    if level == 0:
        from numpy import arange
        for i in [j for j in arange(0, 1 + 10e-10, 1 / p)]:
            node.add_child(Node(i, level + 1))
        for child in node.children:
            tree(child, n, p, level + 1)
    elif level < (n - 1):
        other_beta = 0

        # Traversing up the tree to get other values of beta
        temp = node
        while temp is not None:
            other_beta += temp.data
            temp = temp.parent

        k = (1 - other_beta) / (1 / p)
        from numpy import arange
        for i in [j for j in arange(0, k * (1 / p) + 10e-10, (1 / p))]:
            node.add_child(Node(i, level + 1, node))
        for child in node.children:
            tree(child, n, p, level + 1)
    elif level == (n - 1):
        other_beta = 0
        # Traversing up the tree to get other values of beta
        temp = node
        while temp is not None:
            other_beta += temp.data
            temp = temp.parent
        node.add_child(Node(1 - other_beta, level + 1, node))

    else:
        return


class reference_point:
    def __init__(self, id, coordinates):
        self.id = id
        self.coordinates = coordinates

    def __str__(self):
        s = "id: " + str(self.id) + "\n"
        s += "coordinates: " + str(self.coordinates) + "\n"
        return s


def get_ref_points(root):
    ref_points = []
    assert (root.data == -1 and root.level == 0), "Supplied node is not root"
    visited, stack = set(), [root]
    count = 0
    while len(stack) != 0:
        vertex = stack.pop()
        if vertex not in visited:
            if len(vertex.children) == 0:
                temp = vertex
                points = []
                while temp is not None:
                    points = [temp.data] + points
                    temp = temp.parent
                ref_points.append(reference_point(count, points))
                count += 1
            stack.extend(vertex.children)
            visited.add(vertex)
    return ref_points


from math import factorial


def comb(n, r):
    return factorial(n) / factorial(r) / factorial(n - r)


def setDivs(number_of_objectives):
    """reference points have two level when number of objectives > 5 (to keep number of reference points under check"""
    if number_of_objectives == 3:
        number_of_reference_points = [12, 0]
    elif number_of_objectives == 5:
        number_of_reference_points = [6, 0]
    elif number_of_objectives == 8 or number_of_objectives == 10:
        number_of_reference_points = [3, 2]
    elif number_of_objectives == 15:
        number_of_reference_points = [2, 1]
    else:
        print "This case is not handled. Number of objectives: ", number_of_objectives
        exit()

    return number_of_reference_points


def generate_weight_vector(division, number_of_objectives):
    root = Node(-1)
    tree(root, number_of_objectives, division)
    return get_ref_points(root)


def two_level_weight_vector_generator(divisions, number_of_objectives):
    division1 = divisions[0]
    division2 = divisions[1]

    if division1 != 0: N1 = comb(number_of_objectives + division1 - 1, division1)
    if division2 != 0: N2 = comb(number_of_objectives + division2 - 1, division2)

    first_layer = []
    second_layer = []
    if N1 != 0:  first_layer = generate_weight_vector(division1, number_of_objectives)
    if N2 != 0:
        second_layer = generate_weight_vector(division2, number_of_objectives)
        mid = 1 / number_of_objectives
        for tsl_objectives in second_layer:
            tsl_objectives.id += int(N1)
            tsl_objectives.coordinates = [(t + mid) / 2 for t in tsl_objectives.coordinates]

    return first_layer + second_layer


def get_betaq(rand, alpha, eta=30):
    betaq = 0.0
    if rand <= (1.0 / alpha):
        betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
    else:
        betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
    return betaq


def sbxcrossover(problem, parent1, parent2, cr=1, eta=30):
    """
    Adapted from the code from Dr. Deb's NSGAII code [http://www.iitk.ac.in/kangal/codes.shtml]
    """

    assert (len(parent1.decisionValues) == len(parent2.decisionValues)), "Parents are sick"

    from copy import deepcopy
    child1 = [0 for _ in xrange(len(parent1.decisionValues))]
    child2 = [0 for _ in xrange(len(parent1.decisionValues))]

    if random() > cr: return parent1, parent2
    for index in xrange(len(parent1.decisionValues)):

        # import pdb
        # pdb.set_trace()

        # Should these variables be considered for crossover
        if random() > 0.5:
            child1[index] = parent1.decisionValues[index]
            child2[index] = parent2.decisionValues[index]
            continue

        # Are these variable the same
        EPS = 1.0e-14
        if abs(parent1.decisionValues[index] - parent2.decisionValues[index]) <= EPS:
            print "boom"
            child1[index] = parent1.decisionValues[index]
            child2[index] = parent2.decisionValues[index]
            continue

        lower_bound = problem.decisions[index].low
        upper_bound = problem.decisions[index].up

        y1 = min(parent1.decisionValues[index], parent2.decisionValues[index])
        y2 = max(parent1.decisionValues[index], parent2.decisionValues[index])
        random_no = random()

        # child 1
        beta = 1.0 + (2.0 * (y1 - lower_bound) / (y2 - y1))
        alpha = 2.0 - beta ** (-(eta + 1.0))
        betaq = get_betaq(random_no, alpha, eta)

        child1[index] = 0.5 * ((y1 + y2) - betaq * (y2 - y1))

        # child 2
        beta = 1.0 + (2.0 * (upper_bound - y2) / (y2 - y1))
        alpha = 2.0 - beta ** -(eta + 1.0)
        betaq = get_betaq(random_no, alpha, eta)

        child2[index] = 0.5 * ((y1 + y2) + betaq * (y2 - y1))

        child1[index] = max(lower_bound, min(child1[index], upper_bound))
        child2[index] = max(lower_bound, min(child2[index], upper_bound))

    return jmoo_individual(problem, child1), jmoo_individual(problem, child2)


def sbxmutation(problem, individual, eta=20):
    """
    Adapted from the code from Dr. Deb's NSGAII code [http://www.iitk.ac.in/kangal/codes.shtml]
    """
    mr = 1 / len(problem.decisions)

    parent = individual.decisionValues
    mutant = [0 for _ in individual.decisionValues]

    for index, p in enumerate(parent):
        if random() > mr:
            mutant[index] = parent[index]
            continue

        lower_bound = problem.decisions[index].low
        upper_bound = problem.decisions[index].up

        delta1 = (p - lower_bound) / (upper_bound - lower_bound)
        delta2 = (upper_bound - p) / (upper_bound - lower_bound)

        mutation_power = 1 / (eta + 1)

        random_no = random()

        if random_no < 0.5:
            xy = 1 - delta1
            val = 2.0 * random_no + (1.0 - 2.0 * random_no) * (xy ** (eta + 1.0))
            deltaq = val ** mutation_power - 1.0
        else:
            xy = 1 - delta2
            val = 2.0 * (1 - random_no) + 2.0 * (random_no - 0.5) * (xy ** (eta + 1.0))
            deltaq = 1.0 - val ** mutation_power

        mutant[index] = max(lower_bound, min((parent[index] + deltaq * (upper_bound - lower_bound)), upper_bound))
    individual.decisionValues = mutant

    return individual


def variation(problem, individual_index, population):
    """ SBX regeneration Technique """

    from random import randint
    another_parent = individual_index
    while another_parent == individual_index: another_parent = randint(0, len(population)-1)

    parent1 = population[individual_index]
    parent2 = population[another_parent]

    child1, _ = sbxcrossover(problem, parent1, parent2)

    mchild1 = sbxmutation(problem, child1)
    mchild1.evaluate()
    return mchild1


def compute_ideal_points(problem, population):
    ideal_point = [1e32 for _ in xrange(len(problem.objectives))]
    for i in xrange(len(problem.objectives)):
        for individual in population:
            if individual.fitness.fitness[i] < ideal_point[i]:
                ideal_point[i] = individual.fitness.fitness[i]
    return ideal_point


def compute_max_points(problem, population):
    max_points = [-1e32 for _ in xrange(len(problem.objectives))]
    for i in xrange(len(problem.objectives)):
        for individual in population:
            if individual.fitness.fitness[i] > max_points[i]:
                max_points[i] = individual.fitness.fitness[i]
    return max_points


def asf_function(individual, index, ideal_point):
    max_value = -1e32
    epsilon = 1e-6
    for i, obj in enumerate(individual.fitness.fitness):
        temp_value = abs(obj - ideal_point[i])
        if index != i: temp_value /= epsilon
        if temp_value > max_value: max_value = temp_value
    return max_value

def compute_extreme_points(problem, population, ideal_point):
    extreme_points = []
    for i in xrange(len(problem.objectives)):
        index = -1
        min_value = 1e32
        for individual in population:
            assert(len(individual.fitness.fitness) == len(problem.objectives)), "somethings wrong"
            asf_value = asf_function(individual, i, ideal_point)
            if asf_value < min_value: index = i
        extreme_points.append(population[index].fitness.fitness)

    assert(len(extreme_points) == len(problem.objectives)), "Number of extreme points should be equal to number of objectives"
    return extreme_points

def computer_intercept_points(problem, extreme_points, ideal_point, max_point):
    import numpy
    assert(len(extreme_points) == len(problem.objectives)), "Length of extreme points should be equal to the number of objectives of the problem"
    assert(len(extreme_points[0]) == len(ideal_point)), "Length of extreme points and ideal points should be the same"
    temp_L = []
    for extreme_point in extreme_points:
        temp = []
        for i, j in zip(extreme_point, ideal_point): temp.append(i-j)
        temp_L.append(temp)

    EX = numpy.array(temp_L)
    intercepts = [-1 for _ in problem.objectives]
    if numpy.linalg.matrix_rank(EX) == len(EX):
        UM = numpy.matrix([[1] for _ in problem.objectives])
        AL0 = numpy.linalg.inv(EX)
        AL = AL0 * UM
        for i, ideal_co in enumerate(ideal_point):
            try:
                temp_aj = 1/AL[i] + ideal_co
            except ZeroDivisionError:
                break
            if temp_aj > ideal_co:
                intercepts[i] = temp_aj
            else: break
        if i != len(problem.objectives):
            for k, max_v in enumerate(max_point):
                intercepts[k] = max_v

    else:
        for k,max_value in enumerate(max_point):
            intercepts[k] = max_value # zmax
    return intercepts



def normalization(problem, population, intercept_point, ideal_point):
alkdjaslkdjsalkdj
def convert_to_jmoo(problem, pareto_fronts):
    population = []
    for front_no, front in enumerate(pareto_fronts):
        for i, dIndividual in enumerate(front):
            cells = []
            for j in xrange(len(dIndividual)):
                cells.append(dIndividual[j])
            population.append(jmoo_individual(problem, cells, dIndividual.fitness.values))

    from itertools import chain
    assert(len(list(chain(*pareto_fronts))) <= len(population)), "Non Dominated Sorting is wrong!"

    return population


# ---------------------- Helper --------------------------------




def nsgaiii_selector2(problem, population):
    assert (len(population) % 4 == 0), "The population size needs to be multiple if 4. Look at footnote page 584"
    return population, 0


def nsgaiii_regenerate2(problem, population):
    assert (len(population) == jmoo_properties.MU), "The population should be equal to the population size as defined in jmoo_properties"
    assert (len(population) % 4 == 0), "The population size needs to be multiple if 4. Look at footnote page 584"

    mutants = []
    for count, individual in enumerate(population): mutants.append(variation(problem, count, population))
    assert (len(mutants) == len(population)), "The population after regeneration must be double"
    return mutants, len(mutants)


def nsgaiii_recombine2(problem, population, selectees, k):
    assert (len(population) % 4 == 0), "The population size needs to be multiple if 4. Look at footnote page 584"
    assert (len(population + selectees) == 2 * len(population)), "The recombination population should be 2 * len(population)"
    evaluate_no = 0
    for individual in population+selectees:
            if not individual.valid:
                individual.evaluate()
                evaluate_no += 1
    Individuals = jmoo_algorithms.deap_format(problem, population + selectees)
    pareto_fronts = sortNondominated(Individuals, k)

    minus_last_front_population = convert_to_jmoo(problem, pareto_fronts[:-1])
    last_front = convert_to_jmoo(problem, pareto_fronts[-1:])
    population = minus_last_front_population + last_front

    from itertools import chain
    assert(len(minus_last_front_population) + len(last_front) == len(list(chain(*pareto_fronts)))), "Length of the population and mgpopulation should be equal to pareto_fronts"

    ideal_point = compute_ideal_points(problem, population)
    max_point = compute_max_points(problem, population)
    extreme_points = compute_extreme_points(problem, population, ideal_point)
    intercept_point = computer_intercept_points(problem, extreme_points, ideal_point, max_point)
    population = normalization(problem, population, intercept_point, ideal_point)
    print
