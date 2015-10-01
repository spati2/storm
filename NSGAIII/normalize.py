from __future__ import division
import os, sys, inspect
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe()))[0],"..")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
from jmoo_individual import *
from ref_point import cover


def find_ideal_points(problem, objectives):
    # print "U Length of the population: ", len(objectives)
    assert(len(problem.objectives) == len(objectives[0])), "Length of the objectives is not correct"
    utopia = [1e32 if x.lismore is True else -1e32 for x in problem.objectives]
    for individual in objectives:
        for i, obj in enumerate(individual):
           if problem.objectives[i].lismore is True:
               if obj < utopia[i]:
                   utopia[i] = obj
           else:
               if obj > utopia[i]:
                   utopia[i] = obj
    return utopia

def translate_objectives(problem, population, utopia):
    assert(len(problem.objectives) == len(population[0].fitness.fitness)), "Length of the objectives is not correct"
    # print "Length of the population: ", len(population)
    for individual in population:
        obj = individual.fitness.fitness
        temp = []
        for i in xrange(len(obj)):
            t = obj[i] - utopia[i]
            assert(t >= 0), "t should be greater or equal to 0"
            temp.append(t)
        individual.translated = temp
    return population

def get_extreme_points(problem, population):
    """
    :param problem:
    :param objectives: This is the translated objectives
    :return: intercepts of ith objective axis
    """
    def compute_asf(objectives, weights):
        temp = -1e32
        for o,w in zip(objectives, weights):
            t = o/w
            if t > temp:
                temp = t
        return temp

    points = []
    for i, obj in enumerate(problem.objectives):
        weight = [1e-6 for _ in problem.objectives]
        weight[i] = 1
        asf = 1e32
        extreme = None
        for individual in population:
            t = compute_asf(individual.translated, weight)
            if t < asf:
                asf = t
                extreme = individual.fitness.fitness
        points.append(extreme)
    return points
    #return [point[i] for i, point in enumerate(points)]


def maxpoints(problem, population):
    # print "MX Length of the population: ", len(population)
    assert(len(population) != 1), "Length of population can't be 1"
    maxp = []
    for o in xrange(len(problem.objectives)):
        maxp.append(sorted([individual.fitness.fitness[o] for individual in population], reverse=True)[0])
        # for i, obj in enumerate(individual.translated):
        #     if problem.objectives[i].lismore is True:
        #         if obj > maxp[i]:
        #             maxp[i] = obj
        #     else:
        #         if obj < maxp[i]:
        #             maxp[i] = obj
    # print "maxp: ", maxp
    return maxp

def final_normalize(intercepts, utopia, population):
    for individual in population:
        temp = []
        for no, (i, u) in enumerate(zip(intercepts, utopia)):
            # if i == u: print "oo o"
            if abs(i - u) > 1e-10:
                temp.append(individual.translated[no] / float(i - u))
            else:
                temp.append(individual.translated[no] / 1e-10)   # hacks from Dr Chiang, avoid div 0
        individual.normalized = temp
    return population

def deduplicate(lis):
    new_k = []
    for elem in lis:
        if elem not in new_k:
            new_k.append(elem)
    return new_k

def gauss_elimination(A):
    """
    http://martin-thoma.com/solving-linear-equations-with-gaussian-elimination/#tocAnchor-1-3
    """
    n = len(A)
    for i in range(0, n):
        # Search for maximum in this column
        maxEl = abs(A[i][i])
        maxRow = i
        for k in range(i+1, n):
            if abs(A[k][i]) > maxEl:
                maxEl = abs(A[k][i])
                maxRow = k

        # Swap maximum row with current row (column by column)
        for k in range(i, n+1):
            tmp = A[maxRow][k]
            A[maxRow][k] = A[i][k]
            A[i][k] = tmp

        # Make all rows below this one 0 in current column
        for k in range(i+1, n):
            c = -A[k][i]/(A[i][i] + 1e-6)
            for j in range(i, n+1):
                if i == j:
                    A[k][j] = 0
                else:
                    A[k][j] += c * A[i][j]

    # Solve equation Ax=b for an upper triangular matrix A
    x = [0 for i in range(n)]
    for i in range(n-1, -1, -1):
        x[i] = A[i][n]/(A[i][i] + 1e-6)
        for k in range(i-1, -1, -1):
            A[k][n] -= A[k][i] * x[i]
    # for o in x:
    #     try:
    #         assert( o != 0)," something's wrong"
    #     except:
    #         print x
    #         print A
    #         assert( o != 0)," something's wrong"
    return x


def compute_asf(individual, weights):
    assert(len(individual.translated) == len(weights)), "Something's wrong"
    return max([float(o/w) for o, w in zip(individual.fitness.fitness, weights)])

def compute_extreme_points(problem, S_t, objective_number):
    # construct weights
    weight = [1e-6 for _ in xrange(len(problem.objectives))]
    weight[objective_number] = 1
    assert(len(problem.objectives) == len(weight)), "There is a length mismatch"

    #compute ASF
    return sorted([(compute_asf(individual, weight), individual) for individual in S_t if individual.front_no == 0], key=lambda x: x[0])[0][1]



def normalize(problem, population, Z_r, Z_s, Z_a):

    extreme_points = []
    ideal_points = []

    # adding new field called translated to all the members of the population
    for pop in population:
        pop.translated = []
        pop.normalized = []


    """
    1. Find the ideal point
    2. Translate the objectives by substracting the min value from the objective function
    """
    for i in xrange(len(problem.objectives)):
        z_j_min = min([individual.fitness.fitness[i] for individual in population if individual.front_no == 0])
        ideal_points.append(z_j_min)
        for index, individual in enumerate(population):
            individual.translated.append(individual.fitness.fitness[i] - z_j_min)


    for i in xrange(len(problem.objectives)):
        extreme_points.append(compute_extreme_points(problem, population, i))

    if len(extreme_points) != len(deduplicate(extreme_points)):
        # print "Duplicate exists",
        print "-" * 20 + ">"
        a = [0 for _ in problem.objectives]
        # for i, obj in enumerate(problem.objectives):
        #     a[i] = extreme_points[i].fitness.fitness[i]  # Changed using Dr. Chiang's code
        a = maxpoints(problem, population)
        # print a
        # exit()
    else:
        # print "-" * 20 + ">"
        # Calculate the intercepts (Gaussian elimination)
        from fractions import Fraction
        n = len(extreme_points)
        A = [[0 for j in range(n+1)] for i in range(n)]
        for i in xrange(n):
            for j in xrange(n): A[i][j] = Fraction(extreme_points[i].fitness.fitness[j])
            A[i][n] = Fraction(1/1)
        a = [float(1/(aa)) for aa in gauss_elimination(A)]
        # for e in extreme_points:
        #     print e.fitness.fitness
        # import time
        # time.sleep(1)
    population = final_normalize(a, ideal_points, population)


    return population

if __name__ == "__main__":
    A = [[0 for _ in xrange(4)] for _ in xrange(3)]
    print A
    for i, x in enumerate([-1, 1, 2]):
        A[0][i] = x
    for i, x in enumerate([2, 0, -3]):
        A[1][i] = x
    for i, x in enumerate([5, 1, -2]):
        A[2][i] = x
    A[0][3] = 1
    A[1][3] = 1
    A[2][3] = 1
    gauss_elimination(A)
