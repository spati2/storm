from __future__ import division
import os, sys, inspect
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe()))[0],"../..")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
print cmd_subfolder
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
            print dir(individual)
            raw_input()
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
    # return max([float(o/w) for o, w in zip(individual.fitness.fitness, weights)])
    return max([float(o/w) for o, w in zip(individual.translated, weights)])


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
    2. Translate the objectives by subtracting the min value from the objective function
    """
    for i in xrange(len(problem.objectives)):
        z_j_min = min([individual.fitness.fitness[i] for individual in population if individual.front_no == 0])
        ideal_points.append(z_j_min)
        for index, individual in enumerate(population):
            individual.translated.append(individual.fitness.fitness[i] - z_j_min)


    for i in xrange(len(problem.objectives)):
        extreme_points.append(compute_extreme_points(problem, population, i))

    # print "Extreme Points: ", len(extreme_points) != len(deduplicate(extreme_points))
    # for extreme_point in extreme_points:
    #     print extreme_point.translated
    # assert(len(extreme_points) != len(deduplicate(extreme_points))), "Extreme point bugs"


    if len(extreme_points) != len(deduplicate(extreme_points)):
        # print "Duplicate exists",
        # print "-" * 20 + ">"
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

def easy_normalize(problem, population, Z_r, Z_s, Z_a):
    """
        Normalization technique adapted from "An improved NSGA-III procedure for evolutionary many objective optimization"
    """
    z_min_points = []  # Ideal points: which means minimum values of all objectives
    z_max_points = []  # Inverse Ideal points: which means maximum values of all objectives
    for i in xrange(len(problem.objectives)):
        temp_objectives = sorted([individual.fitness.fitness[i] for individual in population])
        z_min_points.append(temp_objectives[0])
        z_max_points.append(temp_objectives[-1])
    assert(len(z_min_points) == len(z_min_points)), "The length of the z_min_points and z_max_points must be same"
    assert(len(z_min_points) == len(problem.objectives)), "The length of the z_min_points and number of objectives must be same"

    for individual in population:
        temp_normalized_values = [0 for _ in xrange(len(problem.objectives))]
        for i in xrange(len(problem.objectives)):
            temp_normalized_values[i] = (individual.fitness.fitness[i] - z_min_points[i])/(z_max_points[i] - z_min_points[i])
        individual.normalized = temp_normalized_values

    return population


def test():
    import numpy
    EX = numpy.matrix([[15.707736310987924,211.2763436713055,265.5626953528718],[151.18874467099698,800.8106458512383,74.47433535814318],[9.030300437360912,0.0,1057.5671626332546]])
    zideal = [4.5699810820927, 0.0, 57.8939921967703]
    print "rank: ", numpy.linalg.matrix_rank(EX)
    print "row: ", len(EX)
    intercepts = [-1 for _ in zideal]
    if numpy.linalg.matrix_rank(EX) == len(EX):
        UM = numpy.matrix([[1] for _ in zideal])
        print UM
        AL0 = numpy.linalg.inv(EX)
        AL = AL0 * UM

        print UM
        print AL


        for i in xrange(len(zideal)):
            try:
                temp_aj = 1/AL[i] + zideal[i]
            except ZeroDivisionError:
                break
            if temp_aj > zideal[i]:
                for k in xrange(len(zideal)):
                    intercepts[k] = 100 # zmax
    else:
        for k in xrange(len(zideal)):
            intercepts[k] = 100 # zmax

    print intercepts



if __name__ == "__main__":
    # A = [[0 for _ in xrange(4)] for _ in xrange(3)]
    # print A
    # for i, x in enumerate([626.005840126248, 17.65177402253587, 0.0]):
    #     A[0][i] = x
    # for i, x in enumerate([55.385028410583004, 1056.186303469722, 111.34874228099781]):
    #     A[1][i] = x
    # for i, x in enumerate([0.0, 10.683255094756921, 1226.4509105230284]):
    #     A[2][i] = x
    # A[0][3] = 1
    # A[1][3] = 1
    # A[2][3] = 1
    # gauss_elimination(A)

    test()
