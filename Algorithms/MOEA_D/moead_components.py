
from __future__ import division
import random
# import jmoo_properties
from jmoo_individual import *
distance_matrix = []
ideal_points = []
reference_points = []
indivpoint = []

from Techniques.euclidean_distance import euclidean_distance
from Techniques.math_functions import combination


division_dict = {"3": [12,0],
                 "5": [6,0],
                 "8": [3,2],
                 "10": [3, 2],
                 "15": [2, 1]}


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

def generate_weight_vector(division, number_of_objectives):
    root = Node(-1)
    tree(root, number_of_objectives, division)
    return get_ref_points(root)


def two_level_weight_vector_generator(divisions, number_of_objectives):
    division1 = divisions[0]
    division2 = divisions[1]

    N1 = 0
    N2 = 0

    if division1 != 0: N1 = combination(number_of_objectives + division1 - 1, division1)
    if division2 != 0: N2 = combination(number_of_objectives + division2 - 1, division2)

    first_layer = []
    second_layer = []
    if N1 != 0:  first_layer = generate_weight_vector(division1, number_of_objectives)
    if N2 != 0:
        second_layer = generate_weight_vector(division2, number_of_objectives)
        mid = 1 / number_of_objectives
        for tsl_objectives in second_layer:
            tsl_objectives.id += int(N1)
            tsl_objectives.coordinates = [(t + mid) / 2 for t in tsl_objectives.coordinates]

    import numpy
    for l in first_layer + second_layer:
        l.coordinates = [ll.item() for ll in l.coordinates]
        for ll in l.coordinates:
            assert(type(ll) != numpy.float64), "Seomthing is wrong"
    return first_layer + second_layer


def create_reference_points(problem):
    global reference_points, division_dict
    number_of_objectives = len(problem.objectives)
    reference_points = two_level_weight_vector_generator(division_dict[str(number_of_objectives)], number_of_objectives)


def assign_weights():
    global reference_points
    from random import shuffle
    shuffle(reference_points)
    return_point = reference_points[0]
    reference_points.pop(0)
    return return_point.coordinates


def create_distance_matrix(population):
    global distance_matrix
    weights = [pop.weight for pop in population]
    distance_matrix = [[[0, i] for i, _ in enumerate(xrange(len(weights)))] for _ in xrange(len(weights))]
    for i in xrange(len(weights)):
        for j in xrange(len(weights)):
            distance_matrix[i][j][0] = euclidean_distance(weights[i], weights[j])
        assert(distance_matrix[i][i][0] == 0), "Diagonal of Distance matrix should be 0"


def create_ideal_points(problem):
    global ideal_points, indivpoint
    ideal_points = [1e30 if obj.lismore else 1e-30 for obj in problem.objectives]
    indivpoint = [None for _ in problem.objectives]


def update_ideal_points(problem, individual):
    global ideal_points, indivpoint
    from copy import deepcopy
    obj_in = deepcopy(individual.fitness.fitness)
    assert(len(ideal_points) == len(obj_in)), "Length of ideal points are not equal to length of objectives"
    for i, obj in enumerate(problem.objectives):
        if obj.lismore:
            if ideal_points[i] > obj_in[i]:
                ideal_points[i] = obj_in[i]
                indivpoint[i] = deepcopy(individual)
        else:
            if ideal_points[i] < obj_in[i]:
                ideal_points[i] = obj_in[i]
                indivpoint[i] = deepcopy(individual)


def find_neighbours(pop_id, configuration):
    temp = [x[1] for x in sorted(distance_matrix[pop_id], key=lambda l: l[0])[1:configuration["MOEAD"]["T"]+1]]
    assert(len(set(temp)) == len(temp)), "There should be no repetition"
    return temp


def trim(mutated, low, up):
    assert(low < up), "There is a mix up between low and up"
    if random.random > 0.5:
        return max(low, min(mutated, up))
    else:
        return low + ((mutated - low) % (up - low))


def assign_id_to_member(population):
    for i, pop in enumerate(population):
        pop.id = i
    return population


def mutate(problem, individual, configuration):
    from numpy.random import random
    eta_m_ = configuration["MOEAD"]["ETA_M_DEFAULT_"]
    distributionIndex_ = eta_m_
    output = jmoo_individual(problem, individual.decisionValues)

    probability = 1/len(problem.decisions)
    for var in xrange(len(problem.decisions)):
        if random() <= probability:
            y = individual.decisionValues[var]
            yU = problem.decisions[var].up
            yL = problem.decisions[var].low
            delta1 = (y - yL)/(yU - yL)
            delta2 = (yU - y)/(yU - yL)
            rnd = random()

            mut_pow = 1.0/(eta_m_ + 1.0)
            if rnd < 0.5:
                xy = 1.0 - delta1
                val = 2.0 * rnd + (1 - 2 * rnd) * (xy ** (distributionIndex_ + 1.0))
                deltaq = val ** mut_pow - 1
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0-rnd) + 2.0 * (rnd-0.5) * (xy ** (distributionIndex_+1.0))
                deltaq = 1.0 - (val ** mut_pow)

            y += deltaq * (yU - yL)
            if y < yL: y = yL
            if y > yU: y = yU

            output.decisionValues[var] = y

    return output


def get_betaq(rand, alpha, eta=30):
    betaq = 0.0
    if rand <= (1.0 / alpha):
        betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
    else:
        betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
    return betaq


def polynomial_mutation(problem, individual, configuration):
    from numpy.random import random
    eta_m_ = configuration["MOEAD"]["ETA_M_DEFAULT_"]
    distributionIndex_ = eta_m_
    output = jmoo_individual(problem, individual.decisionValues)

    probability = 1/len(problem.decisions)
    for var in xrange(len(problem.decisions)):
        if random() <= probability:
            y = individual.decisionValues[var]
            yU = problem.decisions[var].up
            yL = problem.decisions[var].low
            delta1 = (y - yL)/(yU - yL)
            delta2 = (yU - y)/(yU - yL)
            rnd = random()

            mut_pow = 1.0/(eta_m_ + 1.0)
            if rnd < 0.5:
                xy = 1.0 - delta1
                val = 2.0 * rnd + (1 - 2 * rnd) * (xy ** (distributionIndex_ + 1.0))
                deltaq = val ** mut_pow - 1
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0-rnd) + 2.0 * (rnd-0.5) * (xy ** (distributionIndex_+1.0))
                deltaq = 1.0 - (val ** mut_pow)


            y +=  deltaq * (yU - yL)
            if y < yL: y = yL
            if y > yU: y = yU

            output.decisionValues[var] = y

    return output


def sbxcrossover(problem, parent1, parent2, configuration):

    EPS = 1.0e-14
    distribution_index = configuration["MOEAD"]["ETA_C_DEFAULT_"]
    probability = configuration["MOEAD"]["SBX_Probability"]
    from numpy.random import random
    offspring1 = jmoo_individual(problem, parent1.decisionValues)
    offspring2 = jmoo_individual(problem, parent2.decisionValues)

    number_of_variables = len(problem.decisions)
    if random() <= probability:
        for i in xrange(number_of_variables):
            valuex1 = offspring1.decisionValues[i]
            valuex2 = offspring2.decisionValues[i]
            if random() <= 0.5:
                if abs(valuex1 - valuex2) > EPS:
                    if valuex1 < valuex2:
                        y1 = valuex1
                        y2 = valuex2
                    else:
                        y1 = valuex2
                        y2 = valuex1

                    yL = problem.decisions[i].low
                    yU = problem.decisions[i].up
                    rand = random()
                    beta = 1.0 + (2.0 * (y1 - yL) / (y2 - y1))
                    alpha = 2.0 - beta ** (-1 * (distribution_index + 1.0))

                    if rand <= 1/alpha:
                        betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (distribution_index + 1.0))
                    else:
                        betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (distribution_index + 1.0))

                    c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
                    beta = 1.0 + (2.0 * (yU - y2) / (y2 - y1))
                    alpha = 2.0 - beta ** -(distribution_index + 1.0)

                    if rand <= (1.0 / alpha):
                        betaq = (rand * alpha) ** (1.0 / (distribution_index + 1.0))
                    else:
                        betaq = ((1.0 / (2.0 - rand * alpha)) ** (1.0 / (distribution_index + 1.0)))

                    c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))

                    if c1 < yL: c1 = yL
                    if c2 < yL: c2 = yL
                    if c1 > yU: c1 = yU
                    if c2 > yU: c2 = yU

                    if random() <= 0.5:
                        offspring1.decisionValues[i] = c2
                        offspring2.decisionValues[i] = c1
                    else:
                        offspring1.decisionValues[i] = c1
                        offspring2.decisionValues[i] = c2
                else:
                    offspring1.decisionValues[i] = valuex1
                    offspring2.decisionValues[i] = valuex2
            else:
                offspring1.decisionValues[i] = valuex2
                offspring2.decisionValues[i] = valuex1

    return offspring1, offspring2


def variation(problem, individual_index, population, configuration):
    """ SBX regeneration Technique """

    from random import randint
    from copy import deepcopy
    individual = deepcopy([pop for pop in population if pop.id == individual_index][-1])
    other_parent = individual.neighbor[randint(0, configuration["MOEAD"]["T"]-1)]
    another_parent = other_parent
    while another_parent == other_parent: another_parent = individual.neighbor[randint(0, configuration["MOEAD"]["T"]-1)]

    assert(len([pop for pop in population if pop.id == other_parent]) == 1), "id should be unique"
    print len([pop for pop in population if pop.id == another_parent])
    print type(another_parent), another_parent
    assert(len([pop for pop in population if pop.id == another_parent]) == 1), "id should be unique"

    from copy import deepcopy
    parent1 = deepcopy([pop for pop in population if pop.id == other_parent][-1])
    parent2 = deepcopy([pop for pop in population if pop.id == another_parent][-1])

    child1, _ = sbxcrossover(problem, parent1, parent2, configuration)
    mchild1 = polynomial_mutation(problem, child1, configuration)
    mchild1.evaluate()  #checked| correct

    assert(mchild1.valid is True), "Something is wrong| Check if the evaluation is complete"
    return mchild1


def weighted_tche(problem, individual_fitness, weight_vector, local_indivpoint):
    global ideal_points
    assert(len(weight_vector) == len(individual_fitness)), "Number of weights should be equal to objectives"
    scale = []
    for i in xrange(len(problem.objectives)):
        min_value = 1e30
        max_value = -1e30
        for j in xrange(len(problem.objectives)):
            tp = local_indivpoint[j].fitness.fitness[j]
            if tp > max_value: max_value = tp
            if tp < min_value: min_value = tp
        scale.append(max_value - min_value)
        if max_value == min_value: return 1e30

    max_fun = -1e30
    for n in xrange(len(problem.objectives)):
        diff = (individual_fitness[n] - ideal_points[n])/scale[n]
        if weight_vector[n] == 0:
            feval = 0.0001 * diff
        else:
            feval = diff * weight_vector[n]
        if feval > max_fun: max_fun = feval
    return feval



def update_neighbor(problem, individual, mutant, population, dist_function, configuration):
    global indivpoint
    from copy import deepcopy
    new_population = deepcopy(population)

    for i in xrange(configuration["MOEAD"]["T"]):

        k = individual.neighbor[i]
        f1 = dist_function(problem, individual.fitness.fitness, individual.weight, indivpoint)
        f2 = dist_function(problem, mutant.fitness.fitness, individual.weight, indivpoint)
        if f2 < f1:
            print "bppm"
            something is wrong here
            for pop in new_population:
                if pop.id == k:  new_population.remove(pop)
            new_candidate = jmoo_individual(problem, mutant.decisionValues, mutant.fitness.fitness)
            new_candidate.id = i
            new_candidate.neighbor = deepcopy(pop.neighbor)
            new_candidate.weight = deepcopy(pop.weight)
            new_population.append(new_candidate)

    ids = [pop.id for pop in new_population]
    assert(len(set(ids)) == len(ids)), "Something is wrong"
    return new_population


    # assert(len([pop for pop in population if pop.id == individual.id]) == 1), "Id should be unique"
    # for pop in new_population:
    #     if pop.id == individual.id:
    #         popi = deepcopy(pop)
    #         new_population.remove(pop)
    #
    # assert(len(new_population) + 1 == len(population)), "Just removed an element"
    #
    # new_candidate = jmoo_individual(problem, mutant.decisionValues, mutant.fitness.fitness)
    # new_candidate.id = deepcopy(popi.id)
    # new_candidate.neighbor = deepcopy(popi.neighbor)
    # new_candidate.weight = deepcopy(popi.weight)
    #
    # new_population.append(new_candidate)
    #
    # for i in individual.neighbor:
    #     neigh = [pop for pop in new_population if pop.id == i]
    #     # print pop.id, len(neigh), len(population)
    #     assert(len(neigh) == 1), "Something is wrong"
    #     neigh = neigh[-1]
    #     d = dist_function(problem, individual.weight, neigh)
    #     e = dist_function(problem, individual.weight, mutant)
    #     if d < e:
    #         for pop in new_population:
    #             if pop.id == i:  new_population.remove(pop)
    #         new_candidate = jmoo_individual(problem, mutant.decisionValues, mutant.fitness.fitness)
    #         new_candidate.id = i
    #         new_candidate.neighbor = deepcopy(pop.neighbor)
    #         new_candidate.weight = deepcopy(pop.weight)
    #         new_population.append(new_candidate)
    #
    # assert(len(new_population) == len(population)), "Length of new population should be equal to population"
    # return new_population



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

    return jmoo_individual(problem, [float(d) for d in solution], None)


def evolve_neighbor(problem, individual_index, population, configuration):
    mutant = variation(problem, individual_index, population, configuration)
    mutant.evaluate()
    return mutant


def initialize_moead(problem, population, configuration):
    global reference_points
    for individual in population:
        if not individual.valid:
            individual.evaluate()
    population = assign_id_to_member(population)
    create_reference_points(problem)
    for pop in population:
        pop.weight = assign_weights()
        assert(int(round(sum(pop.weight), 0)) == 1), "The weights should add up to unity"
    create_ideal_points(problem)

    create_distance_matrix(population)
    for i, pop in enumerate(population): pop.neighbor = find_neighbours(i, configuration)
    for pop in population: update_ideal_points(problem, pop)
    return population, len(population)
# remeber to add len(population) to number of evals


def moead_selector(problem, population, configuration):
    print "#",
    from copy import deepcopy
    from random import shuffle
    new_population = deepcopy(population)
    indexes = [i for i in xrange(len(new_population))]
    shuffle(indexes)
    for no in indexes:
        pop = evolve_neighbor(problem, no, new_population, configuration)
        update_ideal_points(problem, pop)
        new_population = update_neighbor(problem, new_population[no], pop, new_population, weighted_tche, configuration)


    return new_population, len(population)


def moead_mutate(problem, population, configuration):
    return population, 0


def moead_recombine(problem, unusedSlot, mutants, configuration):
    return mutants, 0





# ------------------------------------------------ Tests --------------------------------------------------- #
def _find_weights():
    def one_test():
        summa = random.randint(1,100)
        number = random.randint(1,summa)
        ret = assign_weights(number, summa)
        print ret
        exit()
        return ret

    def test_list(lis):
        temp = [1 if num < 1/100 else 0 for num in lis]
        if sum(temp) > 0:
            return True
        elif sum(temp) == 0:
            return False
        else:
            raise Exception("Shouldn't happen")

    for _ in xrange(100):
        assert(test_list(one_test()) is False), "Something's wrong"


def _euclidean_distance():
    def one_test():
        dimension = random.randint(1, 100)
        first = [random.randint(-100, 100) for _ in xrange(dimension)]
        second = [random.randint(-100, 100) for _ in xrange(dimension)]
        from scipy.spatial import distance
        try:
            assert(round(euclidean_distance(first, second), 6) == round(distance.euclidean(first, second),6)), "Mistakes" \
                                                                                                               " discovered"
        except:
            print "First: ", first
            print "Second: ", second
            print "My: ", euclidean_distance(first, second)
            print "Their: ", distance.euclidean(first, second)
            return False
        return True

    for _ in xrange(10000):
        print ".",
        assert(one_test() is True), "Test Failed"





if __name__ == "__main__":
    _find_weights()
