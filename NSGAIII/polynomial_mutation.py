from __future__ import division
import random


def pmutation(problem, individual, eta=20):
    mr = 1/len(problem.decisions)

    parent = individual.decisionValues
    mutant = [0 for _ in individual.decisionValues]

    for index, p in enumerate(parent):
        if random.random() > mr:
            mutant[index] = parent[index]
            continue

        lower_bound = problem.decisions[index].low
        upper_bound = problem.decisions[index].up

        delta1 = (p - lower_bound)/(upper_bound - lower_bound)
        delta2 = (upper_bound - p)/(upper_bound - lower_bound)

        mutation_power = 1/(eta+1)

        random_no = random.random()

        if random_no < 0.5:
            xy = 1 - delta1
            val = 2.0 * random_no + (1.0 - 2.0 * random_no) * (xy ** (eta+1.0))
            deltaq = val ** mutation_power - 1.0
        else:
            xy = 1 - delta2
            val = 2.0 * (1 - random_no) + 2.0 * (random_no - 0.5) * (xy ** (eta+1.0))
            deltaq = 1.0 - val ** mutation_power

        mutant[index] = max(lower_bound, min((parent[index] + deltaq * (upper_bound - lower_bound)), upper_bound))
    individual.decisionValues = mutant

    return individual

