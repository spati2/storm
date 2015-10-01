from __future__ import division
import os, inspect, sys, random

cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe()))[0],"../../Techniques")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

from euclidean_distance import euclidean_distance

def scores(individual, stars):
    cols = len(stars[0].east.fitness.fitness)
    temp = -1e32
    selected = -1
    individual.anyscore = -1e30
    for count, star in enumerate(stars):
        try:
            a = euclidean_distance(individual.decisionValues, star.east.decisionValues)
            b = euclidean_distance(individual.decisionValues, star.west.decisionValues)
            c = euclidean_distance(star.east.decisionValues, star.west.decisionValues)
            x = (a**2 + c**2 - b**2) / (2*c)
            y = (a**2 - x**2)**0.5
            r = len(stars) - 1  # Number of poles - midpoint

            diff = euclidean_distance(star.east.fitness.fitness, star.west.fitness.fitness)
            temp_score = ((a/b) * diff/(y**2))
        except:
            temp_score = 1e32

        if temp < temp_score:
            temp = temp_score
            selected = count

    assert(selected > -1), "Something's wrong"
    individual.anyscore = temp
    return selected, individual


def scores2(individual, stars):
    cols = len(stars[0].east.fitness.fitness)
    temp = -1e32
    selected = -1
    individual.anyscore = -1e30
    nearest_contours = []
    for count, star in enumerate(stars):
        try:
            a = euclidean_distance(individual.decisionValues, star.east.decisionValues)
            b = euclidean_distance(individual.decisionValues, star.west.decisionValues)
            c = euclidean_distance(star.east.decisionValues, star.west.decisionValues)
            x = (a**2 + c**2 - b**2) / (2*c)
            y = (a**2 - x**2)**0.5
            r = len(stars) - 1  # Number of poles - midpoint

            diff = euclidean_distance(star.east.fitness.fitness, star.west.fitness.fitness)
            temp_score = ((a/b) * diff/(y**2))
        except:
            temp_score = 1e32

        nearest_contours.append([count, temp_score])

    return [x[0] for x in sorted(nearest_contours, key=lambda item:item[-1], reverse=True)[:6]], individual

