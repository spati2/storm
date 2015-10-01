
"""
##########################################################
### @Author Joe Krall      ###############################
### @copyright see below   ###############################

    This file is part of JMOO,
    Copyright Joe Krall, 2014.

    JMOO is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    JMOO is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with JMOO.  If not, see <http://www.gnu.org/licenses/>.
    
###                        ###############################
##########################################################
"""

"Brief notes"
"Report tool for keeping track of stats during MOEAs"

import math
import jmoo_algorithms
from jmoo_individual import *
import jmoo_properties
from utility import *
from deap.tools.support import ParetoFront
from jmoo_preprocessor import IGDMEASURE

import os, inspect, sys
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe()))[0], "Techniques")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import IGD_Calculation

def readpf(problem):
    # print problem.name.split("_")[0]
    filename = "./PF/" + problem.name.split("_")[0] + "(" + str(len(problem.objectives)) + ")-PF.txt"
    f = open(filename, "r")
    true_PF = []
    for line in f:
        temp = []
        for x in line.split():
            temp.append(float(x))
        true_PF.append(temp)
    return true_PF


class jmoo_stats:
    "A single stat box - a simple record"
    def __init__(stats, population, fitnesses, fitnessMedians, fitnessSpreads, numEval, gen, IBD, IBS, changes):
        stats.population = population
        stats.fitnesses = fitnesses
        stats.fitnessMedians = fitnessMedians
        stats.fitnessSpreads = fitnessSpreads
        stats.numEval = numEval
        stats.gen = gen
        stats.IBD = IBD
        stats.IBS = IBS
        stats.changes = changes
        
     
          
class jmoo_stats_box:
    "Management of one stat box per generation"
    def __init__(statBox, problem, alg, foam=None):
        statBox.problem = problem
        statBox.numEval = 0
        statBox.box = []
        statBox.alg = alg
        statBox.foam = [{} for o in problem.objectives]
        statBox.bests = [100.0 for o in problem.objectives]
        statBox.bests_actuals = [0 for o in problem.objectives]
        statBox.lives = 3
    
    def update(statBox, population, gen, numNewEvals, initial = False, printOption=True):
        "add a stat box - compute the statistics first"
        filename = "data/results_"+statBox.problem.name + "-p" + str(len(population)) + "-d" + \
                   str(len(statBox.problem.decisions)) + "-o" + str(len(statBox.problem.objectives))+\
                   "_"+statBox.alg.name+".datatable"
        fa = open(filename, 'a')
        
        # Calculate percentage of violations
        violationsPercent = sum([ 1 for pop in population if statBox.problem.evalConstraints(pop.decisionValues)])/float(len(population))
        
        # Update Number of Evaluations
        statBox.numEval += numNewEvals
        #front = population
        #for pop in population:
        #    if not pop.valid: pop.evaluate()
        population = [pop for pop in population if pop.fitness.valid]


        fitnesses = [individual.fitness.fitness for individual in population if individual.valid]

        # Split Columns into Lists
        fitnessColumns = [[fit[i] for fit in fitnesses] for i,obj in enumerate(statBox.problem.objectives)]


    
        # Calculate Medians and Spreads
        fitnessMedians = [median(fitCol) for fitCol in fitnessColumns]
        fitnessSpreads = [spread(fitCol) for fitCol in fitnessColumns]
        
        # Initialize Reference Point on Initial Run
        if initial == True:
            statBox.referencePoint = [o.med for o in statBox.problem.objectives]

            

        # Calculate IBD & IBS
        norms = [[min(fitnessColumns[i]+[statBox.referencePoint[i]]), max(fitnessColumns[i]+[statBox.referencePoint[i]])] for i,obj in enumerate(statBox.problem.objectives)]

        lossInQualities = [{"qual": loss_in_quality(statBox.problem, [statBox.referencePoint], fit, norms), "index": i} for i,fit in enumerate(fitnesses)]
        
        lossInQualities.sort(key=lambda(r): r["qual"])
        if len(fitnesses) > 0: 
            best_fitness = fitnesses[lossInQualities[0]["index"]]
        else:
            best_fitness = fitnessMedians
        lossInQualities = [item["qual"] for item in lossInQualities]
        #best_fitness = [min(fitCol) for fitCol in fitnessColumns if len(fitCol) > 0]

        # + IGD Calculation: This would only work if the true PF is known.
        if IGDMEASURE is True:
            approximate = []
            true_PF = readpf(statBox.problem)
            for individual in population:
                temp = []
                for x in individual.fitness.fitness: temp.append(round(x, 5))
                approximate.append(temp)
        # - IGD Calculation

        IBD = median(lossInQualities)
        IBS = spread(lossInQualities)
        if IGDMEASURE is True:
            IGD = IGD_Calculation.IGD(approximate, true_PF)
        
        if initial == True:
            IBD = 1.0
            statBox.referenceIBD = 1.0
            # TODO: Need to come up with an smart way to assign reference IGD: This is stupid I ran into the problems
            # with this
            if IGDMEASURE is True:
                statBox.referenceIGD = 1e30
        
        
        changes = []
        # Print Option
        if printOption == True:
            outString = ""
            
            if initial:
                outString += str(statBox.numEval) + ","
                for med, spr, initmed, obj, o in zip(statBox.referencePoint, [0 for x in statBox.problem.objectives],
                                                 statBox.referencePoint, statBox.problem.objectives,
                                                 range(len(statBox.problem.objectives))):
                    change = percentChange(med, initmed, obj.lismore, obj.low, obj.up)
                    changes.append(float(change.strip("%")))
                    statBox.bests[o] = changes[-1]
                    statBox.bests_actuals[o] = med
                    outString += str("%8.4f" % med) + "," + change + "," + str("%8.4f" % spr) + ","
                    if statBox.numEval in statBox.foam[o]: statBox.foam[o][statBox.numEval].append(change)
                    else: statBox.foam[o][statBox.numEval] = [change]
                outString += str("%8.4f" % IBD) + "," + percentChange(statBox.referenceIBD, statBox.referenceIBD, True, 0, 1) + "," + str("%8.4f" % IBS)
                if IGDMEASURE is True:
                    outString += "," + str("%8.4f" % IGD) + "," + percentChange(statBox.referenceIGD, statBox.referenceIGD, True, 0, 1e3)
            else:
                outString += str(statBox.numEval) + ","
                for med, spr, initmed, obj, o in zip(best_fitness, fitnessSpreads, statBox.referencePoint,
                                                 statBox.problem.objectives, range(len(statBox.problem.objectives))):
                    change = percentChange(med, initmed, obj.lismore, obj.low, obj.up)
                    changes.append(float(change.strip("%")))
                    if changes[-1] < statBox.bests[o]: 
                        statBox.bests[o] = changes[-1]
                        statBox.bests_actuals[o] = med
                    outString += str("%8.4f" % med) + "," + change + "," + str("%8.4f" % spr) + ","
                    if statBox.numEval in statBox.foam[o]: statBox.foam[o][statBox.numEval].append(change)
                    else: statBox.foam[o][statBox.numEval] = [change]
                outString += str("%8.4f" % IBD) + "," + percentChange(IBD, statBox.referenceIBD, True, 0, 1) + "," + str("%8.4f" % IBS)
                print outString  + ", violations: " + str("%4.1f" % violationsPercent)
                if IGDMEASURE is True:
                    outString += "," + str("%8.4f" % IGD) + "," + percentChange(IGD, statBox.referenceIGD, True, 0, 1e3)




            # if initial:
            #     if IGDMEASURE is True:
            #         print outString  + ", violations: " + str("%4.1f" % violationsPercent) + "||IGD: " + str(IGD_Calculation.IGD(approximate, true_PF))
            #     else: print outString  + ", violations: " + str("%4.1f" % violationsPercent)
            # else:
            #     # print outString  + ", violations: " + str("%4.1f" % violationsPercent) + "||IGD: " + str(IGD(approximate, true_PF))
            #     # print str(statBox.numEval) + "|IGD: |" + str(IGD(approximate, true_PF)) + "|Fitness:| ", normalized_median
            #     print "|Fitness: |", fitnessMedians
            fa.write(outString + "\n")
        
            
        # Add Stat to the Stat Box
        trunk = []
        for i,pop in enumerate(population):
            trunk.append(jmoo_individual(statBox.problem, pop.decisionValues, pop.fitness.fitness))
            #if i < 5: print trunk[-1].decisionValues, statBox.problem.evalConstraints(trunk[-1].decisionValues)
        statBox.box.append(jmoo_stats(trunk, fitnesses, best_fitness, fitnessSpreads, statBox.numEval, gen, IBD, IBS, changes))
        fa.close()
###########
### Utility Functions
###########

def percentChange(new, old, lismore, low, up):
    # print "old: ", old
    # print "new: ", new
    return str("%1.1f" % changeFromOld(new, old, lismore, low, up)) + "%"

def changeFromOld(new, old, lismore, low, up):
    if new < 0 or old < 0: 
        ourlismore = not lismore
        new = abs(new)
        old = abs(old)
    else: ourlismore = lismore
    # if new == 0 or old == 0: return 0 if ourlprintismore else 110
    new = normalize(new, low, up)
    old = normalize(old, low, up)
    if old == 0: x = 0
    else: x = (new/float(old))
    if x == 0: 
        if ourlismore: return 0
        else: return 1
    else: return 100.0*x**(1 if ourlismore else -1)
def median(list):
    return getPercentile(list, 50)

def spread(list):
    return getPercentile(list, 75) - getPercentile(list, 25)

def getPercentile(list, percentile):
        if len(list) == 0: return 0
        #sort the list
        list.sort()
        
        k = (len(list) - 1) * (percentile/100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            val = list[int(k)]
        else:
            d0 = list[int(f)] * (c-k)
            d1 = list[int(c)] * (k-f)
            val = d0+d1
        return val

def normalize(x, min, max):
    tmp = float((x - min)) / \
                (max - min + 0.000001) 
    if   tmp > 1 : return 1
    elif tmp < 0 : return 0
    else         : return tmp 
    
def loss_in_quality(problem, pop, fit1, norms):
    "Loss in Quality Indicator"
    weights = [-1 if o.lismore else +1 for o in problem.objectives]    
    k = len(weights)
    
    # Calculate the loss in quality of removing fit1 from the population
    F = []
    for X2 in pop:
        F.append(-k/(sum([-math.exp(-w*(normalize(p2,n[0],n[1]) - normalize(p1,n[0],n[1]))/k) for w,p1,p2,n in zip(weights,fit1,X2,norms)])))
    F1 = sum(F)
    
    return F1