
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
"Objective Space Plotter"

from pylab import *
import csv
from jmoo_problems import *
from jmoo_algorithms import *
from jmoo_properties import *
from utility import *
import numpy
from time import *
import os
from deap.tools.support import ParetoFront

def joes_charter_reporter(problems, algorithms, tag=""):
    date_folder_prefix = strftime("%m-%d-%Y")
    
    fignum = 0
    
            
    base = []
    final = []
    RRS = []
    data = []
    foam = []
    baseline =[]
    
    for p,prob in enumerate(problems):
        base.append([])
        final.append([])
        RRS.append([])
        data.append([])
        foam.append([])
        finput = open("data/" + prob.name + "-p" + str(MU) + "-d"  + str(len(prob.decisions)) + "-o" + str(len(prob.objectives)) + "-dataset.txt", 'rb')
        reader = csv.reader(finput, delimiter=',')
        initial = []

        filename = "data/" + prob.name + "-p" + str(MU) + "-d"  + str(len(prob.decisions)) + "-o" + str(len(prob.objectives)) + "-dataset.txt"
        row_count = sum(1 for _ in csv.reader( open(filename)))
        for i,row in enumerate(reader):
            if i > 1 and i != row_count-1:
                    row = map(float, row)
                    assert(prob.validate(row) is True), "Something's wrong"
                    initial.append(prob.evaluate(row)[-1])
        baseline.append(initial)


            
        for a,alg in enumerate(algorithms):
            
            f2input = open(DATA_PREFIX + RRS_TABLE + "_" + prob.name + "-p" + str(MU) + "-d"  + str(len(prob.decisions)) + "-o" + str(len(prob.objectives)) + "_" + alg.name + DATA_SUFFIX, 'rb')
            f3input = open("data/results_" + prob.name + "-p" + str(MU) + "-d"  + str(len(prob.decisions)) + "-o" + str(len(prob.objectives)) + "_" + alg.name + ".datatable", 'rb')
            f4input = open(DATA_PREFIX + "decision_bin_table" + "_" + prob.name + "-p" + str(MU) + "-d"  + str(len(prob.decisions)) + "-o" + str(len(prob.objectives)) + "_" + alg.name + DATA_SUFFIX, 'rb')

            # print f2input.name
            # print f3input.name
            # print f4input.name
            # data/RRS_TABLE__DTLZ1_12_8-p100-d12-o8_GALE.datatable
            # data/results_DTLZ1_12_8-p100-d12-o8_GALE.datatable
            # data/decision_bin_table_DTLZ1_12_8-p100-d12-o8_GALE.datatable
            # import pdb
            # pdb.set_trace()
            
            reader2 = csv.reader(f2input, delimiter=',')
            reader3 = csv.reader(f3input, delimiter=',')
            reader4 = csv.reader(f4input, delimiter=',')
            base[p].append( [] )
            final[p].append( [] )
            RRS[p].append( [] )
            data[p].append( [] )
            foam[p].append( [] )
            
            
            
            
            
            
            for i,row in enumerate(reader4):
                n = len(prob.decisions)
                o = len(prob.objectives)
                candidate = [float(col) for col in row[:n]]
                fitness = [float(col) for col in row[n:n+o]]#prob.evaluate(candidate)
                final[p][a].append(candidate+fitness)
            
            """
            for o,obj in enumerate(prob.objectives): 
                RRS[p][a].append([])
                RRS[p][a][o] = {}
                foam[p][a].append([])
                foam[p][a][o] = {}
            
                        
            for i,row in enumerate(reader2):
                k = len(prob.objectives)
                fitness = [float(col) for col in row[-k-1:-1]]
                for o,fit in enumerate(fitness):
                    n = int(row[k])
                    n = (int(round(n/5.0)*5.0))
                    if n in RRS[p][a][o]: RRS[p][a][o][n].append(fit)
                    else: RRS[p][a][o][n] = [fit]    
            """
            for i,row in enumerate(reader3):
                if not str(row[0]) == "0":
                    for j,col in enumerate(row):
                        if i == 0:
                            data[p][a].append([])
                        else:
                            if not col == "":

                                data[p][a][j].append(float(col.strip("%)(")))
                    """
                    # row is now read
                    if i > 0:
                        for o,obj in enumerate(prob.objectives):
                            n = data[p][a][0][-1]
                            n = (int(round(n/20.0)*20.0))
                            if n in foam[p][a][o]: foam[p][a][o][n].append(float(data[p][a][o*3+2][-1]))
                            else: foam[p][a][o][n] = [float(data[p][a][o*3+2][-1])]         
                    """ 





    fignum = 0
    colors = ['r', 'b', 'g']
    from matplotlib.font_manager import FontProperties
    font = {'family' : 'sans-serif',
            'weight' : 'normal',
            'size'   : 8}
    
    matplotlib.rc('font', **font)
    fontP = FontProperties()
    fontP.set_size('x-small')
    
    
    codes = ["b*", "r.", "g*"]
    
    line =  "-"
    dotted= "--"
    algnames = [alg.name for alg in algorithms]
    axy = [0,1,2,3]
    axx = [0,0,0,0]
    codes2= ["b-", "r-", "g-"]
    colors= ["b", "r", "g"]
    ms = 8
    from mpl_toolkits.mplot3d import Axes3D
    #fig  = plt.figure()
    #ax = fig.gca(projection='3d')
    
    #"""
    #hypervolume and diversity deb metric stuff
    # print 'hello'
    #
    # for p,prob in enumerate(problems):
    #     for a,alg in enumerate(algorithms):
    #
    #         n = len(prob.decisions)
    #         population = [jmoo_individual(prob, f[:n], f[n:]) for f in final[p][a]]
    #         population = deap_format(prob, population)
    #         front = population
    #         front.sort(key=lambda x: x.fitness.values)
    #
    #         from stats import diversity
    #         import hv
    #         refPoint = ['s' for o in prob.objectives]
    #         for row in population:
    #             for i,fit in enumerate(row.fitness.values):
    #                 if refPoint[i] == 's': refPoint[i] = row.fitness.values[i]
    #                 else:
    #                     if prob.objectives[i].lismore:
    #                         if refPoint[i] < row.fitness.values[i]: refPoint[i] = row.fitness.values[i]
    #                     else:
    #                         if refPoint[i] > row.fitness.values[i]: refPoint[i] = row.fitness.values[i]
    #
    #         finput = open("data/" + prob.name + "-p" + str(MU) + "-d"  + str(len(prob.decisions)) + "-o" + str(len(prob.objectives)) + "-dataset.txt", 'rb')
    #         reader = csv.reader(finput, delimiter=',')
    #
    #         hellPoint = []
    #         midPoint = []
    #         initialFront = []
    #         lowPoint = []
    #
    #         for i,line in enumerate(reader):
    #             if i > 100:
    #                 midPoint.append(float(line[1]))
    #                 if (prob.objectives[i-100-1].lismore):
    #                     hellPoint.append(float(line[2]))
    #                     lowPoint.append(float(line[0]))
    #                 else:
    #                     hellPoint.append(float(line[0]))
    #                     lowPoint.append(float(line[2]))
    #                     hellPoint[i-100-1] *= -1
    #                     midPoint[i-100-1] *= -1
    #                     refPoint[i-100-1] *= -1
    #                     lowPoint[i-100-1] *= -1
    #             else:
    #                 if i > 0:
    #                     n = len(prob.decisions)
    #                     o = len(prob.objectives)
    #                     candidate = [float(col) for col in line[:n]]
    #                     fitness = prob.evaluate(candidate)
    #                     initialFront.append(jmoo_individual(prob, candidate, fitness))
    #
    #         initialFront = deap_format(prob, initialFront)
    #         refPoint = ['s' for o in prob.objectives]
    #         for row in initialFront:
    #             for i,fit in enumerate(row.fitness.values):
    #                 if refPoint[i] == 's': refPoint[i] = row.fitness.values[i]
    #                 else:
    #                     if prob.objectives[i].lismore:
    #                         if refPoint[i] < row.fitness.values[i]: refPoint[i] = row.fitness.values[i]
    #                     else:
    #                         if refPoint[i] > row.fitness.values[i]: refPoint[i] = row.fitness.values[i]
    #         for item in front:
    #             item.fitness.weightedvalues = [item.fitness.values[o] for o in range(len(prob.objectives))]
    #         for item in initialFront:
    #             item.fitness.weightedvalues = [item.fitness.values[o] for o in range(len(prob.objectives))]
    #         for o,obj in enumerate(prob.objectives):
    #             if obj.lismore == False:
    #                 for item in front:
    #                     item.fitness.weightedvalues[o] = item.fitness.weightedvalues[o]*-1
    #                 for item in initialFront:
    #                     item.fitness.weightedvalues[o] = item.fitness.weightedvalues[o]*-1
    #
    #         #COMPUTE THE DEB SPREAD
    #         if len(front) > 0:
    #             div = diversity(front, front[0].fitness.values, front[-1].fitness.values)
    #         else:
    #             div = 0
    #
    #         #COMPUTE THE HV
    #         start = time()
    #         HV = hv.HyperVolume(refPoint)
    #         result =    HV.compute([item.fitness.weightedvalues for item in front])
    #
    #         #resultInit =    HV.compute([item.fitness.weightedvalues for item in initialFront])
    #         #HV = hv.HyperVolume(hellPoint)
    #         #hellResult = HV.compute([item.fitness.weightedvalues for item in front])
    #         #hellResultInit = HV.compute([item.fitness.weightedvalues for item in initialFront])
    #         #HV = hv.HyperVolume(midPoint)
    #         #midResult = HV.compute([item.fitness.weightedvalues for item in front])
    #         #midResultInit = HV.compute([item.fitness.weightedvalues for item in initialFront])
    #
    #
    #         end = time()
    #         hvtime = (end-start)
    #         zzz = prob.name + "-p" + str(MU) + "-d"  + str(len(prob.decisions)) + "-o" + str(len(prob.objectives)) + "," + alg.name + "," + str(len(front))
    #         zzz += "," + str( div ) + ", " + str(result) + ", " + str(hvtime)# + "," + str(midResult) + "," + str(resultInit) + "," + str(hellResultInit) + ',' + str(midResultInit)
    #         print zzz
    #
    #
    #
    #         #import matplotlib.pyplot as plt
    #         # import numpy
    #         #
    #         #front = numpy.array([ind.fitness.values for ind in population])
    #         #optimal_front = numpy.array(optimal_front)
    #         #plt.scatter(optimal_front[:,0], optimal_front[:,1], c="r")
    #         #plt.scatter(front[:,0], front[:,1], c="b")
    #         #plt.axis("tight")
    #         #plt.show()
    #         #d_ = []
    #         #for i,fit_i in enumerate(front):
    #         #    fma = []
    #         #    for j,fit_j in enumerate(front):
    #         #        if not i == j:
    #         #            fma.append(sum([abs(fit_i.fitness.values[k] - fit_j.fitness.values[k]) for k in range(len(prob.objectives))]))
    #         #    d_.append(min(fma))
    #         #d_bar = average(d_)
    #         #ssm =   ((1 / float(len(front) - 1)) * sum([(d_bar - d_i)**2 for d_i in d_]))**0.5
    #         #print prob.name + "," + alg.name + ", size of pareto front: ", len(front), ", spacing: ", ssm
    #
    # #"""
       
    f, axarr = plt.subplots(2, len(prob.objectives))   #for dtlz123456
    #f, axarr = plt.subplots(3, len(prob.objectives)-1)   #for pom3abc
    #f, axarr = plt.subplots(3, len(prob.objectives))   #for xomo gr fl o2
    F = gcf()
    DefaultSize = F.get_size_inches()
    F.set_size_inches( (DefaultSize[0]*1.5, DefaultSize[1]) )
    for p, prob in enumerate(problems):
                oo = -1
                for o, obj in enumerate(prob.objectives):
                    min_yaxis = 1e32
                    max_yaxis = -1e32
                    b = base[p][o]
                    if o == 11:
                       pass
                    else:
                        oo += 1
                        maxEvals = 0
                        for a,alg in enumerate(algorithms):
                            maxEvals = max(maxEvals, max(data[p][a][0]))


                        # print baseline
                        from numpy import percentile
                        first_percentile = percentile(baseline[p], 25)
                        second_percentile = percentile(baseline[p], 50)
                        third_percentile = percentile(baseline[p], 75)


                        print first_percentile, second_percentile, third_percentile


                        for i, sc in enumerate([first_percentile, second_percentile, third_percentile]):
                            keylist = []
                            scores = []
                            keylist.extend([j for j in xrange(10, int(maxEvals - 3000))])
                            scores.extend([sc for _ in xrange(10, int(maxEvals - 3000))])
                            axarr[oo].plot(keylist, scores, label=str(i+1)+"quadrant", marker="+", color="BLACK", markersize=7, markeredgecolor='none')



                        for a,alg in enumerate(algorithms):    
                            
                            scores = {}
                            
                            for score,eval in zip(data[p][a][o*3+1], data[p][a][0]):
                                eval = int(round(eval/5.0)*5.0)
                                # print score
                                if eval in scores: scores[eval].append(score)
                                else: scores[eval] = [score]

                            
                            keylist = []
                            scorelist = []
                            smallslist = []
                            for eval in sorted(scores.keys()):
                                lq = getPercentile(scores[eval], 25)
                                uq = getPercentile(scores[eval], 75) 
                                scores[eval] = [score for score in scores[eval] ]#if score >= lq and score <= uq ]
                                for item in scores[eval]:
                                    keylist.append(eval)
                                    scorelist.append(min(scorelist + [item]))
                                    if len(smallslist) == 0: 
                                        smallslist.append(min(scores[eval]))
                                    else:
                                        smallslist.append(    min(min(scores[eval]), min(smallslist))  )


                            
                            if oo==0:
                                axarr[oo].set_ylabel(prob.name + "\n_o"+str(len(prob.objectives)), fontweight='bold', fontsize=14)
                            if p ==0: axarr[oo].set_title(prob.objectives[oo].name, fontweight='bold', fontsize=14)
                            if p ==(len(problems)-1): axarr[oo].set_xlabel("(Log) NumEvals")
                            ax2 = axarr[oo].twinx()
                            ax2.get_yaxis().set_ticks([])
                            if oo==(len(prob.objectives)-1): ax2.set_ylabel("Quality")
                            # #print scorelist
                            # print "-" *30
                            # print alg.name, o
                            # print keylist
                            print alg.name
                            print scorelist
                            # exit()
                            # print min(scorelist) - 0.1, max(scorelist) + 0.1
                            axarr[oo].plot(keylist, scorelist, label=alg.name, marker=alg.type, color=alg.color, markersize=7, markeredgecolor='none') #MARKER PLOTS
                            #axarr[p][oo].plot([min(keylist)]+keylist, [100]+smallslist, color=alg.color) #BOTTOMLINE
                            axarr[oo].plot([x for x in range(0,10000,10)], [100 for x in range(0,10000,10)], color="Black") #BASELINE
                            min_yaxis = min(min(scorelist), min_yaxis)
                            # max_yaxis = max(max(scorelist), max_yaxis)
                            max_yaxis = max(max(max(scorelist), max_yaxis), third_percentile)
                            axarr[oo].set_autoscale_on(True)
                            axarr[oo].set_xlim([-10, 10000])

                            #axarr[p][oo].set_ylim([20, 160])# -- xomo
                            #axarr[p][oo].set_ylim([-5, 115])
                            axarr[oo].set_ylim([min_yaxis -  0.1 * max_yaxis, max_yaxis + 0.1 * max_yaxis])# -- tera
                            # axarr[oo].set_ylim([int(min_yaxis*0.9), int(max_yaxis*1.1)])  # NRP/MONRP
                            axarr[oo].set_xscale('log', nonposx='clip')
                            if oo == 0:
                                axarr[oo].legend(loc='best')
                        
                            
    if not os.path.isdir('charts/' + date_folder_prefix):
        os.makedirs('charts/' + date_folder_prefix)
    
    fignum = len([name for name in os.listdir('charts/' + date_folder_prefix)]) + 1
    print fignum

    plt.savefig('charts/' + date_folder_prefix + '/figure' + str("%02d" % fignum) + "_" + prob.name + "_" + tag + '.png', dpi=100)
    cla()
    clf()
    close()
    #show()

    # #--- For IGD -- Values #
    # min_number = 1e32
    # max_number = -1e32
    #
    # for p,prob in enumerate(problems):
    #     for a,alg in enumerate(algorithms):
    #
    #         # to handle multiple runs
    #         scores = {}
    #         for score,eval in zip(data[p][a][-2], data[p][a][0]):
    #             # print score
    #             if eval in scores: scores[eval].append(score)
    #             else: scores[eval] = [score]
    #
    #         score_list = []
    #         try:
    #             for eval in sorted(data[p][a][0]):
    #                 score_list.append(median(scores[int(eval)]))
    #         except:
    #             import traceback
    #             traceback.print_exc()
    #             print scores.keys()
    #             exit()
    #
    #         # for xx, yy in zip(data[p][a][0], score_list):
    #         #     print xx, yy
    #         # exit()
    #
    #
    #         plt.plot(sorted(data[p][a][0]), score_list, label=alg.name, marker=alg.type, color=alg.color) #MARKER PLOTS )
    #         min_number = min(min(data[p][a][-2]), min_number)
    #         max_number = max(max(data[p][a][-2]), max_number)
    #         #max_number = 10
    #
    #
    #
    # plt.xlabel('Generations')
    # plt.ylabel('IGD')
    # plt.title('Variation of IGD')
    # plt.legend()
    # print min_number, max_number
    # plt.ylim([min_number - 1, max_number + 1])# -- tera
    # plt.savefig('charts/' + date_folder_prefix + '/figure' + str("%02d" % fignum) + "_" + prob.name + "_" + "IGD" + '.png', dpi=100)
    # cla()




