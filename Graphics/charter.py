
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

# from pylab import *
from time import *
import os

from pylab import *

from jmoo_properties import *
from utility import *


def charter_reporter(problems, algorithms, Configurations, tag=""):
    date_folder_prefix = strftime("%m-%d-%Y")
    MU = Configurations["Universal"]["Population_Size"]
            
    initial_data = []
    final = []
    RRS = []
    generation_data = []
    foam = []
    baseline =[]
    
    for p,prob in enumerate(problems):
        initial_data.append([])
        RRS.append([])
        generation_data.append([])
        foam.append([])
        filename = "Data/" + prob.name + "-p" + str(MU) + "-d"  + str(len(prob.decisions)) + "-o" + str(len(prob.objectives)) + "-dataset.txt"
        fd_initial_data = open(filename, 'rb')
        reader_initial_data = csv.reader(fd_initial_data, delimiter=',')
        initial = []
        
        row_count = sum(1 for _ in csv.reader(open(filename)))
        for i,row in enumerate(reader_initial_data):
            if i > 1 and i != row_count-1:
                    row = map(float, row)
                    try: initial.append(prob.evaluate(row)[-1])
                    except: pass
                
        baseline.append(initial)

        for a, alg in enumerate(algorithms):
            fd_initial_data = open("data/" + prob.name + str(MU) + "dataset.txt", 'rb')
            reader_initial_data = csv.reader(fd_initial_data, delimiter=',')

            fd_statistic_file = open("Data/results_" + prob.name + "-p" + str(MU) + "-d"  + str(len(prob.decisions)) + "-o" + str(len(prob.objectives)) + "_" + alg.name + ".datatable", 'rb')
            reader_statistic_file = csv.reader(fd_statistic_file, delimiter=',')

            initial_data[p].append([])
            generation_data[p].append([])
            I am refactoring this!!

            for i,row in enumerate(reader_initial_data):
                if 0 < i <= 100:
                    candidate = [float(col) for col in row]
                    fitness = prob.evaluate(candidate)
                    initial_data[p][a].append(candidate+fitness)
            
            for i,row in enumerate(reader_statistic_file):
                if not str(row[0]) == "0":
                    for j, col in enumerate(row):
                        if i == 0: generation_data[p][a].append([])
                        else:
                            if not col == "": generation_data[p][a][j].append(float(col.strip("%)(")))


    fignum = 0
    colors = ['r', 'b', 'g']
    from matplotlib.font_manager import FontProperties
    font = {'family' : 'sans-serif',
            'weight' : 'normal',
            'size'   : 8}

    from matplotlib import rc
    rc('font', **font)
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
    #fig  = plt.figure()
    #ax = fig.gca(projection='3d')
    


    f, axarr = plt.subplots(3, len(prob.objectives))   #for dtlz123456
    # f, axarr = plt.subplots(3, len(prob.objectives)-1)   #for pom3abc
    #f, axarr = plt.subplots(3, len(prob.objectives))   #for xomo gr fl o2


    F = gcf()
    DefaultSize = F.get_size_inches()
    F.set_size_inches( (DefaultSize[0]*1.5, DefaultSize[1]) )
    for p, prob in enumerate(problems):
                oo = -1
                for o, obj in enumerate(prob.objectives):
                    min_yaxis = 1e32
                    max_yaxis = -1e32
                    b = initial_data[p][o]
                    if o == 11:
                       pass
                    else:
                        oo += 1
                        maxEvals = 0
                        for a,alg in enumerate(algorithms):
                            maxEvals = max(maxEvals, max(generation_data[p][a][0]))


                        # print baseline
                        from numpy import percentile
                        first_percentile = percentile(baseline[p], 25)
                        second_percentile = percentile(baseline[p], 50)
                        third_percentile = percentile(baseline[p], 75)

                        for i, sc in enumerate([first_percentile, second_percentile, third_percentile]):
                            keylist = []
                            scores = []
                            keylist.extend([j for j in xrange(10, int(maxEvals - 3000))])
                            scores.extend([sc for _ in xrange(10, int(maxEvals - 3000))])
                            axarr[oo].plot(keylist, scores, label=str(i+1)+"quadrant", marker="+", color="BLACK", markersize=7, markeredgecolor='none')



                        for a,alg in enumerate(algorithms):    
                            
                            scores = {}
                            
                            for score,eval in zip(generation_data[p][a][o*3+1], generation_data[p][a][0]):
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
                            axarr[oo].set_ylim([min_yaxis -  0.1 * max_yaxis, max_yaxis + 0.1 * max_yaxis])# -- Tera
                            # axarr[oo].set_ylim([int(min_yaxis*0.9), int(max_yaxis*1.1)])  # NRP/MONRP
                            axarr[oo].set_xscale('log', nonposx='clip')
                            if oo == 0:
                                axarr[oo].legend(loc='best')
                        
                            
    if not os.path.isdir('Charts/' + date_folder_prefix):
        os.makedirs('Charts/' + date_folder_prefix)
    
    fignum = len([name for name in os.listdir('Charts/' + date_folder_prefix)]) + 1
    print fignum

    plt.savefig('Charts/' + date_folder_prefix + '/figure' + str("%02d" % fignum) + "_" + prob.name + "_" + tag + '.png', dpi=100)
    cla()
    clf()
    close()




