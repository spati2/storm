
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

from pylab import *

from jmoo_properties import *
from utility import *
from Algorithms.DEAP.tools.support import ParetoFront


def read_initial_population(prob, filename):
    fd_initial_data = open(filename, 'rb')
    reader_initial_data = csv.reader(fd_initial_data, delimiter=',')
    initial = []
    row_count = sum(1 for _ in csv.reader(open(filename)))
    for i,row in enumerate(reader_initial_data):
        if i > 1 and i != row_count-1:
                row = map(float, row)
                try: initial.append(prob.evaluate(row)[-1])
                except: pass
    return initial


def charter_reporter(problems, algorithms, Configurations, tag=""):
    date_folder_prefix = strftime("%m-%d-%Y")

    fignum = 0
    from Graphics.PerformanceMeasures.base import get_archive_data
    get_archive_data(problems, algorithms)

    base = []
    final = []
    data = []

    for p,prob in enumerate(problems):
        base.append([])
        final.append([])
        data.append([])

        for a,alg in enumerate(algorithms):

            # finput = open("data/" + prob.name + "-p" + str(Configurations["Universal"]["Population_Size"]) + "-d"  + str(len(prob.decisions)) + "-o" + str(len(prob.objectives)) + "-dataset.txt", 'rb')
            f3input = open("data/results_" + prob.name + "-p" + str(Configurations["Universal"]["Population_Size"]) + "-d"  + str(len(prob.decisions)) + "-o" + str(len(prob.objectives)) + "_" + alg.name + ".datatable", 'rb')
            # f4input = open(DATA_PREFIX + "decision_bin_table" + "_" + prob.name+ "-p" + str(Configurations["Universal"]["Population_Size"]) + "-d"  + str(len(prob.decisions)) + "-o" + str(len(prob.objectives))  + "_" + alg.name + DATA_SUFFIX, 'rb')
            # reader = csv.reader(finput, delimiter=',')
            reader3 = csv.reader(f3input, delimiter=',')
            # reader4 = csv.reader(f4input, delimiter=',')
            base[p].append( [] )
            final[p].append( [] )
            data[p].append( [] )
            print f3input

            # for i,row in enumerate(reader):
            #     if i <= 100 and i > 0:
            #         candidate = [float(col) for col in row]
            #         fitness = prob.evaluate(candidate)
            #         base[p][a].append(candidate+fitness)
            #
            # for i,row in enumerate(reader4):
            #     n = len(prob.decisions)
            #     candidate = [float(col) for col in row[:n]]
            #     fitness = prob.evaluate(candidate)
            #     final[p][a].append(candidate+fitness)

            for i,row in enumerate(reader3):
                if not str(row[0]) == "0":
                    for j,col in enumerate(row):
                        if i == 0:
                            data[p][a].append([])
                        else:
                            if not col == "":
                                data[p][a][j].append(float(col.strip("%)(")))






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




    for p,prob in enumerate(problems):
                f, axarr = plt.subplots(len(prob.objectives))#+1, len(prob.objectives))

                for o, obj in enumerate(prob.objectives):
                    maxEvals = 0
                    for a,alg in enumerate(algorithms):
                        maxEvals = max(maxEvals, max(data[p][a][0]))
                    for a,alg in enumerate(algorithms):

                        scores = {}
                        for score,eval in zip(data[p][a][o*3+2], data[p][a][0]):
                            eval = int(round(eval/5.0)*5.0)
                            if eval in scores: scores[eval].append(score)
                            else: scores[eval] = [score]

                        keylist = [1]
                        scorelist = [100]
                        smallslist = [100]
                        for eval in sorted(scores.keys()):
                            lq = getPercentile(scores[eval], 25)
                            uq = getPercentile(scores[eval], 75)
                            scores[eval] = [score for score in scores[eval] if score >= lq and score <= uq ]
                            for item in scores[eval]:
                                keylist.append(eval)
                                scorelist.append(item)
                                if len(smallslist) == 0:
                                    smallslist.append(min(scores[eval]))
                                else:
                                    smallslist.append(    min(min(scores[eval]), min(smallslist))  )

                        axarr[o].plot(keylist, scorelist, linestyle='None', label=alg.name, marker=alg.type, color=alg.color, markersize=8, markeredgecolor='none')
                        axarr[o].plot(keylist, smallslist, color=alg.color)
                        # axarr[o].set_ylim(0, 130)
                        # axarr[o].set_autoscale_on(True)
                        axarr[o].set_xlim([-10, 10000])
                        axarr[o].set_xscale('log', nonposx='clip')
                        axarr[o].set_ylabel(obj.name)


                if not os.path.isdir('charts/' + date_folder_prefix):
                    os.makedirs('charts/' + date_folder_prefix)

                f.suptitle(prob.name)
                fignum = len([name for name in os.listdir('charts/' + date_folder_prefix)]) + 1
                plt.legend(loc='lower center', bbox_to_anchor=(1, 0.5))
                plt.savefig('charts/' + date_folder_prefix + '/figure' + str("%02d" % fignum) + "_" + prob.name + "_" + tag + '.png', dpi=100)
                cla()
    #show()

