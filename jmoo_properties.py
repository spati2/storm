
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
"Property File.  Defines default settings."

from jmoo_algorithms import *
from jmoo_problems import *

# from Problems.CPM.cpm import *
# from Problems.CPM.cpm_reduction import *
# from Problems.NRP.nrp import *
# from Problems.MONRP.monrp import *



# JMOO Experimental Definitions
algorithms = [
              # jmoo_GALE(),
              # jmoo_NSGAII(),
              # jmoo_SPEA2(),
              # jmoo_DE(),
              # jmoo_MOEAD(),
              # jmoo_NSGAIII(),
                jmoo_NSGAIII_New()
              ]

problems =[
    # dtlz1(9, 5),
    # dtlz2(14, 5),
    # dtlz3(14, 5),
    # dtlz4(14, 5),
    # dtlz1(7, 3),
    # dtlz2(12, 3),
    # dtlz3(12, 3),
    # dtlz4(12, 3),
    dtlz1(12, 8),
    # dtlz2(17, 8),
    # dtlz3(17, 8),
    # dtlz4(17, 8),
    # dtlz1(14, 10), dtlz2(19, 10),
    # dtlz3(19, 10),
    # dtlz4(19, 10),
    # dtlz1(19, 15),
    # dtlz2(24, 15),
    # dtlz3(24, 15),
    # dtlz4(24, 15)
    # NRP(50, 5, 5, 20, 120)
    #MONRP(50, 5, 5, 20, 120)
    # cpm_apache(),cpm_X264(), cpm_SQL_4553(), cpm_SQL_100(), cpm_LLVM(), cpm_BDBJ(), cpm_BDBC()
    # cpm_apache_training_reduction(treatment=None),
    # cpm_X264(treatment=None),
    # cpm_SQL(treatment=None),
    # cpm_LLVM(treatment=None),
    # cpm_BDBJ(treatment=None),
    # cpm_BDBC(treatment=None)


    # #, fonseca(3), srinivas(), schaffer(), osyczka2(),# water()
       #camel(), ant(),  forrest(), ivy(), jedit(), lucene(), poi(), synapse(), velocity(), xerces(),
     #antRF()  , camelRF(),  forrestRF(), ivyRF(), jeditRF(), luceneRF(), poiRF(), synapseRF(), velocityRF(), xercesRF(),
     #antW()  , camelW(),  forrestW(), ivyW(), jeditW(), luceneW(), poiW(), synapseW(), velocityW(), xercesW()


]

build_new_pop = False                                       # Whether or not to rebuild the initial population


# JMOO Universal Properties
repeats = 1   #Repeats of each MOEA
MU      = 92   #Population Size
PSI     = 400    #Maximum number of generations

# Properties of GALE
GAMMA   = 0.15  #Constrained Mutation Parameter
EPSILON = 1.00  #Continuous Domination Parameter
LAMBDA =  3     #Number of lives for bstop

# Propoerties of DE
F = 0.75 # extrapolate amount
CF = 0.3 # prob of cross over

# Properties of MOEA/D
T = 30  # Neighbourhood size
MOEAD_F = 0.5
MOEAD_CF = 1.0

# Properties of Anywhere
ANYWHERE_EXPLOSION = 5
ANYWHERE_POLES = 20  # number of actual poles is 2 * ANYWHERE_POLES

# Properties of NSGAIII
# NSGA3_P = 5 # not required anymore since this is not strictly followed. Looked at nsga3 paper section V


# File Names
DATA_PREFIX        = "Data/"
DEFECT_PREDICT_PREFIX = "defect_prediction/"
VERSION_SPACE_PREFIX = "version_space/"

"decision bin tables are a list of decisions and objective scores for a certain model"
DECISION_BIN_TABLE = "decision_bin_table"

"result scores are the per-generation list of IBD, IBS, numeval,scores and change percents for each objective - for a certain model"
RESULT_SCORES      = "result_"

SUMMARY_RESULTS    = "summary_"

RRS_TABLE = "RRS_TABLE_"
DATA_SUFFIX        = ".datatable"


