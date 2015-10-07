
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
"Standardized MOEA code for running any MOEA"

from jmoo_algorithms import *
from jmoo_stats_box import *
from jmoo_properties import *
from Algorithms.GALE.Fastmap.Moo import *
from pylab import *
import jmoo_properties

import os, sys, inspect


def jmoo_evo(problem, algorithm, repeat=-1, toStop = bstop):
    """
    ----------------------------------------------------------------------------
     Inputs:
      -@problem:    a MOP to optimize
      -@algorithm:  the MOEA used to optimize the problem
      -@toStop:     stopping criteria method
    ----------------------------------------------------------------------------
     Summary:
      - Evolve a population for a problem using some algorithm.
      - Return the best generation of that evolution
    ----------------------------------------------------------------------------
     Outputs:
      - A record (statBox) of the best generation of evolution
    ----------------------------------------------------------------------------
    """
    
    # # # # # # # # # # #
    # 1) Initialization #
    # # # # # # # # # # #
    stoppingCriteria = False                             # Just a flag for stopping criteria
    statBox          = jmoo_stats_box(problem,algorithm) # Record keeping device
    gen              = 0                                 # Just a number to track generations
    numeval = 0
    
    # # # # # # # # # # # # # # # #
    # 2) Load Initial Population  #
    # # # # # # # # # # # # # # # #
    # Though this is not important I am sticking to NSGA3 paper
    # if algorithm.name == "NSGA3":
    #     print "-"*20 + "boom"
    #     jmoo_properties.PSI = jmoo_properties.max_generation[problem.name]
    #     jmoo_properties.MU = population_size[problem.name.split("_")[-1]]

    population = problem.loadInitialPopulation(jmoo_properties.MU)
    assert(len(population) == MU), "The population loaded from the file must be equal to MU"






    # # # # # # # # # # # # # # #
    # 3) Collect Initial Stats  #
    # # # # # # # # # # # # # # #
    statBox.update(population, 0, numeval, initial=True)

    # # # # # # # # # # # # # # #
    # 4) Generational Evolution #
    # # # # # # # # # # # # # # #
    
    while gen < jmoo_properties.PSI and stoppingCriteria is False:
        gen+= 1
        # # # # # # # # #
        # 4a) Selection #
        # # # # # # # # #

        problem.referencePoint = statBox.referencePoint
        selectees, evals = algorithm.selector(problem, population)
        numNewEvals = evals

        # # # # # # # # # #
        # 4b) Adjustment  #
        # # # # # # # # # #
        selectees, evals = algorithm.adjustor(problem, selectees)
        numNewEvals += evals


        
        # # # # # # # # # # #
        # 4c) Recombination #
        # # # # # # # # # # #

        population, evals = algorithm.recombiner(problem, population, selectees, MU)
        numNewEvals += evals
        assert(len(population) == MU), "Length of the population should be equal to MU"
        # # # # # # # # # # #
        # 4d) Collect Stats #
        # # # # # # # # # # #
        statBox.update(population, gen, numNewEvals)
        
            
        # # # # # # # # # # # # # # # # # #
        # 4e) Evaluate Stopping Criteria  #
        # # # # # # # # # # # # # # # # # #
        stoppingCriteria = toStop(statBox)
        # stoppingCriteria = False

        assert(len(statBox.box[-1].population) == MU), "Length in the statBox should be equal to MU"

    return statBox
