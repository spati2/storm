
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
"Command line interface."

from jmoo_properties import *
# from jmoo_defect_prediction_properties import *
from jmoo_core import *
from jmoo_version_space import version_space_search
from jmoo_version_space import build_version_space_chart
   
"""
------------
Quick Notes
------------

The Main Interface of JMOO.  Run this python script from the commmand line with or without command line arguments.

- The Core behind JMOO is to perform 'tests' and then prepare 'reports' about those tests.
- - A 'test' is a collection of problems and algorithms.  Each algorithm is tested against each problem.

- To define tests (and many other properties), please see jmoo_properties.py
- To define reports, see below in this python script.
"""
   

db = open('what.txt', 'w')
                                

# Process command line arguments.  These modify properties of JMOO.
tag = ""
reportOnly = False
chartOnly = False
binsOnly = False
noReports = True
dfreportOnly = False
dfrankOnly = False
dfchartOnly = False
versionspace = False
vschart = False
df = False
optimization = False

for i,arg in enumerate(sys.argv):
    if arg == "-n" or arg == "-N":
        repeats = sys.argv[i+1]
    if arg == "-NEW" or arg == "-new" or arg == "-New":
        build_new_pop = True
    if arg == "-MU" or arg == "-mu" or arg == "-Mu":
        MU = sys.argv[i+1]
    if arg == "-tag" or arg == "-Tag" or arg == "-TAG":
        tag = sys.argv[i+1]
    if arg == "-reportOnly":
        reportOnly = True
    if arg == "-chartOnly":
        reportOnly = True
        chartOnly = True
    if arg == "-binsOnly":
        binsOnly = True
        reportOnly = True

    if arg == "-defect":
        df = True
    if arg == "-opti":
        optimization = True

    if arg == "-defectreport":
        dfreportOnly = True
    if arg == "-defectchart":
        dfchartOnly = True
    if arg == "-defectranking":
        dfrankOnly = True
    if arg == "-versionspace":
        versionspace = True
    if arg == "-vschart":
        vschart = True




        
print "VSCHART: ", vschart

# Build new initial populations if suggested.  Before tests can be performed, a problem requires an initial dataset.
if build_new_pop:
    # Thought this if condition is not important but I am just sticking to the NSGA paper
    if "NSGA3" not in [algorithm.name for algorithm in algorithms]:
        for problem in problems:
            initialPopulation(problem, MU)
    else:
        # This was added specifically for NSGA-III
        for problem in problems:
            # Only for nsga3 experiments
            # initialPopulation(problem, population_size[problem.name.split("_")[-1]])
            initialPopulation(problem, MU)

        
# Wrap the tests in the jmoo core framework
if not vschart:
    tests = jmoo_test(problems, algorithms)

# Define the reports
if chartOnly == True: reports = [jmoo_chart_report(tests)]
elif binsOnly: reports = [jmoo_decision_report(tests)]

elif reportOnly: reports = [jmoo_stats_report(tests)]
elif noReports: reports = []
else: reports = [jmoo_stats_report(tests), jmoo_decision_report(tests), jmoo_chart_report(tests)]

if dfreportOnly is True:
    reports = [jmoo_df_report("stats")]
if dfchartOnly is True:
    reports = [jmoo_df_report("charts")]
if dfrankOnly is True:
    reports = [jmoo_df_report("ranking", tests)]


# Associate core with tests and reports
if not vschart:
    core = JMOO(tests, reports)


print "here"

# Perform the tests
if df and not reportOnly and not dfreportOnly and not dfchartOnly and not dfrankOnly and not versionspace and not vschart:
    core.doDefectPrediction()

if not reportOnly and optimization:
    core.doTests()

if versionspace and not reportOnly and not dfreportOnly and not dfchartOnly and not dfrankOnly and not vschart:
    version_space_searcsh(core)

if vschart and not reportOnly and not dfreportOnly and not dfchartOnly and not dfrankOnly and not versionspace:
    print "there"
    build_version_space_chart()

# Prepare the reports
if not vschart:
    core.doReports(tag)