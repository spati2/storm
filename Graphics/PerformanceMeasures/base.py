from __future__ import division
__author__ = 'george'
import sys,os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

def get_archive_data(problems, algorithms):

    from DataFrame import ProblemFrame
    data = [ProblemFrame(problem, algorithms) for problem in problems]
    print data[-1].get_frontiers()
    print data[-1].get_extreme_points()
    import pdb
    pdb.set_trace()