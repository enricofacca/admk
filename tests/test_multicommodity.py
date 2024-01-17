# import Solver 
from copy import deepcopy

import sys
import os

# Import Admk solver for graphs
sys.path.append('../src/')
from admk import Graph
from admk import MinNorm
from admk import AdmkControls
from admk import AdmkSolver
from admk import AdmkSolution

import numpy as np
from scipy.linalg import norm 
#from scipy import linalg
import time as cputiming
import os


def test_main(verbose=0):
    # set the topology of the graph
    #
    # 4-3-2
    #   |/|
    #   0-1
    #
    topol=np.array([
        [0,1],
        [0,2],
        [0,3],
        [2,3],
        [1,2],
        [3,4]])
    print(len(topol))
    weight = np.ones(len(topol))

    # Init. graph problem, incidence matrix and its transpose
    graph = Graph(topol.transpose())
    incidence_matrix = graph.signed_incidence_matrix()
    incidence_matrix_transpose = incidence_matrix.transpose()
    print(incidence_matrix.shape)

    # 2 forcing terms
    forcing_1 = np.ones(graph.n_nodes)
    forcing_1[0] = -sum(forcing_1[1:])
    print(forcing_1[0],sum(forcing_1[1:]))
    forcing_2 = np.ones(graph.n_nodes)
    forcing_2[-1] = -sum(forcing_2[:-1])
    forcing = np.concatenate([forcing_1,forcing_2])
    print(forcing_2)
    
    # TIME VARYING FORCING: both defintions should work
    def timeforcing(t):
        return forcing#*np.sin(np.pi/2+20*np.pi*t)

    # time_varying_forcing = forcing
    problem = MinNorm(incidence_matrix_transpose,
                      rhs_of_time=forcing,
                      q_exponent=1.0,
                      weight=weight)
 

    # Init solver controls class
    # At initialization we set the main controls
    # (tolerances, maximum number of iterations, verbosity)
    # All controls can be set afterward using set method
    ctrl = AdmkControls(tol_optimization = 1e-3,
                        tol_constraint = 1e-8,
                        method='explicit_tdens',
                        max_iter = 200,
                        max_restart = 5,
                        verbose=1,
                        log=0,
                        log_file='admk.log')
    
    # deltat controls
    ctrl.set_method_ctrl('deltat',{
        'control':'adaptive2',
        'initial': 1e-2,
        'min': 1e-2,
        'max': 1e-1,
        'expansion': 1.05,
        'contraction': 2.0
    }
    )
    # linear solver
    # matrix is singualr. we need to relax it with + relax*identity 
    ctrl.set_method_ctrl('relax_Laplacian',1e-10)
    ctrl.set_method_ctrl(['ksp','type'],'cg')
    ctrl.set_method_ctrl(['pc','type'],'icc')
    ctrl.set_method_ctrl(['pc','factor_drop_tolerance','dt'],1e-4)
    ctrl.set_method_ctrl(['pc','type'],'hypre')

    # Init solver
    admk = AdmkSolver(problem, ctrl)
    # It is still positible to set ctrl "manually" 
    admk.ctrl.set_method_ctrl(['pc','type'],'hypre')
    # Here we also set the initial guess
    sol0 = deepcopy(admk.solution) # first option
    sol0 = AdmkSolution(problem) # second option
    pot0, tdens0 = sol0.subfunctions()
    tdens0[:]=2.0
    admk.set_initial_guess(sol0)
    
    ierr = admk.solve()
    print('ierr=',ierr,admk.ierr_dictionary(ierr))
    pot, tdens, vel = admk.solution.get_problem_solution()

    pot0 = admk.solution.get_subpotential(0)
    pot1 = admk.solution.get_subpotential(1)
    print('pot0=',pot0)
    print('pot1=',pot1)
    print('tdens=',tdens)
    print('vel=',vel)
    
    # check if convergence is achieved
    return 0

if __name__ == "__main__":
    sys.exit(test_main(2))
