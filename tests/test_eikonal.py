# import Solver 
from copy import deepcopy as cp

import sys
import os

# Import Admk solver for graphs
sys.path.append('../src/')
from admk import Graph
from admk import MinNorm
from admk import AdmkControls
from admk import AdmkSolver


import numpy as np
from scipy.linalg import norm 
#from scipy import linalg
import time as cputiming
import os

    
def test_main(verbose=0):
    # set the topology of the graph
    topol=np.array([[1,2],
           [1,4],
           [3,4],
           [2,3],
           [1,3],
           [2,4],
           [4,5]])
    # 0-index
    topol-=1

    # set the weight of the weight
    weight=np.ones(7)

    # aasign initial and final configuration of Optimal Transport
    source = np.zeros(5)
    target = np.zeros(5)
    source[0]=1
    target[1:]=0.25

    # this must sum to zero
    rhs = source - target 
    

    # Init. graph problem and matrix 
    graph=Graph(topol.T);
    
    # Init. signed incidence matrix
    incidence_matrix = graph.signed_incidence_matrix()
    E_matrix=incidence_matrix.transpose()

    # Init problem inputs (rhs, q, exponent)
    problem = MinNorm(E_matrix,  rhs, q_exponent=1.0, weight=weight)
    
    
    # Init solver
    admk = AdmkSolver(problem)
    tdpot = admk.initial_solution()

    # Init solver controls to default
    ctrl = AdmkControls(tol_optimization = 1e-3,
                        tol_constraint = 1e-8,
                        method='explicit_tdens',
                        max_iter = 1000,
                        max_restart = 5,
                        verbose=0,
                        log=0,
                        log_file='admk.log')
    
    # Tuning methods
    ctrl.set_method_ctrl('deltat',{
        'control':'expanding',
        'initial': 1e-1,
        'min': 1e-2,
        'max': 5e-1,
        'expansion': 1.05,
        'contraction': 2.0
    }
    )
    # linear solver
    # matrix is singualr. we need to relax it with + relax*identity 
    ctrl.set_method_ctrl('relax_Laplacian',1e-10)
    ctrl.set_method_ctrl(['ksp','type'],'preonly')
    ctrl.set_method_ctrl(['pc','type'],'lu')
    ctrl.set_method_ctrl(['pc','factor_drop_tolerance','dt'],1e-4)

    
    
    # run solver (tdpot is changed in place)
    admk.solve(problem, tdpot, ctrl)

    # get mu, pot, vel
    u, mu, vel = admk.get_otp_solution(tdpot)

    # shift potential to get zero at the root node
    u -= u[0] 

    # print results
    if verbose > 0:
        print('The potential u in this case is minus the distance from the first node')
        print(f'{u=}')
        print('The transport density mu count the mass passing throught each edge')
        print(f'{mu=}')
        print('The velocity=mu gradient (pot) describes the flux of mass')
        print(f'{vel=}')

    
    return 0

if __name__ == "__main__":
    sys.exit(test_main(1))
