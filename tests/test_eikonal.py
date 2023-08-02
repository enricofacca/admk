# import Solver 
from copy import deepcopy as cp

import sys
import os

# Import Admk solver for graphs
sys.path.append('../src/')
from admk import Graph
from admk import MinNorm
from admk import TdensPotentialVelocity
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
    problem = MinNorm(E_matrix, weight)
    problem.set_inputs(rhs,1.0)# 1.0 become is Optimal Transport
    consistency = problem.check_inputs()
    
    # Init tdpot varaibles (mu,u) in the manuscript
    tdpot = TdensPotentialVelocity(graph.n_edges,graph.n_nodes)

    # Init solver
    admk = AdmkSolver(problem)

    # Set solver controls
    ctrl = AdmkControls()
    ctrl.deltat = 1.0
    ctrl.deltat_control = 1
    ctrl.min_deltat = 1e-1
    ctrl.max_deltat = 1e4
    ctrl.verbose = verbose
    ctrl.time_discretization_method = 'explicit_gfvar'
    ctrl.time_discretization_method = 'implicit_gfvar'
    
    # run solver (tdpot is changed in place)
    admk.solve(problem, tdpot, ctrl)

    # get mu, pot, vel
    pot, mu, vel = tdpot.get_otp_solution(problem)

    # shift potential to get zero at the root node
    pot -= pot[0] 

    # print results
    if verbose > 0:
        print('The potential pot in this case is minus the distance from the first node')
        print(f'{pot=}')
        print('The transport density mu count the mass passing throught each edge')
        print(f'{mu=}')
        print('The velocity=mu gradient (pot) describes the flux of mass')
        print(f'{vel=}')

    
    return 0

if __name__ == "__main__":
    sys.exit(test_main(1))
