# import Solver 
from rcis import Solver
from copy import deepcopy as cp

import sys
# Import I/O for timedata
try:
    sys.path.append('/home/fh/srcs/globals/python/timedata/')
    import timedata as td
except:
    print("Global repo non found")

# Import geometry tools
sys.path.append('/home/fh/srcs/geometry/python/')
import meshtools as mt

# Import geometry tools
sys.path.append('/home/fh/srcs/geometry/python/')
import meshtools as mt

# Import Admk solver for graphs
sys.path.append('../src/')
from admk import Graph
from admk import MinNorm
from admk import TdensPotentialVelocity
from admk import AdmkControls
from admk import AdmkSolver

# import Iterative solver
from rcis import CycleControls


import numpy as np
from scipy.linalg import norm 
#from scipy import linalg
import time as cputiming
import os


# read graph and coordinate graphs
inputs_folder='inputs/'
test=sys.argv[1]
coord,topol,flags = mt.read_grid(inputs_folder+'eikonal_'+test+'/grid_'+test+'_graph.dat')

# read rhs
rhs=td.read_steady_timedata(inputs_folder+'eikonal_'+test+'/eik_'+test+'_rhs.dat').flatten()
print('rhs',len(rhs))
isource=np.argmax(rhs)
print('Source node',isource, 'located at:', coord[isource,:])

# read weight
weigth=td.read_steady_timedata(inputs_folder+'eikonal_'+test+'/weight_'+test+'.dat').flatten()

# read optimal solution 
optpot=td.read_steady_timedata(inputs_folder+'eikonal_'+test+'/eik_'+test+'_optpot.dat').flatten()


# Init. graph problem
graph=Graph(topol);

# Init. signed incidence matrix
incidence_matrix=graph.signed_incidence_matrix()
incidence_matrix_transpose=incidence_matrix.transpose()

# Init problem inputs (rhs, q, exponent)
problem=MinNorm(incidence_matrix_transpose,weigth)
problem=problem.set_inputs(rhs,1.0)
consistency=problem.check_inputs()

# Init tdpot varaibles (tdpot)
tdpot=TdensPotentialVelocity(graph.n_edge,graph.n_node)

# Init solver
ctrl = AdmkControls()
admk = AdmkSolver(ctrl)
admk.ctrl.deltat_control = 1
admk.ctrl.min_deltat = 1e-1
admk.ctrl.max_deltat = 1e4

admk.ctrl.verbose=2

admk.ctrl.time_discretization_method = 'explicit_gfvar'
admk.ctrl.time_discretization_method = 'implicit_gfvar'

# Create a class to problem info
class InfoProblem():
    def __init__(self):
        self.wasserstain=0.0
info_admk=InfoProblem()
        
# init update cycle controls
time_controls=CycleControls(1000)



# Start main cycle
[tdpot,ierr,admk] = admk.syncronize(problem,tdpot)
tdpot_old=cp(tdpot)
hystory=[]
while(time_controls.flag >=0):
    # call reverse communication
    flags, sol, admk = (
        time_controls.reverse_communication(admk, problem, tdpot)
    )
    
    ###################################################
    # select action according to flag.flag and flag.info
    ######################################################


    if (flags.flag == 1):
        # Here the user evalutes if convergence is achieved
        var = (
            norm(tdpot.tdens - tdpot_old.tdens) /
            (norm(tdpot.tdens) * admk.ctrl.deltat)
        )

        print(' ')
        print('var=',var)
        if (var < 1e-4):
            flags.flag = -1
            flags.info = 0
            
    if (flags.flag == 2):
        """ Study state system """
        print(' ')
        print('iter=',time_controls.iterations,'time=',tdpot.time)
        #print('{:.2E}'.format(min(tdpot.tdens))+'<=TDENS<='+'{:.2E}'.format(max(tdpot.tdens)))
        #print('{:.2E}'.format(min(tdpot.pot))+'<=POT  <='+'{:.2E}'.format(max(tdpot.pot)))
        grad=problem.potential_gradient(tdpot.pot)
        print('{:.2E}'.format(min(abs(grad)))+'<=|GRAD| <='+'{:.2E}'.format(max(abs(grad))))

        #print('{:.2E}'.format(min(optpot))+'<=POT <='+'{:.2E}'.format(max(optpot)))

        root = np.unravel_index(np.argmin(optpot, axis=None), optpot.shape)
        temp=tdpot.pot+optpot
        temp=temp-tdpot.pot[root]
        
        #print(tdpot.pot[root])
        print('err_pot',np.linalg.norm(temp)/np.linalg.norm(optpot))
        hystory.append([sol.time,cp(admk.info)])
        
        print(len(tdpot.tdens),len(topol))
        #td.write(tdpot.time, point_data={"pot": tdpot.pot})

    if (flags.flag == 3):
        """ Here user have to set solver controls for next update """
        admk.ctrl.set_before_iteration()
        print('new_deltat=',admk.ctrl.deltat)
        
        # copy data before update
        tdpot_old=cp(tdpot)
            
    if (flags.flag == 4): 
        """ Reset controls after a failure """
        admk.ctrl.reset_after_failure(flags.ierr)
