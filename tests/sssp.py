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

# import Iterative solver
sys.path.append(os.path.abspath("../../rcis/src/"))
from rcis import CycleControls


import numpy as np
from scipy.linalg import norm 
#from scipy import linalg
import time as cputiming
import os


#
# Three subroutine (read_grid, write_steady_data,
# read_steady_timedata)for data reading and writing
# 
def read_grid(filename):
    f_grid = open(filename, 'r')
    input_lines = f_grid.readlines()
    nnode = int(input_lines[0].split()[0])
    ntria = int(input_lines[1].split()[0])
    try:
        nnodeincell= int(input_lines[1].split()[1])
    except:
        nnodeincell=3
    coord=np.zeros([nnode,3])
    triang=np.zeros([ntria,nnodeincell],dtype=int)
    flags=np.zeros(ntria,dtype=int)

    inode=0
    try:
        coord[inode][:]=[float(w) for w in input_lines[2].split()[0:2]]
        ncoord=2
    except:
        coord[inode][:]=[float(w) for w in input_lines[2].split()[0:3]]
        ncoord=3

    inode=1
    for line in input_lines[3:2+nnode]:
        coord[inode][:]=[float(w) for w in line.split()[0:ncoord]]
        inode=inode+1
    itria=0
    for line in input_lines[2+nnode:]:
        triang[itria][:]=[int(w)-1 for w in line.split()[0:nnodeincell]]
        flags[itria]=line.split()[nnodeincell]
        itria=itria+1
    return coord, triang, flags;

#
# Fucntion for a fast writing to file of timedata .
# For steady-state data only 
#
def write_steady_timedata(filename,data):
    file_out=open(str(filename), 'w')
    try:
        dimdata=data.shape[1]
        ndata=data.shape[0]
    except:
        dimdata=1
        ndata=len(data)

    file_out.write(
	str(dimdata) + " " + 
	str(ndata) + " !  dim ndata" 
	+"\n")
    nrm_data=np.zeros(ndata)
    for i in range(ndata):
        nrm_data[i]=np.linalg.norm(data[:][i])
        nnz=(abs(nrm_data) != 0.0).sum()
	
    file_out.write("time    -1.0e-30 \n")
    file_out.write(str(ndata)+" \n")
    try:
        for i in range(ndata):
            file_out.write(str(i+1)+" " + 
			   " ".join(map(str,data[:][i])) +"\n")
    except:
        for i in range(ndata):
            file_out.write(str(i+1)+" " +
                           str(data[i]) +"\n")
            
    file_out.write("time  1.0e+30 \n")
    file_out.close()
    return;

#
# Function for a fast reading from file of timedata .
# For steady-state data only 
#
def read_steady_timedata(filename):
    fin=open(str(filename), 'r')
    lines = fin.readlines()
    dimdata=int(lines[0].split()[0])
    ndata=int(lines[0].split()[1])
    data=np.zeros([ndata,dimdata])   
    ninputs= int(lines[2].split()[0])
    for i in range(ninputs):
        line=lines[3+i]
        idata=int(line.split()[0])
        data[idata-1,:]=[float(w) for w in line.split()[1:]]
        fin.close()
    return data;
    
def test_main(test='0032',verbose=0):
    # read graph and coordinate graphs
    inputs_folder='inputs/'
    coord,topol,flags = read_grid(inputs_folder+'eikonal_'+test+'/grid_'+test+'_graph.dat')

    # read rhs
    rhs=read_steady_timedata(inputs_folder+'eikonal_'+test+'/eik_'+test+'_rhs.dat').flatten()
    isource=np.argmax(rhs)
    
    # read weight
    weigth=read_steady_timedata(inputs_folder+'eikonal_'+test+'/weight_'+test+'.dat').flatten()
    
    # read optimal solution 
    optpot=read_steady_timedata(inputs_folder+'eikonal_'+test+'/eik_'+test+'_optpot.dat').flatten()


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

    admk.ctrl.verbose=verbose
    
    admk.ctrl.time_discretization_method = 'explicit_gfvar'
    admk.ctrl.time_discretization_method = 'implicit_gfvar'
    
    # Create a class to store problem info
    class InfoProblem():
        def __init__(self):
            self.wasserstain=0.0
            info_admk=InfoProblem()
	    
    # init update cycle controls
    flags=CycleControls(1000)

    
    # Start main cycle
    admk.syncronize(problem,tdpot,flags.ierr)
    tdpot_old=cp(tdpot)
    hystory=[]
    while(flags.flag >=0):
        #
        # Call reverse communication and
        # select action according to flag.flag and flag.info
        #
        flags.reverse_communication()

        if (flags.flag == 1):
            # update system
            admk.iterate(problem, tdpot, flags.ierr)

        if (flags.flag == 2):
            # Here the user evalutes if convergence is achieved
            var = (
                norm(tdpot.tdens - tdpot_old.tdens) /
                (norm(tdpot.tdens) * admk.ctrl.deltat)
            )
            if (verbose > 0):
                print(' ')
                print('var=',var)

            if (var < 1e-4):
                flags.flag = -1
                flags.info = 0
        if (flags.flag == 3):
            """ Study state system """
            grad=problem.potential_gradient(tdpot.pot)
            root = np.unravel_index(np.argmin(optpot, axis=None), optpot.shape)
            temp=tdpot.pot+optpot
            temp=temp-tdpot.pot[root]
            if (verbose > 1):
                print(' ')
                print('iter=',flags.iterations,'time=',tdpot.time)
                print('{:.2E}'.format(min(abs(grad))) +
                      '<=|GRAD| <=' +
                      '{:.2E}'.format(max(abs(grad))))
                print('err_pot',np.linalg.norm(temp) / np.linalg.norm(optpot))
            hystory.append([tdpot.time,cp(admk.info)])
        if (flags.flag == 4):
            """ Reset controls after a failure """
            admk.ctrl.reset_after_failure(flags.ierr)

        if (flags.flag == 5):
            """ Here user have to set solver controls for next update """
            admk.ctrl.set_before_iteration()

            # copy data before update
            tdpot_old=cp(tdpot)


    # check if convergence is achieved
    root = np.unravel_index(np.argmin(optpot, axis=None), optpot.shape)
    temp=tdpot.pot+optpot
    temp=temp-tdpot.pot[root]
    assert (norm(temp)<1e-3)
    return 0

if __name__ == "__main__":
    sys.exit(test_main())
