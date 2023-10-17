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

import pandas as pd

def get_topology(filename):
    df = pd.read_csv(filename, sep=',')
    topol = np.array([df['source'].values, df['target'].values])
    return topol, df['length'].values

def get_transfer_nodes(filename):
    """
    Get the list of nodes where the transfer is non-zero
    """
    return pd.read_csv(filename, sep=',').iloc[:,0].values


def get_tranfert(filename,station_id):
    """
    Get the transfer from a station to all the nodes
    """
    df = pd.read_csv(filename, sep=',')
    return df[str(station_id)].fillna(0).values

def node2index(nodes_list):
    """
    Given a list of nodes, return
    node2index: node2index[node_id] = i and -1 if node_id is not in nodes_list
    """
    n_nodes = np.amax(nodes_list)
    node2index = - np.ones(n_nodes+1, dtype=int)*2*n_nodes
    for i, node in enumerate(nodes_list):
        node2index[node] = i
    return node2index

def test_main(verbose=0):
    # read topology
    topol, weight = get_topology('NetworkData/edge_attributes.csv')

    # get nodes list
    nodes_list = np.unique(topol)
    
    # build a map from node to index in nodes_list
    node2ind = node2index(nodes_list)
    
    # convert topology to be index based
    topol_index = np.array([node2ind[topol[0]], node2ind[topol[1]]])

    # Init. graph problem
    graph=Graph(topol_index)
    print('graph n_edges',graph.n_edges,'n_nodes',graph.n_nodes)
    


    # get the list of nodes involved in the transfer
    transfer_nodes = get_transfer_nodes('NetworkData/rods_station_am_matrix_nodes.csv')

    # create rhs term from a given station, say, the first one
    forcings = []
    for root_node in transfer_nodes[[0,1]]:
        rhs = get_tranfert('NetworkData/rods_station_am_matrix_nodes.csv', root_node)
        forcing = np.zeros(len(nodes_list))
        # balance the mass
        forcing[node2ind[transfer_nodes]] = rhs
        mass = forcing.sum()
        forcing[node2ind[root_node]] = -mass
        forcings.append(forcing)


    forcing = np.concatenate(forcings)
   
   
    # Init. signed incidence matrix
    incidence_matrix = graph.signed_incidence_matrix()
    incidence_matrix_transpose = incidence_matrix.transpose()
    
    # Init problem (same graph)
    print(incidence_matrix_transpose.size)
    
    # TIME VARYING FORCING: both defintions should work
    def time_varying_forcing(t):
        return forcing*np.sin(2*np.pi*t)
    # time_varying_forcing = forcing
    problem = MinNorm(incidence_matrix_transpose,
                      rhs_of_time=time_varying_forcing,
                      q_exponent=1.0,
                      weight=weight)


    # Init solver and admk solution
    admk = AdmkSolver(problem)
    solution = admk.initial_solution()
    

    # Init solver controls to default
    ctrl = AdmkControls(tol_optimization = 1e-3,
                        tol_constraint = 1e-8,
                        method='explicit_tdens',
                        max_iter = 1000,
                        max_restart = 5,
                        verbose=1,
                        log=0,
                        log_file='admk.log')
    
    # deltat controls
    ctrl.set_method_ctrl('deltat',{
        'control':'fixed',
        'initial': 5e-1,
        'min': 1e-2,
        'max': 5e-1,
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

    
    ctrl.approach_linear_solver = 'direct'
    # large fillin and low drop tolerance its like having a direct solver
    ctrl.outer_prec_fillin = 30
    ctrl.outer_prec_drop_tolerance = 1e-5

    
    

    # solve
    ierr = admk.solve(problem, solution, ctrl)
    print('ierr=',ierr,admk.ierr_dictionary(ierr))

    # check if convergence is achieved
    return 0

if __name__ == "__main__":
    sys.exit(test_main(2))
