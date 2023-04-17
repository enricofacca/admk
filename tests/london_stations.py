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
    print(nodes_list)
    n_nodes = np.amax(nodes_list)
    print(np.amin(nodes_list),n_nodes)
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
    
    # get the list of nodes involved in the transfer
    transfer_nodes = get_transfer_nodes('NetworkData/rods_station_am_matrix_nodes.csv')

    # create rhs term from a given station, say, the first one
    rhs = get_tranfert('NetworkData/rods_station_am_matrix_nodes.csv', transfer_nodes[0])
    forcing = np.zeros(len(nodes_list))
    # balance the mass
    forcing[node2ind[transfer_nodes]] = rhs
    mass = forcing.sum()
    forcing[node2ind[transfer_nodes[0]]] = -mass

    print('f', forcing.size)
    print('w', weight.size)
    print('inc',topol_index.shape)
   

    # Init. graph problem
    graph=Graph(topol_index)
    print('graph size',graph.n_edges,graph.n_nodes)
    
    # Init. signed incidence matrix
    incidence_matrix = graph.signed_incidence_matrix()
    incidence_matrix_transpose = incidence_matrix.transpose()
    
    # Init problem (same graph)
    print(incidence_matrix_transpose.size)
    problem = MinNorm(incidence_matrix_transpose, weight)

    # set problem inputs (forcing loads, powers, etc) and check 
    problem = problem.set_inputs(forcing, 1.0)
    consistency = problem.check_inputs()
    
    # Init container for transport problem solution with
    # solution.tdens=edge conductivity
    # solution.pot=potential
    # solution.flux=conductivity * potential gradient
    solution = TdensPotentialVelocity(graph.n_edges,graph.n_nodes)

    # Init solver
    admk = AdmkSolver()

    # Init solver controls
    ctrl = AdmkControls()
    
    # mehtod and max_iter
    ctrl.time_discretization_method = 'explicit_tdens'
    ctrl.max_iter = 1000
    
    # deltat controls
    ctrl.deltat_control = 'expanding'
    ctrl.deltat = 1e-1
    ctrl.min_deltat = 1e-2
    ctrl.max_deltat = 5e-1
    
    # verbosity
    ctrl.verbose = verbose
    
    # solve
    ierr = admk.solve(problem, solution, ctrl)

    print(ierr)

    # check if convergence is achieved
    return 0

if __name__ == "__main__":
    sys.exit(test_main(2))
