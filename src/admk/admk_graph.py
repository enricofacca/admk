# import Solver 
from copy import deepcopy as cp

import sys

import numpy as np
import scipy as sp
import scipy.sparse.linalg as splinalg
from scipy.linalg import norm 
import time as cputiming
import os
from .linear_solvers import implicit_block_diag   


from .linear_solvers import info_ksp
from .linear_solvers import get_info
from .linear_solvers import KSPReasons
from petsc4py import PETSc
import multiprocessing

from multiprocessing import RawArray, Array
        
class MinNorm:
    """
    This class contains the inputs of problem GraphDmk 
    min |v|^{q>=1}_w : A v = rhs 
    with 
    - |v|^{q>=1}_w = \sum_{i} |v_i|^q* w_i
      where w is a strictly positive vector
    - A signed incidence matrix of graph G
      rows number = number of nodes
      columns number = number of edges    
    - rhs_of_time = right-hand side. it can be a function of time
    - q_exponent = exponent of the norm
    - weight = weight in the norm
    - initial_time = initial rhs in case of time varying
    """
    def __init__(self, matrix,
                    rhs_of_time, 
                    q_exponent=1.0, 
                    weight=None,
                    initial_time=0.0):
        """
        Constructor of problem setup
        """
        self.matrix = matrix
        self.n_row = matrix.shape[0]
        self.n_col = matrix.shape[1]
        
        self.matrixT = self.matrix.transpose()



        # edge weight
        if (weight is None):
            weight = np.ones(self.n_col)
        self.weight = weight
        self.inv_weight = 1.0/weight
        
        matvec_grad = lambda x: self.inv_weight * self.matrixT.dot(x)
        self.gradient = splinalg.LinearOperator((self.n_col,self.n_row),matvec_grad)    

        
        
        # set inputs 
        self.time_varying_inputs = False
        # set forcing term
        if callable(rhs_of_time):
            self.rhs_of_time = rhs_of_time
            self.time_varying_inputs = True
        else:
            # define a constant in time rhs
            self.rhs_of_time = lambda t: rhs_of_time

        self.initial_time = initial_time
            
        # We set the rhs at the initial time
        # to get the number of rhs and fix the rhs 
        # if fix in time
        self.rhs = self.rhs_of_time(initial_time)
        print('rhs',np.linalg.norm(self.rhs))
        if (len(self.rhs) % self.n_row != 0):
            myError = ValueError(f'Passed rhs.shape[0]={len(self.rhs):%d} is not a '+
                                 'multiple of {self.n_row:%d} = nrow')
            raise myError
        else:
            self.n_rhs = len(self.rhs) // self.n_row


        if callable(q_exponent):
            self.q_exponent_of_time = q_exponent
            self.time_varying_inputs = True
        else:
            # define a constant in time exponent
            self.q_exponent_of_time = lambda t: q_exponent
        # We set the exponent at the initial time
        self.q_exponent = self.q_exponent_of_time(initial_time)

        ierr = self.check_inputs()
        if (ierr != 0):
            print('Error in inputs at time t=',initial_time)

        if self.n_rhs != 1:
            self.grad = implicit_block_diag([self.gradient]*self.n_rhs)
            self.div = implicit_block_diag([self.matrix]*self.n_rhs)
        else:
            self.grad = self.gradient
            self.div = self.matrix

    def update_inputs(self,time):
        """
        Update inputs at time t
        """
        # nothing to if self.time_varying_inputs is False
        # see set_inputs procedure
        if self.time_varying_inputs:            
            # update and check the inputs 
            self.rhs = self.rhs_of_time(time)
            self.q_exponent = self.q_exponent_of_time(time)

            ierr = self.check_inputs()
            if (ierr != 0):
                raise ValueError('Error in inputs at time t=',time)
        
    
    def check_inputs(self):
        """
        Method to check problem inputs consistency
        """
        ierr=0
        for i in range(self.n_rhs):
            begin = i*self.n_row
            end = (i+1)*self.n_row
            balance = np.sum(self.rhs[begin:end])/np.linalg.norm(self.rhs[begin:end])
            if balance > 1e-11:
                print(f'Rhs{i:d} is not balanced {balance:.1E}')
                ierr=1
        return ierr

    def potential_gradient(self, pot):
        """
        Procedure to compute gradient of the potential
        grad=W^{-1} A^T pot

        Args:
        pot: real (nrow of A)-np.array with potential

        Returns:
        grad: real (ncol of A)-np.array with gradient
        """
        grad = self.inv_weight * self.matrixT.dot(pot)
        return grad
    
    def constraint_residual(self, vel):
        """
        Procedure to compute residual of the constraint
        res = A vel - rhs

        Args:
        vel: real (ncol of A)-np.array with velocity

        Returns:
        res: real (nrow of A)-np.array with residual
        """
        rhs_norm = np.linalg.norm(self.rhs)
        print(f'{rhs_norm=}')
        res = np.linalg.norm(self.div.dot(vel) - self.rhs) / rhs_norm
        return res

class Graph:
    """
    This class contains the inputs
    of problem GraphDmk. Namely
    min |v|^{q>=1} : A v = rhs
    with A signed incidence matrix of graph G
    """  
    def __init__(self, topol, weight=None):
        """
        Constructor from raw data

        Args:
        topol:  (2,n_edge) integer np.array with node conenctivity 
                The order define the orientation
        weight: (n_edge) real np.array with weigth associate to edge
                Default=1.0

        Returns:
        Initialize class GraphDmk
        """

        # member with edge number
        self.n_edges    = topol.shape[1]
        
        # member with nodes number
        self.n_nodes = np.amax(topol) + 1 - np.amin(topol)
        
        # graph topology
        self.topol = cp(topol)

        # edge weight
        if (weight is None):
            weight = np.ones(self.n_edges)
        self.weight = weight


    def signed_incidence_matrix(self):
        """
        Build signed incidence matrix
        
        Args:
        topol: (2,ndege)-integer np-array 1-based ordering
        
        Result:
        matrix : signed incidence matrix 
        """
        # build signed incidence matrix
        indptr  = np.zeros([2*self.n_edges]).astype(int) # rows
        indices = np.zeros([2*self.n_edges]).astype(int) # columns
        data    = np.zeros([2*self.n_edges])                # nonzeros
        for i in range(self.n_edges):
            indptr[2*i:2*i+2]  = int(i)
            indices[2*i] = int(self.topol[0,i])
            indices[2*i+1] = int(self.topol[1,i])
            data[2*i:2*i+2]    = [1.0,-1.0]
            #print(topol[i,:],indptr[2*i:2*i+2],indices[2*i:2*i+2],data[2*i:2*i+2])
        signed_incidence = sp.sparse.csr_matrix((data, (indptr,indices)),shape=(self.n_edges, self.n_nodes))
        return signed_incidence

    def save_time_series(self):
        import meshio
        
        points=coord
        cells=[("line", topol)]
        mesh = meshio.Mesh(
            points,
            cells,
        )
        mesh.write(
            "grid.xdmf",  # str, os.PathLike, or buffer/open file
            # file_format="vtk",  # optional if first argument is a path; inferred from extension
        )
        
        file_xdmf=meshio.xdmf.TimeSeriesWriter('grid2.xdmf')
        file_xdmf.__enter__()
        file_xdmf.write_points_cells(points, cells)
        
        file_xdmf.write_data(0.0, point_data={"pot": pot})
        file_xdmf.__exit__()


        filename='sol.xdmf'
        
        with meshio.xdmf.TimeSeriesWriter(filename) as writer:
            writer.write_points_cells(points, cells)
            #file_xdmf=meshio.xdmf.TimeSeriesWriter(filename,data_format='HDF')
            #file_xdmf.__enter__()
            #file_xdmf.write_points_cells(points, cells)

#
# following code is taken from firedrake
#           
def flatten_parameters(parameters, sep="_"):
    """Flatten a nested parameters dict, joining keys with sep.

    :arg parameters: a dict to flatten.
    :arg sep: separator of keys.

    Used to flatten parameter dictionaries with nested structure to a
    flat dict suitable to pass to PETSc.  For example:

    .. code-block:: python3

       flatten_parameters({"a": {"b": {"c": 4}, "d": 2}, "e": 1}, sep="_")
       => {"a_b_c": 4, "a_d": 2, "e": 1}

    If a "prefix" key already ends with the provided separator, then
    it is not used to concatenate the keys.  Hence:

    .. code-block:: python3

       flatten_parameters({"a_": {"b": {"c": 4}, "d": 2}, "e": 1}, sep="_")
       => {"a_b_c": 4, "a_d": 2, "e": 1}
       # rather than
       => {"a__b_c": 4, "a__d": 2, "e": 1}
    """
    new = type(parameters)()

    if not len(parameters):
        return new

    def flatten(parameters, *prefixes):
        """Iterate over nested dicts, yielding (*keys, value) pairs."""
        sentinel = object()
        try:
            option = sentinel
            for option, value in parameters.items():
                # Recurse into values to flatten any dicts.
                for pair in flatten(value, option, *prefixes):
                    yield pair
            # Make sure zero-length dicts come back.
            if option is sentinel:
                yield (prefixes, parameters)
        except AttributeError:
            # Non dict values are just returned.
            yield (prefixes, parameters)

    def munge(keys):
        """Ensure that each intermediate key in keys ends in sep.

        Also, reverse the list."""
        for key in reversed(keys[1:]):
            if len(key) and not key.endswith(sep):
                yield key + sep
            else:
                yield key
        else:
            yield keys[0]

    for keys, value in flatten(parameters):
        option = "".join(map(str, munge(keys)))
        if option in new:
            print("Ignoring duplicate option: %s (existing value %s, new value %s)",
                    option, new[option], value)
        new[option] = value
    return new


def nested_set(dic, keys, value, create_missing=True):
    d = dic
    for key in keys[:-1]:
        if key in d:
            d = d[key]
        elif create_missing:
            d = d.setdefault(key, {})
        else:
            return dic
    if keys[-1] in d or create_missing:
        d[keys[-1]] = value
    return dic

class AdmkControls:
    """
    Class with Admk Solver 
    """

    def __init__(self,
                 tol_optimization = 1e-3,
                 tol_constraint = 1e-8,
                 method='explicit_tdens',
                 max_iter = 200,
                 max_restart = 5,
                 method_ctrl = None,
                 verbose=0,
                 log=0,
                 log_file='admk.log'
    ):        
        """
        Set the controls of the Dmk solver
        """
        self.tol_opt = tol_optimization
        self.tol_constraint = tol_constraint
        self.max_iter = max_iter
        self.max_restart = max_restart
        self.method = method
        #: info on standard output
        self.verbose=verbose
        #: info on log file
        self.log=log
        self.log=log_file


        self.time_discretization_method = self.method
        # methods speficific controls
        if self.method == 'explicit_tdens':
            self.method_ctrl =  {
                # global controls
                'tdens_min' : 1e-8,
                # time stepping controls
                'deltat' : {'initial': 0.01,
                            'control': 'fixed',
                            'min':1e-4,
                            'max':1e0,
                            'expansion': 1.05,
                            'contraction': 2.0,
                },
                # linear solver controls 
                'ksp': {
                    'type':'cg',
                    'rtol': tol_constraint,
                    'norm_type': 'unpreconditioned',
                },
                'pc':{
                    'type': 'hypre',
                    # used if for ilu only 
                    'factor_drop_tolerance':{
                        'dt': 1e-4,
                        'maxrowcount': 30
                    }
                },
                'relax_Laplacian': 1e-10,
                # saving of evolution
                'save':{
                    'sol': {
                        'directory': './runs/',
                        # 'no','some','last'
                        'mode':'no',
                        'frequency': 10,
                    },
                    'matrices': {
                        'directory': './runs/matrices/',
                        'mode':'no',
                        'frequency': 10,
                    }
                }
            }
        self.deltat = self.method_ctrl['deltat']['initial']
        
        #: real: minimum newton step
        self.min_newton_step = 5e-2
        self.contraction_newton_step = 1.05
        self.min_C = 1e-6
        
        
        #: Fillin for incomplete factorization
        self.outer_prec_fillin=20
        #: Drop tolerance for incomplete factorization
        self.outer_prec_drop_tolerance=1e-4

        
    def set_method_ctrl(self, keys, value):
        if not isinstance(keys, list):
            keys=[keys]
        
        self.method_ctrl = nested_set(self.method_ctrl,keys,value)
        self.deltat = self.method_ctrl['deltat']['initial']
        #for key in keys[:-1]:
        #    self.method_ctrl = self.method_ctrl.setdefault(key, {})
        #self.method_ctrl[keys[-1]] = value
        
        
    def print_info(self, msg, priority, indent=0):
        """
	    Print messagge to stdout and to log 
        file according to priority passed
        """
        if (self.verbose > priority):
            if indent>0:
                msg='   '*indent+msg
            print(msg)
      

            
    def set_deltat(self, deltat, state, update, state_lower_bound=0):
        """
        Procedure to set new controls after a succesfull update
        """
        deltat_ctrl = self.method_ctrl['deltat']
        if (deltat_ctrl['control'] == 'fixed'):
            deltat = deltat
        elif (deltat_ctrl['control'] == 'expansive'):
            deltat *= deltat_ctrl['expansion']
            deltat = max( min( self.deltat, deltat_ctrl['max']),deltat_ctrl['min'])
        elif (deltat_ctrl['control'] == 'adaptive'):
            order_down = -1
            order_up = 1
            min_u = np.min(update)
            max_u = np.max(update)

            if min_u < 0:
                deltat_l = (10**order_down - 1 ) / min_u
            else:
                deltat_l = self.method_ctrl['deltat']['max']

            if max_u > 0:
                deltat_u = (10**order_up - 1 ) / max_u
            else:
                deltat_u = self.method_ctrl['deltat']['max']

                
            r_np = update
            if np.min(r_np) < 0:
                negative = np.where(r_np<0)
                hdown = (10**order_down-1) / r_np[negative]
                deltat_down = np.min(hdown)
            else:
                deltat_down =  self.method_ctrl['deltat']['max']

            if np.max(r_np) > 0:
                positive = np.where(r_np>0)
                hup = (10**order_up-1) / r_np[positive]
                deltat_up = np.min(hup)
            else:
                deltat_up = self.method_ctrl['deltat']['max']


            deltat = min(deltat_l,deltat_u)
            deltat = max(deltat,self.method_ctrl['deltat']['min'])
            deltat = min(deltat,self.method_ctrl['deltat']['max'])
            
        elif (deltat_ctrl['control'] == 'adaptive2'):
            u_min = min(update)
            if (u_min<0):
                negative_indeces  = np.where(update < 0)[0]
                step = np.min((state_lower_bound - state[negative_indeces]) / update[negative_indeces])
                if (step < deltat_ctrl['min']):
                    raise ValueError("delta from adaptive strategy={step:.1e} is smaller than {deltat_ctrl['min']=}")
            else:
                step=deltat_ctrl['max']
            deltat = min(step,deltat_ctrl['max'])
        else:
            raise ValueError("deltat['control'] not supported")
        return deltat

    def reset_after_failure(self,ierr):
        """
        Procedure to set new controls after a succesfull update
        """
        self.deltat = max( min( self.deltat /
                            self.expansion_deltat, self.max_deltat),
                           self.min_deltat)
        return self



def solve_with_petsc(stiff, rhs, pot, petsc_options):
    """
    Solve linear system with petsc
    """

    n_pot = stiff.shape[0]
    n_rhs = rhs.shape[0] // n_pot
    
    petsc_stiff = PETSc.Mat().createAIJ(size=stiff.shape,
                                        csr=(stiff.indptr, stiff.indices,
                                            stiff.data))
    
    petsc_pot = petsc_stiff.createVecLeft()
    petsc_rhs = petsc_stiff.createVecRight()


    problem_prefix = 'laplacian_solver_'
    ksp = PETSc.KSP().create()
    ksp.setOperators(petsc_stiff)
    ksp.setOptionsPrefix(problem_prefix)

    # copy from https://github.com/FEniCS/dolfinx/blob/230e027269c0b872c6649f053e734ed7f49b6962/python/dolfinx/fem/petsc.py#L618
    # https://github.com/FEniCS/dolfinx/fem/petsc.py
    opts = PETSc.Options()    
    opts.prefixPush(problem_prefix)
    for k, v in petsc_options.items():
        opts[k] = v
    opts.prefixPop()
    ksp.setConvergenceHistory()
    #ksp.pc.setReusePreconditioner(True) # not sure if this is needed
    ksp.setFromOptions()            
    petsc_stiff.setOptionsPrefix(problem_prefix)
    petsc_stiff.setFromOptions()
    petsc_pot.setOptionsPrefix(problem_prefix)
    
    petsc_rhs.setFromOptions()

        
    ierr = 0   
    iter = 0
    res = 0
    pres = 0
    for i_rhs in range(n_rhs):
        # convert to petsc
        #rhs_i = rhs[i_rhs*self.n_pot:(i_rhs+1)*self.n_pot]
        petsc_rhs.setArray(rhs[i_rhs*n_pot:(i_rhs+1)*n_pot])
        petsc_pot.setArray(pot[i_rhs*n_pot:(i_rhs+1)*n_pot])

        # solve
        ksp.solve(petsc_rhs, petsc_pot)

        # store info
        reason = ksp.getConvergedReason()
        last_pres = ksp.getResidualNorm()
        #print(f'{i_rhs=} {KSPReasons[reason]} {i_rhs=} {last_pres=}')
        if reason < 0:
            print (f'{KSPReasons[reason]=} {i_rhs=}')
            ierr = 1
            break
        else:
            last_iter = ksp.getIterationNumber()
            iter+=last_iter
            h = ksp.getConvergenceHistory()
            if len(h)>0:
                resvec = h[-(last_iter+1):]
                rhs_norm = petsc_rhs.norm()
                if rhs_norm > 0: 
                    res=max(res,resvec[-1]/rhs_norm)
            
            last_pres = ksp.getResidualNorm()
            pres = max(pres,last_pres)

            if (res > petsc_options['ksp_rtol']):
                print(f'{KSPReasons[reason]=} {i_rhs=} {res=} rhs={petsc_rhs.norm()} pot={petsc_pot.norm()}')

        # get solution
        pot_i = petsc_pot.getArray()
        pot[i_rhs * n_pot : (i_rhs+1) * n_pot] = pot_i

    return ierr, iter, res, pres        

def initpool(rhs, pot):
    global shared_rhs
    global shared_pot
    shared_rhs = rhs
    shared_pot = pot

def solve_portion(stiff, petsc_options, start, end):
    """
    Solver A pot_i = rhs_i for i in [start,end)
    """
    n_pot = stiff.shape[0]
    
    np_rhs = np.frombuffer(shared_rhs.get_obj(), dtype=np.float64)
    np_pot = np.frombuffer(shared_pot.get_obj(), dtype=np.float64)
    
    rhs_p = np_rhs[start * n_pot: end * n_pot]
    pot_p = np_pot[start * n_pot: end * n_pot]

    ierr, iter, res, pres = solve_with_petsc(stiff, rhs_p, pot_p, petsc_options)
    #print('in',pot_p[0:3])
    np_pot = np.frombuffer(shared_pot.get_obj())
    np.copyto(np_pot[start * n_pot: end * n_pot],pot_p)
    #print(np_potprese])

    return ierr, iter, res, pres


def solve_with_petsc_parallel(stiff, rhs, pot, petsc_options, NUM_WORKERS=None):
    if NUM_WORKERS is None:
        NUM_WORKERS = multiprocessing.cpu_count()
    n_pot = stiff.shape[0]
    n_rhs = len(rhs)// n_pot
    NUM_WORKERS = min(min(NUM_WORKERS, multiprocessing.cpu_count()),n_rhs)
    
    chunk_size = int(n_rhs / NUM_WORKERS) + 1 
    #print(f'{n_rhs=} {chunk_size=} {NUM_WORKERS=}')
    
    rhs_mp = multiprocessing.Array('d', n_pot * n_rhs, lock=True)
    rhs_np = np.frombuffer(rhs_mp.get_obj())
    np.copyto(rhs_np,rhs)
    
    pot_mp = multiprocessing.Array('d', n_pot * n_rhs, lock=True)
    pot_np = np.frombuffer(pot_mp.get_obj())
    np.copyto(pot_np,pot)

    # create shared memory with rhs and pot
    pool = multiprocessing.Pool(processes=NUM_WORKERS, initializer=initpool, initargs=(rhs_mp,pot_mp))

    result = []
    indeces = np.arange(n_rhs)
    for i in range(NUM_WORKERS):
        start = i*chunk_size 
        end = start + chunk_size 
        if i == NUM_WORKERS-1 :
            end = n_rhs
        result.append(pool.apply_async(solve_portion, args=(stiff, petsc_options, start, end)))
    
    
    infos = [r.get() for r in result]
    ierr = sum([ info[0] for info in infos])
    iter = sum([ info[1] for info in infos])
    res = max([ info[2] for info in infos])
    pres = max([ info[3] for info in infos])
    pool.close()
    pool.join()
    pot[:] = np.frombuffer(pot_mp.get_obj())
    
    return ierr, iter, res, pres


class AdmkSolver:
    """
    Solver class for problem
    min \|v\|_{w}^{q} A v = rhs
    with A signed incidence matrix of Graph   
    via Algebraic Dynamic Monge-Kantorovich.
    We find the long time solution of the
    dynamics 
    \dt \Tdens(t)=\Tdens(t) * | \Grad \Pot(\Tdens)|^2 -Tdens^{gamma}    
    """
    def __init__(self,problem):
        """
		Initialize solver with passed controls (or default)
        and initialize structure to store info on solver application
        """
		# init infos
        self.linear_solver_iterations = 0
        # non linear solver
        self.nonlinear_solver_iterations = 0
        self.nonlinear_solver_residum = 0.0

        self.n_pot = problem.n_row
        self.n_tdens = problem.n_col

        self.problem = problem

        self.time = problem.initial_time
    
    
    def initial_solution(self):
        sol = np.zeros(self.n_pot*self.problem.n_rhs+self.n_tdens)
        sol[-self.n_tdens:] = 1.0
        return sol
        

    def get_otp_solution(self, sol):
        n_tdens = self.n_tdens
        n_pot = self.n_pot
        pot, tdens = self.subfunctions(sol)
        vel = np.zeros(self.n_tdens * self.problem.n_rhs)
        for i in range(self.problem.n_rhs):
            vel[i*n_tdens:(i+1)*n_tdens] = tdens *  (self.problem.gradient.dot(pot[i*n_pot:(i+1)*n_pot]))
        return pot, tdens, vel
    

    def subfunctions(self,sol):
        """
        Split solution in pot, tdens component
        """
        pot = sol[:self.n_pot * self.problem.n_rhs]
        tdens = sol[-self.n_tdens:]
        return pot, tdens 

    
    def get_subpotential(self, solution, index):
        """
        Return the potential associated to index-th commodity
        """
        return sol.pot[index*self.n_pot:index*self.n_pot]

    
    def build_stiff(self, matrixA, conductivity):
        """
        Internal procedure to assembly stifness matrix 
        S(tdens)=A conductivity A^T
		
        Args:
         conductivity: non-negative real (ncol of A)-np.array with conductivities

        Returns:
		 stiff: Scipy sparse matrix
        """
        diagt = sp.sparse.diags(conductivity)
        stiff = matrixA.dot(diagt.dot(matrixA.transpose()))
        return stiff


    def syncronize(self, problem, tdpot, ctrl):
        """        
        Args:
         tdpot: Class with unkowns (tdens, pot in this case)
         problem: Class with inputs  (rhs, q_exponent)
         ctrl:  Class with controls
		
        Returns:
        tdpot : syncronized to fill contraint S(tdens) pot = rhs
        info  : control flag (=0 if everthing worked)
        """
        pot, tdens = self.subfunctions(tdpot)
		
        # assembly stiff
        msg = (f'{min(tdens):.2E}<=TDENS<={max(tdens):.2E}')
        ctrl.print_info(msg, 3)

        
        start_time = cputiming.time()
        conductivity = tdens * problem.inv_weight
        stiff = self.build_stiff(problem.matrix, conductivity)
        rhs = problem.rhs.copy()

        n_equ = len(rhs)
        msg = ('ASSEMBLY'+'{:.2f}'.format(-(start_time - cputiming.time()))) 
        ctrl.print_info(msg,3)
    
        #
        # solve linear system
        #        
        relax=ctrl.method_ctrl['relax_Laplacian']
        stiff += relax * sp.sparse.eye(self.n_pot) # matrix is singular

        # merge two linear algebra dict.
        # and flatten to match PETSC syntacts
        # TODO: {**a,**b} merges the dict.
        # I did not find how to get a sub
        petsc_options = (flatten_parameters({
            **{'ksp' :ctrl.method_ctrl.get('ksp')},
            **{
                'ksp_initial_guess_nonzero': True,
                #'ksp_monitor_true_residual': None,
            },
            **{'pc' : ctrl.method_ctrl.get('pc')}
        }))

        
        np = min(multiprocessing.cpu_count(), int(self.problem.n_rhs / 4 ) )
        ierr, iter, res, pres = solve_with_petsc_parallel(stiff, rhs, pot, petsc_options, np)
        #ierr, iter, res, pres = solve_with_petsc(stiff, problem.rhs, tdpot[:self.n_pot * self.problem.n_rhs], petsc_options)
        #print('out',pot[0:3])
        #pot, tdens, vel = self.get_otp_solution(tdpot)
        #print('res=',problem.constraint_residual(vel))



        # info
        if ctrl.verbose >=1 :
            msg=(f'{ierr=} avg_it={int(iter/self.problem.n_rhs):04d}'
                 +f' max(res)={res:.1e}'
                 +f' max(pres)={pres:.1e}')
            print(msg)

        # compute res residuum
        if (ctrl.verbose>1):
            pass#print(info_solver)
        ierr=0
        if (ierr !=0 ) :
            ierr=1

        return ierr
        

    def tdens2gfvar(self,tdens):
        """
        Transformation from tdens variable to gfvar (gradient flow variable)
        """
        gfvar = np.sqrt(tdens)
        return gfvar

    def gfvar2tdens(self,gfvar,derivative_order):
        """
        Compute \phi(gfvar)=tdens, \phi' (gfvar), or \phi''(gfvar)
        """
        if (derivative_order == 0 ):
            tdens = gfvar**2
        elif (derivative_order == 1 ):
            tdens = 2*gfvar
        elif (derivative_order == 2 ):
            tdens = 2*np.ones(len(gfvar))
        else:
            print('Derivative order not supported')
        return tdens

    def build_fnewton_gfvar(self, problem, pot, gfvar, gfvar_old, ctrl):
        # assembly nonlinear equation
        # F_pot=f_newton[1:n_pot]= stiff * pot - rhs
        # F_pot=f_newton[1+n_pot:n_pot + n_tdens] = -weight (gfvar-gfvar_old)/deltat + \grad \Lyapunov        
        tdens = self.gfvar2tdens(gfvar, 0) # 1 means first derivative
        trans_prime = self.gfvar2tdens(gfvar, 1) # 1 means first derivative
        trans_second = self.gfvar2tdens(gfvar, 2) # 2 means second derivative
        grad_pot = problem.potential_gradient(pot)

        
        f_newton = np.zeros(npot+ntdens)
        
        f_newton[0:n_pot] = (problem.matrix.dot(tdens*grad_pot) - problem.rhs)
        f_newton[n_pot:n_pot+n_tdens] = -problem.weight * (
            ( gfvar - gfvar_old ) / ctrl.deltat
            + trans_prime * 0.5* (-grad_pot**2 + 1.0)
        )
        return -f_newton

    def build_jacobian_gfvar(self,problem, pot, gfvar, ctrl):
        # assembly jacobian
        conductivity = tdens*problem.inv_weight
        A_matrix = self.build_stiff(problem.matrix, conductivity)
        B_matrix = sp.sparse.diags(trans_prime * grad_pot).dot(problem.matrixT)
        BT_matrix = B_matrix.transpose()

        # the minus sign is to get saddle point in standard form
        diag_C_matrix = problem.weight * (
            1.0 / ctrl.deltat
            + trans_second * 0.5 * (-grad_pot**2 + 1.0)
        )
        C_matrix = sp.sparse.diags(diag_C_matrix)
        msg=('{:.2E}'.format(min(diag_C_matrix))+'<=C <='+'{:.2E}'.format(max(diag_C_matrix)))
        if (ctrl.verbose >= 3 ):
            print(msg)
        return A_matrix,B_matrix,BT_matrix,C_matrix
    
    
    def iterate(self, problem, tdpot, ctrl):
        """
        Procedure overriding update of parent class(Problem)
        
        Args:
        problem: Class with inputs  (rhs, q_exponent)
        tdpot  : Class with unkowns (tdens, pot in this case)

        Returns:
         tdpot : update tdpot from time t^k to t^{k+1} 

        """
        if ctrl.time_discretization_method == 'explicit_tdens':            
            # compute update
            pot, tdens = self.subfunctions(tdpot)
            grad_pot = problem.grad.dot(pot)
            
            #print('{:.2E}'.format(min(normgrad))+'<=GRAD<='+'{:.2E}'.format(max(normgrad)))
            pmass = problem.q_exponent/(2-problem.q_exponent)
            grad_pot_sq = grad_pot**2
            if (self.problem.n_rhs==1):
                grad_pot_sq_sum = grad_pot_sq
            else:
                mat = vec2mat(grad_pot_sq, problem.n_rhs)
                grad_pot_sq_sum = np.sum(mat,axis=1)

            update = -(-tdens  * grad_pot_sq_sum + tdens**pmass)
            update_direction = (-grad_pot_sq_sum + tdens**(pmass-1))
            
            ctrl.deltat = ctrl.set_deltat(ctrl.deltat, tdens, update_direction)
            msg=f'{np.min(update):.2e}<=UPDATE<={np.max(update):.2e}, deltat={ctrl.deltat:.2e}'
            ctrl.print_info(msg,0,1)

            
            # update tdens
            tdens = tdens + ctrl.deltat * update
            tdens_min = ctrl.method_ctrl['tdens_min']
            tdens[np.where(tdens < tdens_min)] = tdens_min
            tdpot[-self.n_tdens:] = tdens
            
            self.time += ctrl.deltat
            problem.update_inputs(self.time)
            ierr = self.syncronize(problem, tdpot, ctrl)  
            
            
            return ierr

            
        elif (ctrl.time_discretization_method == 'explicit_gfvar'):            
            # compute update
            pot, tdens = self.subfunctions(sol) 
            gfvar = self.tdens2gfvar(tdens) 
            trans_prime = self.gfvar2tdens(gfvar, 1) # 1 means zero derivative so 
            grad = problem.potential_gradient(pot)


            update = - trans_prime * (grad * grad) +  trans_prime
            print('{:.2E}'.format(min(update))+'<=UPDATE<='+'{:.2E}'.format(max(update)))

            # update gfvar and tdens
            gfvar = gfvar - ctrl.deltat * update
            tdens = self.gfvar2tdens(gfvar, 0) # 0 means zero derivative so 

            # compute potential
            self.time = self.time + ctrl.deltat    
            problem.update_inputs(self.time)
            [tdpot,ierr,self] = self.syncronize(problem,tdpot)

            

            return ierr

        elif (ctrl.time_discretization_method == 'implicit_gfvar'):
            #shorthand
            n_pot = problem.n_row
            n_tdens = problem.n_col
            
            # pass in gfvar varaible
            pot, tdens = self.subfunctions(sol)
            gfvar_old = self.tdens2gfvar(tdens)
            gfvar = cp(gfvar_old)
            pot   = cp(pot)
            
            f_newton = np.zeros(n_pot+n_tdens)
            increment = np.zeros(n_pot+n_tdens)
            inewton = 0
            ierr_newton = 0

            # cycle until an error occurs
            while (ierr_newton == 0):
                # assembly nonlinear equation
                # F_pot=f_newton[1:n_pot]= stiff * pot - rhs
                # F_pot=f_newton[1+n_pot:n_pot + n_tdens] = -weight (gfvar-gfvar_old)/deltat + \grad \Lyapunov
                tdens = self.gfvar2tdens(gfvar, 0) # 1 means first derivative
                trans_prime = self.gfvar2tdens(gfvar, 1) # 1 means first derivative
                trans_second = self.gfvar2tdens(gfvar, 2) # 2 means second derivative
                grad_pot = problem.potential_gradient(pot)

    
                f_newton[0:n_pot] = (problem.matrix.dot(tdens*grad_pot) - problem.rhs)
                f_newton[n_pot:n_pot+n_tdens] = -problem.weight * (
                    ( gfvar - gfvar_old ) / ctrl.deltat
                    + trans_prime * 0.5* (-grad_pot**2 + 1.0)
                )
                f_newton = -f_newtonf_newton

                # check if convergence is achieved
                self.nonlinear_solver_residuum = np.linalg.norm(f_newton)
                msg=(str(inewton)+
                     ' |F|_pot  = '+'{:.2E}'.format(np.linalg.norm(f_newton[0:n_pot])) +
                     ' |F|_gfvar= '+'{:.2E}'.format(np.linalg.norm(f_newton[n_pot:n_pot+n_tdens])))
                if (ctrl.verbose >= 2 ):
                    print(msg)
                
                if ( self.nonlinear_solver_residuum < ctrl.tolerance_nonlinear ) :
                    ierr_newton == 0
                    break
                
                # assembly jacobian
                conductivity = tdens*problem.inv_weight
                A_matrix = self.build_stiff(problem.matrix, conductivity)
                B_matrix = sp.sparse.diags(trans_prime * grad_pot).dot(problem.matrixT)
                BT_matrix = B_matrix.transpose()

                # the minus sign is to get saddle point in standard form
                diag_C_matrix = problem.weight * (
                    1.0 / ctrl.deltat
                    + trans_second * 0.5 * (-grad_pot**2 + 1.0)
                )
                C_matrix = sp.sparse.diags(diag_C_matrix)
                

                if (ctrl.save_newton_matrices > 0):
                    base = f'time{self.current_iter:05d}_newton{inewton:05d}'
                    scipy.sparse.save_npz(base+'_A', A, compressed=True)
                    scipy.sparse.save_npz(base+'_B1T', B_matrix, compressed=True)
                    scipy.sparse.save_npz(base+'_B2T', BT_matrix, compressed=True)
                    scipy.sparse.save_npz(base+'_C', C_matrix, compressed=True)
                    np.save(base+'_pot_rhs',f_newton[0:n_pot])
                    np.save(base+'_gfvar_rhs',f_newton[n_pot:npot+ntdens])
                    np.save(base+'_pot',pot)
                    np.save(base+'_gfvar',f_newton[0:n_pot])
                    
                            
                
                inv_C_matrix = sp.sparse.diags(1.0/diag_C_matrix)

                
                # form primal Schur complement S=A+BT * C^{-1} B
                primal_S_matrix = A_matrix+BT_matrix.dot(inv_C_matrix.dot(B_matrix))+1e-12*sp.sparse.eye(n_pot)
                
                
                # solve linear system
                # increment
                primal_rhs = ( f_newton[0:n_pot]
                               + BT_matrix.dot(inv_C_matrix.dot(f_newton[n_pot:n_pot+n_tdens])) )
                increment[0:n_pot] = splinalg.spsolve(primal_S_matrix, primal_rhs,
                                                     use_umfpack=True)
                increment[n_pot:n_pot+n_tdens] = - inv_C_matrix.dot(
                    f_newton[n_pot:n_pot+n_tdens] - B_matrix.dot(increment[0:n_pot]))
                
                
                # line search to ensure C being strictly positive
                finished = False
                newton_step = 1.0
                current_pot = cp(pot)
                current_gfvar = cp(gfvar)
                while ( not finished):
                    # update pot, gfvar and derived components
                    pot = current_pot + newton_step * increment[0:n_pot]
                    gfvar = current_gfvar + newton_step * increment[n_pot:n_pot+n_tdens]
                    trans_second = self.gfvar2tdens(gfvar, 2) # 1 means zero derivative so
                    grad_pot = problem.potential_gradient(pot)

                    diag_C_matrix = problem.weight * (
                        1.0 / ctrl.deltat
                        + trans_second * 0.5*(-grad_pot **2 + 1.0 )
                    )

                    # ensure diag(C) beingstrctly positive
                    if ( np.amin(diag_C_matrix) < ctrl.min_C ):
                        newton_step =  newton_step / ctrl.contraction_newton_step
                        if (newton_step < ctrl.min_newton_step):
                            print('Newton step=',newton_step,'below limit', ctrl.min_newton_step)
                            ierr_newton = 2
                            finished = True
                    else:
                         ierr_newton = 0
                         finished = True
                msg='Newton step='+str(newton_step)
                if (ctrl.verbose >= 3 ):
                    print(msg)
                
                      
                # count iterations
                inewton += 1
                if (inewton == ctrl.max_nonlinear_iterations ):
                    ierr_newton = 1
                    # end of newton
           

            # copy the value in tdpot (even if the are wrong)
            sol[:self.n_pot] = pot
            sol[self.n_pot:] = self.gfvar2tdens(gfvar,0)
            self.time += ctrl.deltat

            # store info algorithm
            self.nonlinear_iterations = inewton


            # pass the newton error (0,1,2)
            ierr = ierr_newton

            return ierr
        else:
            print('value: ctrl.time_discretization_method not supported. Passed:',ctrl.time_discretization_method )
            ierr = 1

            return ierr

            
    def solve(self, problem, tdpot, ctrl, callback):
        """
        Solve the time dependent problem
        Args:
            problem: problem object
            tdpot: solution object
            ctrl: control object
        Returns:
            ierr: error code (0: success)
        """     
        # Syncronize the potential 
        problem.update_inputs(problem.initial_time)
        ierr = self.syncronize(problem, tdpot, ctrl)
        
        # Start main cycle
        self.iterations = 0
        while (ierr == 0) and (self.iterations < ctrl.max_iter):
            """ try to update the sol  """
            tdpot_old = cp(tdpot)
            nrestart = 0
            ierr_iterate = 0
            while ierr_iterate == 0 :
                # update inputs if time varying
                problem.update_inputs(self.time)
                ierr_iterate = self.iterate(problem, tdpot, ctrl)
                if ierr_iterate == 0:    
                    break
                else:
                    nrestart += 1
                    if nrestart == ctrl.max_restart:
                        break
                    ctrl.deltat = ctrl.deltat / 2.0

            # check if the iteration has been successful    
            if ierr_iterate !=0 :
                ierr = 1
                self.ierr_update = ierr_iterate 
                
            
            # check if the maximum number of iterations has been reached
            self.iterations += 1
            if self.iterations == ctrl.max_iter:
                ierr = 2
            
            # Here the user evalutes if convergence is achieved
            pot_old, tdens_old = self.subfunctions(tdpot_old)
            pot, tdens = self.subfunctions(tdpot)
            var = (
                norm(tdens - tdens_old) /
                (norm(tdens) * ctrl.deltat)
            )    
            if (var < 1e-3):
                ierr = 0
                break 
            
            if (ctrl.verbose >= 1):
                print(f'it={self.iterations} var={var:.2e}')

            if callback is not None:
                 callback(self,tdpot,ctrl)

                    
            """ Here user have to set solver controls for next update """
            #ctrl.set_before_iteration()

        return ierr

    def ierr_dictionary(self,ierr):
        """
        Return a description of the error
        """
        if ierr == 0:
            return 'No error'
        if ierr == 1:
            return 'Error in iterate procedure'
        if ierr == 2:
            return 'Maximum number of iterations reached'

def vec2mat(vec, n_rhs):
    """
    Internal procedure to convert vector to matrix
    """
    size = len(vec) // n_rhs
    return vec.reshape(size, n_rhs, order='F')
    
def mat2vec(mat):
    """
    Procedure to convert matrix to vector
    """
    n_rows, n_columns = mat.shape
    return mat.reshape(n_rows*n_columns, 1)
    
def get_portion(pot, n_portion, i):
    """
    Internal procedure to get i-th potential
    """
    n_pot = len(pot) // n_portion
    return pot[i*n_pot:(i+1)*n_pot]
