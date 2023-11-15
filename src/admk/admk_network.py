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
    def __init__(self, 
                 matrix,
                    rhs, 
                    q_exponent=1.0,
                    weight=None,
                    dirichlet_nodes=None):
        """
        Constructor of problem setup
        """
        self.matrix = matrix
        self.n_row = matrix.shape[0]
        self.n_col = matrix.shape[1]
        
        self.matrixT = self.matrix.transpose()

        self.Dirichlet_nodes = dirichlet_nodes

        # edge weight
        if (weight is None):
            weight = np.ones(self.n_col)
        self.weight = weight
        self.inv_weight = 1.0 / weight
        
        self.inv_W = sp.sparse.diags(self.inv_weight)
        self.Grad = diagt.dot(self.matrixT)
        self.Div = self.matrix
        
        
        
            
        # We set the rhs at the initial time
        # to get the number of rhs and fix the rhs 
        # if fix in time
        self.rhs = cp(rhs)
        self.q_exponent = q_exponent

        ierr = self.check_inputs()
        if (ierr != 0):
            raise ValueError('Error in inputs')
    
    
    def check_inputs(self, tolerance_inbalance=1e-11):
        """
        Method to check problem inputs consistency
        """
        ierr=0
        balance = np.sum(self.rhs)/np.linalg.norm(self.rhs)
        if balance > tolerance_inbalance and self.Dirichlet_nodes is None:
            print(f'Rhs is not balanced {balance:.1E}')
            ierr=1
        return ierr

    
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
        res = np.linalg.norm(self.div.dot(vel) - self.rhs) / rhs_norm
        return res

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
                 verbose = 0
                 ):
        """
        Set the controls of the Dmk solver
        """
        self.methods_list = ['explicit_tdens']

        self.all = {
            'tol_opt': tol_optimization,
            'tol_constraint': tol_constraint,
            'max_iter': max_iter,
            'max_restart': max_restart,
            'method': method,
            # monitor controls
            'verbose': verbose,
            'log': 0,
            'log_file': 'admk.log',
            # available methods
            'explicit_euler' : self.explicit_euler(tol_constraint),
            # 
            'relax_Laplacian': 1e-10,
            # linear solver controls for first iteration
            'ksp': {
                'type':'cg',
                'rtol': tol_constraint,
                'norm_type': 'unpreconditioned',
            },
            'pc':{
                'type': 'hypre',
            #    # used if for ilu only 
                'factor_drop_tolerance':{
                    'dt': 1e-4,
                    'maxrowcount': 30
                }
            },
        }

    def explicit_euler(self, tol_constraint):
        ctrl= {
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
        return ctrl
        
    def get(self, keys_list):
        if not isinstance(keys_list, list):
            keys_list=[keys_list]
        return nested_set(self.all,keys_list)


    def set(self, keys, value):
        if not isinstance(keys, list):
            keys=[keys]
        self.all = nested_set(self.all,keys,value)
        if self['method'] == 'explicit_tdens':
            tol_linear = self.get(['explicit_euler','ksp','rtol'])
            if not np.isclose(tol_linear, self.tol_constraint):
                self.all = nested_set(self.all,['explicit_euler','ksp','rtol'],  self.tol_constraint)
                raise UserWarning('Controls are not consistent is not set')

    def current(self):
        """
        Create a copy of the current controls 
        with only the method currently used
        """
        simplified = cp(self.all)

        used = self.get(['method'])
        not_used = [key for key in methods_list if key != used]
        for key in not_used:
            del simplified[key]
        return simplified
                    
                
        
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

def solve_with_petsc(stiff, rhs, pot, petsc_options):
    """
    Solve linear system with petsc
    """

    n_pot = stiff.shape[0]
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
    
    # convert to petsc
    petsc_rhs.setArray(rhs)
    petsc_pot.setArray(pot)

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
    pot = petsc_pot.getArray()
    
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
    def __init__(self,problem, 
                 tol_optimization = 1e-3,
                 tol_constraint = 1e-8,
                 max_iter = 200,
                 n_restart = 5,
                 method = 'explicit_tdens',
                 verbose = 0):
        """
		Initialize solver with passed controls (or default)
        and initialize structure to store info on solver application
        """
        self.problem = problem

        self.ctrl = AdmkControls(tol_optimization,
                                    tol_constraint,
                                    n_restart,
                                    max_iter,
                                    method,
                                    verbose)



		# init infos
        self.linear_solver_iterations = 0
        self.nonlinear_solver_iterations = 0
        self.nonlinear_solver_residum = 0.0
        self.n_pot = problem.n_row
        self.n_tdens = problem.n_col

    
    def initial_solution(self):
        sol = np.zeros(self.n_pot+self.n_tdens)
        sol[-self.n_tdens:] = 1.0
        return sol
        

    def get_otp_solution(self, sol):
        pot, tdens = self.subfunctions(sol)
        vel = tdens * self.problem.gradient.dot(pot)
        return pot, tdens, vel
    

    def subfunctions(self,sol):
        """
        Split solution in pot, tdens component
        """
        pot = sol[:self.n_pot]
        tdens = sol[-self.n_tdens:]
        return pot, tdens 

    
    
    def syncronize(self, tdpot, petsc_options):
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
        self.ctrl.print_info(msg, 3)

        # set matrix and rhs
        start_time = cputiming.time()
        Diag_tdens = sp.sparse.diags(tdens)
        stiff = self.Div.dot(Diag_tdens.dot(self.inv_W.dot(self.Grad)))
        msg = ('ASSEMBLY'+'{:.2f}'.format(-(start_time - cputiming.time()))) 
        self.ctrl.print_info(msg,3)
    
        rhs = problem.rhs.copy()

        #
        # solve linear system
        #
        relax = self.ctrl.get(['relax_Laplacian'])
        stiff += relax * sp.sparse.eye(self.n_pot) # matrix is singular
        ierr, iter, res, pres = solve_with_petsc(stiff, rhs, pot, petsc_options )

        # info
        if self.ctrl.verbose >=1 :
            msg =(f'{ierr=} it={iter:04d}'
                 +f' max(res)={res:.1e}'
                 +f' max(pres)={pres:.1e}')
            print(msg)

        return ierr
    
    def pmass(self):
        return self.problem.q_exponent / ( 2 - self.problem.q_exponent)
    
    def Lagrangian(self,pot,tdens):
        """
        Compute Lagrangian of the problem
        """
        grad_pot = self.problem.gradient.dot(pot)
        forcing = problem.source - problem.sink
        L = ( np.dot(forcing,pot) 
             - np.dot(tdens * grad_pot**2 / 2, self.problem.weight)
             + 0.5 * np.dot(tdens ** self.pmass(), self.problem.weight)
        )   
        return L
        

    def tdens2gfvar(self,tdens):
        """
        Transformation from tdens variable to gfvar (gradient flow variable)
        """
        gfvar = np.sqrt(tdens)
        return gfvar

    def gfvar2tdens(self, gfvar, derivative_order = 0):
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
            # compute update direction
            pot, tdens = self.subfunctions(tdpot)
            grad_pot = problem.grad.dot(pot)
            pmass = problem.q_exponent/(2-problem.q_exponent)
            gradient_Lagrangian = (-grad_pot** + tdens**(pmass-1))
            update = tdens * gradient_Lagrangian

            # set step
            self.deltat = ctrl.set_deltat(self.deltat, tdens, update)
            msg=f'{np.min(update):.2e}<=UPDATE<={np.max(update):.2e}, deltat={ctrl.deltat:.2e}'
            self.ctrl.print_info(msg,0,1)

            
            # update tdens
            tdens = tdens + ctrl.deltat * update
            tdens_min = ctrl.method_ctrl['tdens_min']
            tdens[np.where(tdens < tdens_min)] = tdens_min
            tdpot[-self.n_tdens:] = tdens            
            self.time += self.deltat
            
            # set linear solvers
            petsc_options = (flatten_parameters({
                **{'ksp' :ctrl.get('ksp')},
                **{
                    'ksp_initial_guess_nonzero': True,
                    #'ksp_monitor_true_residual': None,
                },
                **{'pc' : ctrl.method_ctrl.get('pc')}
                }))
            ierr = self.solve_pot(problem, tdpot, petsc_options)  
            
            
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

    def set_ctrl(self, key, value):
        self.ctrl.set(key, value)
            
    def solve(self, tdpot=None):
        """
        Solve the time dependent problem
        Args:
            problem: problem object
            tdpot: solution object
            ctrl: control object
        Returns:
            ierr: error code (0: success)
        """ 
        if tdpot is None:
            tdpot = self.initial_solution()

        # solve first Laplacian
        petsc_options = (flatten_parameters({
                **{'ksp' :ctrl.get('ksp')},
                **{'pc' : ctrl.get('pc')}
                }))
        ierr = self.solve_pot(tdpot, petsc_options)
        
        # Start main cycle
        iter = 0
        while (ierr == 0) and (iter < self.ctrl.max_iter):
            #try to update the solution
            tdpot_old = cp(tdpot)
            nrestart = 0
            ierr_iterate = 0
            while ierr_iterate == 0 :
                # update inputs if time varying
                ierr_iterate = self.iterate(problem, tdpot, ctrl)
                if ierr_iterate == 0:    
                    break
                else:
                    nrestart += 1
                    if nrestart == ctrl.max_restart:
                        break
                    ctrl.deltat = ctrl.deltat / 2.0

            # check if the iteration has been successful    
            if ierr_iterate != 0:
                ierr = 1
                self.ierr_update = ierr_iterate 
                
            
            # check if the maximum number of iterations has been reached
            iter += 1
            if iter == self.ctrl.get('max_iter'):
                ierr = 2
            
            # Here the user evalutes if convergence is achieved
            pot_old, tdens_old = self.subfunctions(tdpot_old)
            pot, tdens = self.subfunctions(tdpot)
            var = (
                norm(tdens - tdens_old) /
                (norm(tdens) * self.deltat)
            )    
            if var < self.ctrl.get('tol_optimization'):
                ierr = 0
                break 
            
            """ Study state system """
            grad_pot = problem.Grad.dot(pot)
            
            if (ctrl.verbose >= 2):
                print(' ')
                print(
                        f'{min(grad_pot):.2E}'
                        +f'<=sum |GRAD|<='
                        +f'{max(grad_pot):.2E}')
                print(
                    f'{min(tdens):.2E}'
                    +f'<=TDENS<='
                    +f'{max(tdens):.2E}')
            
            if (ctrl.verbose >= 1):
                print(f'it={iter} var={var:.2e}')
        
                    
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
