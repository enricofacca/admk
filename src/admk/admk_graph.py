# import Solver 
from copy import deepcopy as cp

import sys

import numpy as np
import scipy as sp
import scipy.sparse.linalg as splinalg
from scipy.linalg import norm 
import time as cputiming
import os
from .linear_solvers import info_linalg
from .linear_solvers import implicit_block_diag   



class TdensPotentialVelocity:
    """This class contains roblem solution tdens,pot and vel such that

    vel = diag(tdens) W^{-1} A^T pot
    
    """
    def __init__(self, n_tdens,n_pot, tdens0=None, pot0=None,time0=None):
        """
        Constructor of TdensPotential class, containing unkonws
        tdens, pot, flux

        Args:
            n_tdens (int) : length of tdens unknow
            n_pot (int) : length of pot unknow
            tdens0 (real) : non-negative initial tdens solution. Default: tdens=1.0 
            pot0 (real) : initial pot solution. Default: pot=0.0
            time0 (real) : initial pot solution. Default: time=0.0
        
        Raise:
        ValueError

        Example:
        graph_dmk=GraphDmk(np.array([0,1],[1,2]))
        tpdot=TdensPotential(graph_dmk,tdens0=np.ones(2),pot0=np.zeros(3))
        
        """
        #: Array size
        
        #: int: Number of tdens variable 
        self.n_tdens = n_tdens
        #: int: Number of tdens variable 
        self.n_pot = n_pot

        #: Tdens array
        self.tdens = np.ones(n_tdens)
        if ( not tdens0 is None):
            # dimension mismatch
            if not length(tdens0) == self.n_tdens:
                myError = ValueError(f'Passed length(tdens0)={len(tdens0):%d} !='+
                                     ' {len(tdens0):%d} = n_tdens')
                raise myError
            # negative values
            if ( any.tdens0 < 0 ) :
                myError = ValueError(f'tdens0 has negative entries')
                raise myError
            # set value
            self.tdens[:]=tdens0[:]
        self.pot=np.zeros(n_pot)
        if ( not pot0 is None):
            # dimension mismatch
            if ( not length(pot0)==self.n_pot):
                myError = ValueError(f'Passed length(pot0)={len(pot0):%d} !='+
                                     ' {len(pot0):%d} = n_pot')
                raise myError
            self.pot[:]=pot0[:]
        self.time=0.0
        if ( not time0 is None):
            self.time=time0
        
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
    - rhs = right-hand side
    """
    def __init__(self, matrix, weight=None):
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
        
        matvec = lambda x: self.inv_weight * self.matrixT.dot(x)
        self.gradient = splinalg.LinearOperator((self.n_col,self.n_row),matvec)    

        self.time_varing_inputs = False

    # def set_inputs(self,rhs,q_exponent=1.0):
    #     """
    #     Method to set problem inputs.

    #     Args:
    #         rhs (real) : vector on the right-hand side of equation
    #                      A vel = rhs
    #         q_exponent (real) : exponent q of the norm |vel|^q
    #     """
    #     print(len(rhs))
    #     if (len(rhs) % self.n_row != 0):
    #         myError = ValueError(f'Passed rhs.shape[0]={len(rhs):%d} is not a '+
    #                              'multiple of {self.nrow:%d} = nrow')
    #         raise myError
    #     else:
    #         self.n_rhs = len(rhs) // self.n_row
    #     print('n_rhs = ',self.n_rhs)

    #     self.rhs = cp(rhs)
    #     self.q_exponent = q_exponent

    #     if self.n_rhs != 1:
    #         self.grad = implicit_block_diag([self.gradient]*self.n_rhs)
    #     else:
    #         self.grad = self.gradient

    #     return self
    
    def set_inputs(self, rhs_of_time, q_exponent=1.0, sample_time=0.0):
        """
        Work in place procedure to set a rhs and q_exponent. 
        They may be function of time that are time varying
        """
        if callable(rhs_of_time):
            self.rhs_of_time = rhs_of_time
            self.time_varing_inputs = True
        else:
            # define a constant in time rhs
            self.rhs_of_time = lambda t: rhs_of_time

        # We set the rhs at the sample time
        # to get the number of rhs and fix the rhs 
        # if fix in time
        self.rhs = self.rhs_of_time(sample_time)
        if (len(self.rhs) % self.n_row != 0):
            myError = ValueError(f'Passed rhs.shape[0]={len(self.rhs):%d} is not a '+
                                 'multiple of {self.nrow:%d} = nrow')
            raise myError
        else:
            self.n_rhs = len(self.rhs) // self.n_row


        if callable(q_exponent):
            self.q_exponent_of_time = q_exponent
            self.time_varing_inputs = True
        else:
            # define a constant in time exponent
            self.q_exponent_of_time = lambda t: q_exponent
        # We set the exponent at the sample time
        self.q_exponent = self.q_exponent_of_time(sample_time)

        ierr = self.check_inputs()
        if (ierr != 0):
            print('Error in inputs at time t=',sample_time)

        if self.n_rhs != 1:
            self.grad = implicit_block_diag([self.gradient]*self.n_rhs)
        else:
            self.grad = self.gradient
    
    def update_inputs(self,time):
        """
        Update inputs at time t
        """
        # nothing to if self.time_varing_inputs is False
        # see set_inputs procedure
        if self.time_varing_inputs:            
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
        file_xdmf.write_data(0.0, point_data={"pot": tdpot.pot})
        file_xdmf.__exit__()


        filename='sol.xdmf'
        
        with meshio.xdmf.TimeSeriesWriter(filename) as writer:
            writer.write_points_cells(points, cells)
            #file_xdmf=meshio.xdmf.TimeSeriesWriter(filename,data_format='HDF')
            #file_xdmf.__enter__()
            #file_xdmf.write_points_cells(points, cells)
        
    

class AdmkControls:
    """
    Class with Admk Solver 
    """
    def __init__(self,
                 deltat=0.01,
                 approach_linear_solver='bicgstab',
                 max_linear_iterations=1000,
                 tolerance_linear=1e-9,
                 max_nonlinear_iterations=30,
                 tolerance_nonlinear=1e-10):
        """
        Set the controls of the Dmk algorithm
        """
        #: character: time discretization approach
        self.time_discretization_method = 'explicit_tdens'

        #: real: time step size
        self.max_iter = 100
        self.deltat = deltat

        # option to adapt time step:
        # fixed : fixed time step
        # expand : adaptive time step according to expansion_deltat
        # contract : adaptive time step according to error
        self.deltat_control = 'fixed'
        self.min_deltat = 1e-2
        self.max_deltat = 1e+2
        self.expansion_deltat = 2

        self.max_restart = 2
        
        #: int: max number of Krylov solver iterations
        self.max_linear_iterations = max_linear_iterations
        #: str: Krylov solver approach
        self.approach_linear_solver = approach_linear_solver
        #: real: Krylov solver tolerance
        self.tolerance_linear = tolerance_linear
        
        #: real: nonlinear solver iteration
        self.tolerance_nonlinear = tolerance_nonlinear

        #: int: Max number of nonlinear solver iterations 
        self.max_nonlinear_iterations = 20
        
        #: real: minimum newton step
        self.min_newton_step = 5e-2
        self.contraction_newton_step = 1.05
        self.min_C = 1e-6
        
        
        #: Fillin for incomplete factorization
        self.outer_prec_fillin=20
        #: Drop tolerance for incomplete factorization
        self.outer_prec_drop_tolerance=1e-4

        #: info on standard output
        self.verbose=0
        #: info on log file
        self.save_log=0
        self.file_log='admk.log'

    def print_info(self, msg, priority):
        """
	    Print messagge to stdout and to log 
        file according to priority passed
        """
        if (self.verbose > priority):
            print(msg)
      

    def set_before_iteration(self):
        """
        Procedure to set new controls after a succesfull update
        """
        if (self.deltat_control == 'fixed'):
            self.deltat = self.deltat
        elif (self.deltat_control == 'expanding'):
            self.deltat = max( min( self.deltat *
                                    self.expansion_deltat, self.max_deltat),
                               self.min_deltat)
        return self

    def reset_after_failure(self,ierr):
        """
        Procedure to set new controls after a succesfull update
        """
        self.deltat = max( min( self.deltat /
                            self.expansion_deltat, self.max_deltat),
                           self.min_deltat)
        return self

        
        
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

    
    
    def initial_solution(self):
        sol = TdensPotentialVelocity(self.n_pot*self.problem.n_rhs, self.n_tdens)
        return sol

    
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
		
        # assembly stiff
        msg = (f'{min(tdpot.tdens):.2E}<=TDENS<={max(tdpot.tdens):.2E}')
        ctrl.print_info(msg, 3)
        
        start_time = cputiming.time()
        conductivity = tdpot.tdens * problem.inv_weight
        stiff = self.build_stiff(problem.matrix, conductivity)
        rhs = problem.rhs.copy()

        n_equ = len(rhs)
        msg = ('ASSEMBLY'+'{:.2f}'.format(-(start_time - cputiming.time()))) 
        ctrl.print_info(msg,3)
    
        #
        # solve linear system
        #
        inode = 1
        grounding = True
        if (grounding):
            stiff[inode,inode] = 1.0e20
            for i_rhs in range(self.problem.n_rhs):
                rhs[inode+i_rhs*self.n_pot] = 0.0

        else:
            stiff=stiff

        # create block diagonal matrix
        matrix2solve = stiff
        if (self.problem.n_rhs>1):
            matrix2solve = sp.sparse.block_diag([matrix2solve]*problem.n_rhs)


        start_time = cputiming.time()
        ilu = splinalg.spilu(matrix2solve,
                             drop_tol=ctrl.outer_prec_drop_tolerance,
                       fill_factor=ctrl.outer_prec_fillin)
        if (ctrl.verbose>2):
            print('ILU'+'{:.2f}'.format(-(start_time - cputiming.time()))) 
        prec = lambda x: ilu.solve(x)
        M = splinalg.LinearOperator((n_equ,n_equ), prec)
        
        # solve linear system
        start_time = cputiming.time()
        [pot,ierr] = splinalg.bicgstab(
            matrix2solve, rhs, x0=tdpot.pot,
            tol=ctrl.tolerance_nonlinear, #restart=20,
            maxiter=ctrl.max_linear_iterations,
            atol=1e-16,
            M=M)
        
        tdpot.pot[:]=pot[:]

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
            grad_pot = problem.grad.dot(tdpot.pot)
            
            #print('{:.2E}'.format(min(normgrad))+'<=GRAD<='+'{:.2E}'.format(max(normgrad)))
            pmass = problem.q_exponent/(2-problem.q_exponent)
            grad_pot_sq = grad_pot**2
            if (self.problem.n_rhs==1):
                grad_pot_sq_sum = grad_pot_sq
            else:
                mat = vec2mat(grad_pot_sq, problem.n_rhs)
                grad_pot_sq_sum = np.sum(mat,axis=1)

            update = -tdpot.tdens  * grad_pot_sq_sum + tdpot.tdens**pmass

            # update tdgrad_poens
            tdpot.tdens = tdpot.tdens - ctrl.deltat * update

            ierr = self.syncronize(problem, tdpot, ctrl)  
            tdpot.time=tdpot.time+ctrl.deltat
            
            return ierr

            
        elif (ctrl.time_discretization_method == 'explicit_gfvar'):            
            # compute update
            gfvar = self.tdens2gfvar(tdpot.tdens) 
            trans_prime = self.gfvar2tdens(gfvar, 1) # 1 means zero derivative so 
            grad = problem.potential_gradient(tdpot.pot)


            update = - trans_prime * (grad * grad) +  trans_prime
            print('{:.2E}'.format(min(update))+'<=UPDATE<='+'{:.2E}'.format(max(update)))

            # update gfvar and tdens
            gfvar = gfvar - ctrl.deltat * update
            tdpot.tdens = self.gfvar2tdens(gfvar, 0) # 0 means zero derivative so 

            # compute potential
            [tdpot,ierr,self] = self.syncronize(problem,tdpot)

            tdpot.time=tdpot.time+ctrl.deltat    

            return ierr

        elif (ctrl.time_discretization_method == 'implicit_gfvar'):
            #shorthand
            n_pot = problem.nrow
            n_tdens = problem.ncol
            
            # pass in gfvar varaible
            gfvar_old = self.tdens2gfvar(tdpot.tdens)
            gfvar = cp(gfvar_old)
            pot   = cp(tdpot.pot)
            
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
                f_newton=-f_newton

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
                msg=('{:.2E}'.format(min(diag_C_matrix))+'<=C <='+'{:.2E}'.format(max(diag_C_matrix)))
                if (ctrl.verbose >= 3 ):
                    print(msg)
                
                C_matrix = sp.sparse.diags(diag_C_matrix)
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
            tdpot.pot[:] = pot[:]
            tdpot.tdens = self.gfvar2tdens(gfvar,0)
            tdpot.time += ctrl.deltat

            # store info algorithm
            self.nonlinear_iterations = inewton


            # pass the newton error (0,1,2)
            ierr = ierr_newton

            return ierr
        else:
            print('value: ctrl.time_discretization_method not supported. Passed:',ctrl.time_discretization_method )
            ierr = 1

            return ierr

            
    def solve(self, problem, tdpot, ctrl):
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
        ierr = self.syncronize(problem, tdpot, ctrl)
        
        # Start main cycle
        iter = 0
        while ierr == 0 :
            """ try to update the sol  """
            tdpot_old = cp(tdpot)
            nrestart = 0
            ierr_iterate = 0
            while ierr_iterate == 0 :
                # update inputs if time varying
                problem.update_inputs(tdpot.time)

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
            iter += 1
            if iter == ctrl.max_iter:
                ierr = 2
            
            # Here the user evalutes if convergence is achieved
            var = (
                norm(tdpot.tdens - tdpot_old.tdens) /
                (norm(tdpot.tdens) * ctrl.deltat)
            )    
            if (var < 1e-3):
                ierr = 0
                break 
            
            """ Study state system """
            grad = problem.grad.dot(tdpot.pot)
            grad_matrix = vec2mat(grad**2,problem.n_rhs)
            
            if (ctrl.verbose > 0):
                print(' ')
                print('var=',var)
        
            if (ctrl.verbose > 1):
                print(' ')
                print('iter,', iter,'time,',tdpot.time)
                grad_pot_sq_sum = np.sum(grad_matrix,axis=1)
                print(
                        f'{min(grad_pot_sq_sum):.2E}'
                        +f'<=sum |GRAD|<='
                        +f'{max(grad_pot_sq_sum):.2E}')
                for i in range(problem.n_rhs):
                    print(
                        f'{min(abs(grad_matrix[:,i])):.2E}'
                        +f'<=|GRAD|_{i:d} <='
                        +f'{max(abs(grad_matrix[:,i])):.2E}')
            
            """ Here user have to set solver controls for next update """
            ctrl.set_before_iteration()

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