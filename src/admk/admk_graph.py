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
            if ( not length(tdens0)==self.n_tdens):
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
        if ( not pot0 is None):
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
    def __init__(self, matrix,weight=None,matrixT=None):
        """
        Constructor of problem setup
        """
        self.matrix = matrix
        self.nrow = matrix.shape[0]
        self.ncol = matrix.shape[1]
        if (matrixT is None):
            self.matrixT=self.matrix.transpose()
        else:
            self.matrixT=matrixT
        
        # edge weight
        if (weight is None):
            weight = np.ones(len(topol))
        self.weight = weight
        self.inv_weight = 1.0/weight

         # allocate space for inputs 
        self.rhs=np.zeros(self.nrow)
        self.q_exponent=1.0

    def set_inputs(self,rhs,q_exponent=1.0):
        """
        Method to set problem inputs.

        Args:
            rhs (real) : vector on the right-hand side of equation
                         A vel = rhs
            q_exponent (real) : exponent q of the norm |vel|^q
        """
        self.rhs[:]=rhs[:]
        self.q_exponent=q_exponent
        return self
    
    def check_inputs(self):
        """
        Method to check problem inputs consistency
        """
        ierr=0
        if (np.sum(self.rhs)>1e-12):
            print('Rhs is not balanced')
            ierr=1
        return ierr

    def potential_gradient(self,pot):
        """
        Procedure to compute gradient of the potential
        grad=W^{-1} A^T pot

        Args:
        pot: real (nrow of A)-np.array with potential

        Returns:
        grad: real (ncol of A)-np.array with gradient
        """
        grad = self.inv_weight*self.matrixT.dot(pot)
        return grad;

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
        self.n_edge    = len(topol)
        if (np.amin(topol) == 0 ):
            # 0-based numbering
            n_node=np.amax(topol)+1
        elif ( np.amin(topol) == 1) :
            # 1-based numbering
            n_node=np.amax(topol)
        # member with nodes number
        self.n_node      = n_node

        # graph topology
        self.topol     = cp(topol)


    def signed_incidence_matrix(self):
        """
        Build signed incidence matrix
        
        Args:
        topol: (2,ndege)-integer np-array 1-based ordering
        
        Result:
        matrix : signed incidence matrix 
        """
        n_tdens=len(self.topol)
        n_pot=np.amax(self.topol)+1
        
        # build signed incidence matrix
        indptr  = np.zeros([2*n_tdens]).astype(int) # rows
        indices = np.zeros([2*n_tdens]).astype(int) # columns
        data    = np.zeros([2*n_tdens])                # nonzeros
        for i in range(n_tdens):
            indptr[2*i:2*i+2]  = int(i)
            indices[2*i] = int(self.topol[i,0])
            indices[2*i+1] = int(self.topol[i,1])
            data[2*i:2*i+2]    = [1.0,-1.0]
            #print(topol[i,:],indptr[2*i:2*i+2],indices[2*i:2*i+2],data[2*i:2*i+2])
        signed_incidence = sp.sparse.csr_matrix((data, (indptr,indices)),shape=(n_tdens, n_pot))
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
        self.deltat = deltat

        # variables for set and reset procedure
        self.deltat_control = 0
        self.min_deltat = 1e-2
        self.max_deltat = 1e+2
        self.expansion_deltat = 2
        
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

      

    def set_before_iteration(self):
        """
        Procedure to set new controls after a succesfull update
        """
        if (self.deltat_control == 0):
            self.deltat = self.deltat
        elif (self.deltat_control == 1):
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

        

# Create a class to store solver info 
class InfoAdmkSolver():
    def __init__(self):
        self.linear_solver_iterations = 0
        # non linear solver
        self.nonlinear_solver_iterations = 0
        self.nonlinear_sovler_residum = 0.0

        
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
    def __init__(self, ctrl = None):
        """
		Initialize solver with passed controls (or default)
        and initialize structure to store info on solver application
        """
        if (ctrl == None):
            self.ctrl = AdmkControls()
        else:
            self.ctrl = cp(ctrl)
			
		# init infos
        self.info = InfoAdmkSolver()

    def print_info(self, msg, priority):
        """
	Print messagge to stdout and to log 
        file according to priority passed
        """
        if (self.ctrl.verbose > priority):
            print(msg)


    def build_stiff(self, matrixA, conductivity):
        """
        Internal procedure to assembly stifness matrix 
        S(tdens)=A conductivity A^T
		
        Args:
         conductivity: non-negative real (ncol of A)-np.array with conductivities

        Returns:
		 stiff: Scipy sparse matrix
        """
        diagt=sp.sparse.diags(conductivity)
        stiff=matrixA.dot(diagt.dot(matrixA.transpose()))
        return stiff


    def syncronize(self, problem, tdpot, ierr):
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
        self.print_info(msg, 3)
        
        start_time = cputiming.time()
        conductivity = tdpot.tdens*problem.inv_weight
        stiff = self.build_stiff(problem.matrix,conductivity)
        rhs = problem.rhs.copy()

        msg = ('ASSEMBLY'+'{:.2f}'.format(-(start_time - cputiming.time()))) 
        self.print_info(msg,3)

        
        
        #
        # solve linear system
        #

        # init counter
        info_solver = info_linalg()
        info_solver.resini=norm(stiff*tdpot.pot-problem.rhs)/norm(problem.rhs)
        
        # ground solution
        inode=np.argmax(abs(problem.rhs))
        grounding=True
        #grounding=False
        if (grounding):
            stiff[inode,inode] = 1.0e20
            rhs[inode] = 0.0
        else:
            stiff=stiff#+1e-12*sp.sparse.eye(tdpot.n_pot)
            
        # scaling=True
        scaling =False
        if ( scaling ):
            d=stiff.diagonal()
            diag_sqrt=sp.sparse.diags(1.0/np.sqrt(d))

            matrix2solve  = diag_sqrt*(stiff*diag_sqrt)
            rhs2solve     = diag_sqrt*rhs
            x0            = diag_sqrt*tdpot.pot
        else:
            matrix2solve = stiff
            rhs2solve    = rhs
            x0           = tdpot.pot

        start_time = cputiming.time()
        ilu = splinalg.spilu(matrix2solve,
                             drop_tol=self.ctrl.outer_prec_drop_tolerance,
                       fill_factor=self.ctrl.outer_prec_fillin)
        if (self.ctrl.verbose>2):
            print('ILU'+'{:.2f}'.format(-(start_time - cputiming.time()))) 
        prec = lambda x: ilu.solve(x)
        M = splinalg.LinearOperator((tdpot.n_pot,tdpot.n_pot), prec)

        
        
        # solve linear system
        start_time = cputiming.time()
        [pot,info_solver.info]=splinalg.bicgstab(
            matrix2solve, rhs2solve, x0=x0,
            tol=self.ctrl.tolerance_nonlinear, #restart=20,
            maxiter=self.ctrl.max_linear_iterations,
            atol=1e-16,
            M=M,
            callback=info_solver.addone)
        
        if (scaling) :
            pot=diag_sqrt*pot
            rhs2solve=np.sqrt(d)*rhs2solve
            print('LINSOL'+'{:.2f}'.format(
                -(start_time - cputiming.time())))

        tdpot.pot[:]=pot[:]
        # compute res residuum
        info_solver.realres=norm(
            matrix2solve.dot(tdpot.pot)-rhs2solve
        )/norm(rhs2solve)
        if (self.ctrl.verbose>1):
            print(info_solver)
        ierr=0
        if (info_solver.info !=0 ) :
            ierr=1
        

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
    
    
    def iterate(self, problem, tdpot, ierr):
        """
        Procedure overriding update of parent class(Problem)
        
        Args:
        problem: Class with inputs  (rhs, q_exponent)
        tdpot  : Class with unkowns (tdens, pot in this case)

        Returns:
         tdpot : update tdpot from time t^k to t^{k+1} 

        """
        if (self.ctrl.time_discretization_method == 'explicit_tdens'):            
            # compute update
            grad = problem.potential_gradient(tdpot.pot)
            flux = tdpot.tdens*grad
            res = problem.matrix*flux-problem.rhs
            print(sp.linalg.norm(res))
            
            #print('{:.2E}'.format(min(normgrad))+'<=GRAD<='+'{:.2E}'.format(max(normgrad)))
            pmass=problem.q_exponent/(2-problem.q_exponent)
            update=-tdpot.tdens  * (grad * grad) + tdpot.tdens**pmass

            # update tdens
            tdpot.tdens = tdpot.tdens - self.ctrl.deltat * update

            [tdpot,ierr,self] = self.syncronize(problem,tdpot)

            
            tdpot.time=tdpot.time+self.ctrl.deltat
            
        elif (self.ctrl.time_discretization_method == 'explicit_gfvar'):            
            # compute update
            gfvar = self.tdens2gfvar(tdpot.tdens) 
            trans_prime = self.gfvar2tdens(gfvar, 1) # 1 means zero derivative so 
            grad = problem.potential_gradient(tdpot.pot)


            update = - trans_prime * (grad * grad) +  trans_prime
            print('{:.2E}'.format(min(update))+'<=UPDATE<='+'{:.2E}'.format(max(update)))

            # update gfvar and tdens
            gfvar = gfvar - self.ctrl.deltat * update
            tdpot.tdens = self.gfvar2tdens(gfvar, 0) # 0 means zero derivative so 

            # compute potential
            [tdpot,ierr,self] = self.syncronize(problem,tdpot)

            tdpot.time=tdpot.time+self.ctrl.deltat    

        elif (self.ctrl.time_discretization_method == 'implicit_gfvar'):
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
                    ( gfvar - gfvar_old ) / self.ctrl.deltat
                    + trans_prime * 0.5* (-grad_pot**2 + 1.0)
                )
                f_newton=-f_newton

                # check if convergence is achieved
                self.info.nonlinear_solver_residuum = np.linalg.norm(f_newton)
                msg=(str(inewton)+
                     ' |F|_pot  = '+'{:.2E}'.format(np.linalg.norm(f_newton[0:n_pot])) +
                     ' |F|_gfvar= '+'{:.2E}'.format(np.linalg.norm(f_newton[n_pot:n_pot+n_tdens])))
                if (self.ctrl.verbose >= 2 ):
                    print(msg)
                
                if ( self.info.nonlinear_solver_residuum < self.ctrl.tolerance_nonlinear ) :
                    ierr_newton == 0
                    break
                
                # assembly jacobian
                conductivity = tdens*problem.inv_weight
                A_matrix = self.build_stiff(problem.matrix, conductivity)
                B_matrix = sp.sparse.diags(trans_prime * grad_pot).dot(problem.matrixT)
                BT_matrix = B_matrix.transpose()

                # the minus sign is to get saddle point in standard form
                diag_C_matrix = problem.weight * (
                    1.0 / self.ctrl.deltat
                    + trans_second * 0.5 * (-grad_pot**2 + 1.0)
                )
                msg=('{:.2E}'.format(min(diag_C_matrix))+'<=C <='+'{:.2E}'.format(max(diag_C_matrix)))
                if (self.ctrl.verbose >= 3 ):
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
                        1.0 / self.ctrl.deltat
                        + trans_second * 0.5*(-grad_pot **2 + 1.0 )
                    )

                    # ensure diag(C) beingstrctly positive
                    if ( np.amin(diag_C_matrix) < self.ctrl.min_C ):
                        newton_step =  newton_step / self.ctrl.contraction_newton_step
                        if (newton_step < self.ctrl.min_newton_step):
                            print('Newton step=',newton_step,'below limit', self.ctrl.min_newton_step)
                            ierr_newton = 2
                            finished = True
                    else:
                         ierr_newton = 0
                         finished = True
                msg='Newton step='+str(newton_step)
                if (self.ctrl.verbose >= 3 ):
                    print(msg)
                
                      
                # count iterations
                inewton += 1
                if (inewton == self.ctrl.max_nonlinear_iterations ):
                    ierr_newton = 1
                    # end of newton


           

            # copy the value in tdpot (even if the are wrong)
            tdpot.pot[:] = pot[:]
            tdpot.tdens = self.gfvar2tdens(gfvar,0)
            tdpot.time = tdpot.time+self.ctrl.deltat

            # store info algorithm
            self.info.nonlinear_iterations = inewton


            # pass the newton error (0,1,2)
            ierr = ierr_newton
        else:
            print('value: self.ctrl.time_discretization_method not supported. Passed:',self.ctrl.time_discretization_method )
            ierr = 1

            


