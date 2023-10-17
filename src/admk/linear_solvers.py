from scipy.linalg import norm 
import scipy.sparse.linalg as splinalg
#from scipy import linalg
import time as cputiming
import os
from petsc4py import PETSc
def _make_reasons(reasons):
    return dict([(getattr(reasons, r), r)
                 for r in dir(reasons) if not r.startswith('_')])
SNESReasons = _make_reasons(PETSc.SNES.ConvergedReason())
KSPReasons = _make_reasons(PETSc.KSP.ConvergedReason())

class info_linalg:
    """
    Class to store information in the linear solvers usage
    """
    def __init__(self):
        self.ierr = 0
        self.iter = 0
        self.resini = 0.0
        self.realres = 0.0

    def __str__(self):
        strout=(str(self.info)+' '+
                str(self.iter)+' '+
                str('{:.2E}'.format(self.resini))+' '
                +str('{:.2E}'.format(self.realres)))
        return strout;
    def addone(self, xk):
        self.iter += 1


def solve_linear_system(stiff, rhs, sol):
    d=stiff.diagonal()
    diag_sqrt=sp.sparse.diags(1.0/np.sqrt(d))

    matrix2solve  = diag_sqrt*(stiff*diag_sqrt)
    rhs2solve     = diag_sqrt*rhs
    x0            = diag_sqrt*tdpot.pot
    
    return matrix2solve, rhs2solve, x0

def scale_back_solution(inv_diag_sqrt, sol, rhs=None):
    sol = inv_diag_sqrt * sol
    if rhs is not None:
        rhs = inv_diag_sqrt * rhs

def implicit_block_diag(matrices):
    """
    Given a list of matrices (for example [A0,A1,A1,A2,A2,A0]) 
    returns the block diagonal matrix as 
    an implicit scipy LinearOperator object
    (in the example, the returned matrix is
    [A0         ]
    [  A1       ]
    [    A1     ]
    [      A2   ]
    [       A2  ]
    [         A0])
    """
    import scipy.sparse.linalg as sp
    import scipy.sparse as sps
    import numpy as np
    n_row = sum([m.shape[0] for m in matrices])
    n_col = sum([m.shape[1] for m in matrices])
    def matvec(x):
        y = np.zeros(n_row)
        begin_row = -matrices[0].shape[0]
        end_row = 0
        begin_col = -matrices[0].shape[1]
        end_col = 0
        for m in matrices:
            begin_row += m.shape[0]
            end_row += m.shape[0]
            begin_col += m.shape[1]
            end_col += m.shape[1]
            y[begin_row:end_row] = m.dot(x[begin_col:end_col])        
        return y
    def rmatvec(x):
        y = np.zeros(n_col)
        begin_row = -matrices[0].shape[1]
        end_row = 0
        begin_col = -matrices[0].shape[0]
        end_col = 0
        for m in matrices:
            begin_row += m.shape[1]
            end_row += m.shape[1]
            begin_col += m.shape[0]
            end_col += m.shape[0]
            y[begin_row:end_row] = m.T.dot(x[begin_col:end_col])        
        return y
    return sp.LinearOperator((n_row,n_col), matvec=matvec, rmatvec=rmatvec)

class controls_linalg:
    """
    Class for storing all controls for linear solvers
    """
    def __init__(self,
                 approach = 'bicgstab',
                 max_iterations = 100,
                 tolerance = 1e-6,
                 verbose = 0):
        if ( approach not in [
                'bicgstab',
                'pcg',
                'gmres']
        ):
            print('Linear solver method not supported. Passed :',str(approach))
            return
        else:
            self.approach = approach
            self.max_iterations = max_iterations
            self.tolerance = 0.0

    def __str__(self):
        strout=(str(self.approach)+':'+
                'tol=',str('{:.2E}'.format(self.tolerance))+','
                'max iter.=',+str('{:.2E}'.format(self.max_iterations)))
        return strout;

def identity_apply(x):
    return x
        
    
class SaddlePointPreconditioner():
    def __init__(self, A,B1T,B2,C):
        self.A = A
        self.B1T = B1T
        self.B2 = B2
        self.C = C

    def setup(approach='primal', prec_type='full',
              inv_SchurPrimal=None,inv_C=None,
              inv_A=None, inv_SchurDual=None):
        self.approach = approach
        self.prec_type = prec_type
        if approach == 'primal':
            if inv_SchurPrimal is None:
                exit()
            else:
                self.inv_SchurPrimal = inv_SchurPrimal
            
            if inv_C is None:
                exit()
            else:
                self.inv_C = inv_C
        elif approach == 'dual':
            if inv_SchurDual is None:
                exit()
            else:
                self.inv_SchurDual = inv_SchurDual
            
            if inv_A is None:
                exit()
            else:
                self.inv_A = inv_A
        else:
            exit()

        prec = lambda x: self.apply(x)
        return splinalg.LinearOperator((n+m,n+m), prec)
  
            
    def apply(x):
        n = self.n
        m = self.m
        y = np.zeros(n+m)
        
        if self.approach == 'primal' and self.prec_type == 'diag':
            y[0:n] = self.inv_SchurPrimal(x[0:n])
            y[n:n+m] = self.inv_C(x[n:n+m])
        
        if self.approach == 'primal' and self.prec_type == 'full':
            # v = C^{-1} x_m
            self.temp[n:n+m] = self.inv_C(x[n:n+m])
            # w = B1T * C ^{-1} x_m 
            self.temp2[0:n] = self.B1T(self.temp[n:n+m])
            self.temp2[0:n] += x[0:n]
            y[0:n] = self.inv_SchurPrimal(self.temp2[0:n])
            
        if self.approach == 'primal' and self.prec_type == 'lower':
            pass
        if self.approach == 'primal' and self.prec_type == 'upper':
            pass
            
        if self.approach == 'dual' and self.prec_type == 'diag':
            y[0:n] = self.inv_A(x[0:n])
            y[n:n+m] = self.inv_SchurDual(x[n:n+m])
            
        if self.approach == 'dual' and self.prec_type == 'full':
            pass
        if self.approach == 'dual' and self.prec_type == 'lower':
            pass
        if self.approach == 'dual' and self.prec_type == 'upper':
            pass

def get_info(ksp):
    """
    Return infos
    """
    reason = ksp.getConvergedReason()
    iterations = ksp.getIterationNumber() 
    h=ksp.getConvergenceHistory()
    resvec = h[-(iterations+1):]
    last_pres = ksp.getResidualNorm()
    return [reason,iterations,resvec,last_pres]

def info_ksp(ksp):
    """
    Return a one-line string with main info about last linear system solved
    """
    info = get_info(ksp)
    reason,iterations,residuals,last_pres = info
    last_pres = ksp.getResidualNorm()
    return str(ksp.getOptionsPrefix())+f' {KSPReasons[reason]} {iterations} {residuals[0]:.1e} {residuals[-1]:.1e} pres {last_pres:.1e}' 
    
    
        
class KrylovSolver():
    def __init__(self, matrix, ctrl, prec = None):
        default_ctrl = {
            'ksp_type':'gmres',
            'ksp_max_it':100,
            'ksp_rtol':1e-6,
            'ksp_atol':1e-50,
            'ksp_initial_guess_nonzero': False,
            #'ksp_monitor': False,
            'pc_side':'left',
            'pc_type':'ilu',
            'pc_factor_level':1,
            'pc_factor_fill':6,
        }   
        self.nrow = matrix.shape[0]
        self.ncol = matrix.shape[1]
        self.matrix = matrix
        self.rhs = rhs
        self.ctrl = ctrl
        if (prec is None):
            if ( ctrl['pc_type'] == 'jacobi'):
                self.inv_diag = 1.0 / self.matrix.diagonal()
                def prec_left(x):
                    return self.inv_diag * x
                self.prec = splinalg.LinearOperator((self.n_row,self.n_col),matvec=lambda x: self.inv_diag * x)
            if ( ctrl['pc_type'] == 'ilu'):
                self.ilu = splinalg.spilu(self.matrix,
                             drop_tol=ctrl.outer_prec_drop_tolerance,
                       fill_factor=ctrl['pc_factor_fill'])
                self.prec = splinalg.LinearOperator((self.n_row,self.n_col),matvec=self.ilu.solve)
        

        if ctrl['ksp_pc_side'] == 'left':
            self.prec_left = prec
            self.prec_right = None
        elif ctrl['ksp_pc_side'] == 'right':
            self.prec_left = None
            self.prec_right = prec

        self.info = info_linalg()
        self.solution = None
        self.time = 0.0
        self.iter = 0
        self.residual = 0.0

    def solve(self, rhs, x):
        # set initial guess
        if (self.ctrl['ksp_initial_guess_nonzero'] == False):
            x[:] = 0.0

        if ( self.ctrl['ksp_type'] == 'preonly'):
            if self.prec_left is not None:
                return self.prec(self.rhs)
            
        if ( self.ctrl.approach == 'bicgstab'):
            [x,ierr] = splinalg.bicgstab(
                self.matrix, rhs, x0=x,
                tol = self.ctrl['ksp_rtol'], 
                maxiter=self.ctrl['ksp_max_it'],
                atol=self.ctrl['ksp_atol'],
                M=self.prec)
        if ( self.ctrl.approach == 'cg'):
            [x,ierr] = splinalg.cg(
                self.matrix, rhs, x0=x,
                tol = self.ctrl['ksp_rtol'], 
                maxiter=self.ctrl['ksp_max_it'],
                atol=self.ctrl['ksp_atol'],
                M=self.prec)
        if ( self.ctrl.approach == 'gmres'):
            [x,ierr] = splinalg.gmres(
                self.matrix, rhs, x0=x,
                tol = self.ctrl['ksp_rtol'], 
                maxiter=self.ctrl['ksp_max_it'],
                atol=self.ctrl['ksp_atol'],
                M=self.prec)

    def asOperator(self):
        return splinalg.LinearOperator((self.n_row,self.n_col),matvec=lambda x: self.solve(rhs))
        

        
        # else:
        #     inode = 1
        #     grounding = True
        #     if (grounding):
        #         stiff[inode,inode] = 1.0e20
        #         for i_rhs in range(self.problem.n_rhs):
        #             rhs[inode+i_rhs*self.n_pot] = 0.0
            
        #     stiff=stiff

        #     # create block diagonal matrix
        #     matrix2solve = stiff
        #     if (self.problem.n_rhs>1):
        #         matrix2solve = sp.sparse.block_diag([matrix2solve]*problem.n_rhs)


        #     start_time = cputiming.time()
        #     ilu = splinalg.spilu(matrix2solve,
        #                      drop_tol=ctrl.outer_prec_drop_tolerance,
        #                      fill_factor=ctrl.outer_prec_fillin)
        #     if (ctrl.verbose>2):
        #         print('ILU'+'{:.2f}'.format(-(start_time - cputiming.time()))) 
        #     prec = lambda x: ilu.solve(x)
        #     M = splinalg.LinearOperator((n_equ,n_equ), prec)
            
        #     # solve linear system
        #     start_time = cputiming.time()
        #     [pot,ierr] = splinalg.bicgstab(
        #         matrix2solve, rhs, x0=tdpot.pot,
        #         tol=ctrl.tolerance_nonlinear, #restart=20,
        #         maxiter=ctrl.max_linear_iterations,
        #         atol=1e-16,
        #         M=M)
        
        #     tdpot.pot[:]=pot[:]
