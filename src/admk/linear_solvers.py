from scipy.linalg import norm 
import scipy.sparse.linalg as splinalg
#from scipy import linalg
import time as cputiming
import os


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
        
    
# class sparse_solver:
#     """
#     Class for the application of inverse of sparse matrices
#     """
#     def __init__(self,matrix,ctrl):
#         if ( ctrl.approach == 'direct' ):
            
#         if ( ctrl.approach == 'incomplete' ):
        


#     def apply(self,rhs):

#         return solution

#     def kill(self):

#ilu_prec = KrylovSolver(A,[0,1,1,2,2,0]{'ksp_tpye':'preconly','pc_type':'ilu'})
#ilu_block = implicit_blk_diag([ilu_prec],[range()])[0,1,1,2,2,0]


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
        

        
