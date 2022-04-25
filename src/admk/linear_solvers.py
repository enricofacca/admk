from scipy.linalg import norm 
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
        
def apply_iterative_solver(matrix, rhs, ctrl, prec_left=None, prec_right=None):
    nequ = len(rh)

    info = info_linalg()
    
    if ( prec_left == None):
        prec_left = sp.sparse.linalg.LinearOperator(matvec=identity, shape=(nequ, nequ), dtype=float)
    if ( prec_right == None):
        prec_left = sp.sparse.linalg.LinearOperator(matvec=identity, shape=(nequ, nequ), dtype=float)

    if ( ctrl.approach  == 'bicgstab'):
        sol,info.ierr = sp.sparse.linalg.cg(jacobian_reduced, rhs_reduced,x0=np.zeros(npot),
                                                tol=tol_linear_newton,atol=0.0,maxiter=max_linear_iteration,
                                                M=prec_left,
                                                
                                                callback=info_solver.addone)
        
    #if ( ctrl.approach  == 'gmres'):
        
