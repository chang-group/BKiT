import numpy as np


class KineticsQK:
   
    def __init__(self, trans_mat, ave_time, nM, bc = ['reflective', 'absorbing']):
        """
        
        """
        
        self.nM = nM
        self.Tmat = self.BoundaryConditions(trans_mat, bc[0], bc[1])       
        self.ave_time = ave_time
        self.const = 8.314*298 / 4184.0  # energy units in kcal/mol
    
    
    def BoundaryConditions(self, trans_mat, bc_beg, bc_end):
        """
        
        """

        kernelsel = trans_mat.copy()
        
        #boundary condition at the end
        if bc_end == 'absorbing':
            kernelsel[self.nM-1, self.nM-2] = 0.0
            #kernelsel[self.nM-2, self.nM-1] = 0.0
        elif bc_end == 'reflective':
            kernelsel[self.nM-1, self.nM-2] = 1.0
            #kernelsel[self.nM-2, self.nM-1] = 1.0
        else:
            pass
        
        #bc condition at the beginning
        if bc_beg == 'absorbing':
            #kernelsel[1, 0] = 1.0
            kernelsel[0, 1] = 0.0
        elif bc_beg == 'reflective':
            #kernelsel[1, 0] = 0.0
            kernelsel[0, 1] = 1.0
        else:
            pass
        
        return kernelsel
    
    
    def PowerIter(self, n_iters):
        """
        
        """
        q = np.random.rand(self.nM)
        for i in range(n_iters):
            q = np.dot(q, self.Tmat) 
            q /= np.linalg.norm(q)

        PMF = - self.const * np.log(np.abs(q) * self.ave_time)
        return PMF, q
    
    def EigenSolver(self):
        """
        Eigenvector of the highest eigenvalue (1.0) happens to be stationary flux
        """
        evals, evecs = np.linalg.eig(self.Tmat.T)
        index_max = evals.argmax()
        q = evecs[:, index_max] #get eigenvec of highest eigenval       
        PMF = - self.const * np.log(np.abs(q) * self.ave_time) 

        return PMF, q
    
class KineticsPK:
   
    def __init__(self, rate_mat, nM, bc = ['reflective', 'absorbing']):
        """
        
        """
        
        self.nM = nM
        self.Rmat = self.BoundaryConditions(rate_mat, bc[0], bc[1])       
        self.const = 8.314*298 / 4184.0  # energy units in kcal/mol
    
    
    def BoundaryConditions(self, trans_mat, bc_beg, bc_end):
        """
        
        """

if __name__=='__main__':

    print("Testing Solvers" )
    