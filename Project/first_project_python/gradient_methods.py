import numpy as np
from line_search import GoldenSection
class FletcherReeves:
    def __init__(self, f, grad_f, tol_FR=1e-6, tol_GS=1e-6, N=100, order=2):
        self.f = f
        self.grad_f = grad_f
        self.tol_FR = tol_FR
        self.tol_GS = tol_GS
        self.N = N
        self.order = order
    
    def optimize(self, x0):
        X = [x0]
        c = [self.grad_f(x0)]
        func_eval = 1
        n = len(x0)
        optimum_point = x0
        optimum_value = self.f(x0)
        k = 0
        beta = np.eye(n)
        
        while np.linalg.norm(c[k]) > self.tol_FR and k < 100:
            if k % n == 0 or k == 1:
                d = -c[k]
            else:
                beta_k = (np.linalg.norm(c[k]) / np.linalg.norm(c[k-1]))**2
                d = -c[k] + beta_k * d
            
            if np.linalg.norm(d) > 1:
                d /= np.linalg.norm(d)
                
            if self.order > 1:
                # Use Golden Section method for line search
                alfa_final, _ = GoldenSection(self.f, X[k], d).optimize()   
                alfa = alfa_final
            else:
                # Use direct solution for alpha
                alfa = -np.dot(c[k], d) / np.dot(np.dot(d.T, np.array(self.hessian(X[k]))), d)
            
            X.append(X[k] + alfa * d)
            c.append(self.grad_f(X[k+1]))
            func_eval += 1
            k += 1
        
        optimum_point = X[k]
        optimum_value = self.f(optimum_point)
        
        return optimum_point, func_eval