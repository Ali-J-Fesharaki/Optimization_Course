import numpy as np
from line_search import GoldenSection

class BFGS:
    def __init__(self, f, grad_f, tol_MvOp=1e-6, tol_LsOp=1e-6, N=100, order=2):
        self.f = f
        self.grad_f = grad_f
        self.tol_MvOp = tol_MvOp
        self.tol_LsOp = tol_LsOp
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
        beta = [np.eye(n)]
        
        while np.linalg.norm(c[k]) > self.tol_MvOp and k < 100:
            v = np.linalg.solve(beta[k], -c[k])
            d = v / np.linalg.norm(v)
            
            if self.order > 1:
                alfa_final, _ = GoldenSection(self.f,d)
                alfa = alfa_final
            else:
                alfa = -np.dot(c[k], d) / np.dot(np.dot(d.T, np.array(self.hessian(X[k]))), d)
                
            X.append(X[k] + alfa * d)
            c.append(self.grad_f(X[k+1]))
            func_eval += 1
            
            p = X[k+1] - X[k]
            q = c[k+1] - c[k]
            
            if k % n == 0:
                beta.append(np.eye(n))
            
            D = np.outer(q, q) / np.dot(q, p)
            E = np.dot(np.dot(beta[k], p), np.dot(p.T, beta[k])) / np.dot(np.dot(p.T, beta[k]), p)
            beta.append(beta[k] + D - E)
            
            k += 1
        
        optimum_point = X[k]
        optimum_value = self.f(optimum_point)
        
        return optimum_point, func_eval
    
    def hessian(self, x):
        hess = np.zeros((len(x), len(x)))
        for i in range(len(x)):
            for j in range(len(x)):
                hess[i, j] = np.gradient(np.gradient(self.f(x), x[j]), x[i])
        return hess
   
import numpy as np

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
                alfa_final, _ = GoldenSection(self.f, X[k], d)
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
    
import numpy as np

# Define a quadratic function
def quadratic(x):
    return np.dot(x, x)

# Define the gradient of the quadratic function
def grad_quadratic(x):
    return 2 * x

if __name__ == "__main__":
    # Initialize BFGS optimizer with the quadratic function and its gradient
    optimizer = BFGS(quadratic, grad_quadratic)

    # Initial guess
    x0 = np.array([1.0, 2.0])

    # Optimize the quadratic function
    optimum_point, optimum_value, func_eval = optimizer.optimize(x0)

    # Print results
    print("Optimal Point:", optimum_point)
    print("Optimal Value:", optimum_value)
    print("Function Evaluations:", func_eval)