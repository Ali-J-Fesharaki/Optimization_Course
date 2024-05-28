import numpy as np

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
                alfa_final, _ = GoldenSection(self.f,d).optimize()
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
    
class DFP:
    def __init__(self, f, grad_f, tol_MvOp=1e-6, tol_Ls=1e-6, N=100, order=2):
        self.f = f
        self.grad_f = grad_f
        self.tol_MvOp = tol_MvOp
        self.tol_Ls = tol_Ls
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
        H = np.eye(n)
        
        while np.linalg.norm(c[k]) > self.tol_MvOp and k < 100:
            d = -np.dot(H, c[k])
            if np.linalg.norm(d) > 1:
                d /= np.linalg.norm(d)
                
            if self.order > 1:
                # Use Line Search method
                alfa_final, _ = GoldenSection(self.f, X[k], d).optimize()
                alfa = alfa_final
            else:
                # Use direct solution for alpha
                alfa = -np.dot(c[k], d) / np.dot(np.dot(d.T, np.array(self.hessian(X[k]))), d)
            
            X.append(X[k] + alfa * d)
            c.append(self.grad_f(X[k+1]))
            func_eval += 1
            
            p = X[k+1] - X[k]
            q = c[k+1] - c[k]

            if k % n == 0:
                H = np.eye(n)
            
            D = np.outer(p, p) / np.dot(p, q)
            E = np.dot(np.dot(H, q), np.dot(q.T, H)) / np.dot(np.dot(q.T, H), q)
            H += D - E
            
            k += 1
        
        optimum_point = X[k]
        optimum_value = self.f(optimum_point)
        
        return optimum_point, optimum_value, func_eval
    
    