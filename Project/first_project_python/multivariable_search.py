import numpy as np
from line_search import GoldenSection
class FunctionWithEvalCounter:
    def __init__(self, f):
        self.f = f
        self.eval_count = 0

    def __call__(self, *args, **kwargs):
        self.eval_count += 1
        return self.f(*args, **kwargs)

    def reset(self):
        self.eval_count = 0

    def get_eval_count(self):
        return self.eval_count
class BFGS:
    def __init__(self, f, grad_f, tol=1e-6, tol_ls=1e-6, max_iter=100, stopping_criteria='gradient_norm'):
        self.f = FunctionWithEvalCounter(f)
        self.grad_f = FunctionWithEvalCounter(grad_f)
        self.tol = tol
        self.tol_ls = tol_ls
        self.max_iter = max_iter
        self.stopping_criteria = stopping_criteria
        self.LS_function_evaluation=0
    def optimize(self, x0):
        self.f.reset()
        self.grad_f.reset()

        X = [x0]
        C = [self.grad_f(x0)]
        n = len(x0)
        beta = [np.eye(n)]
        k = 0

        while k < self.max_iter:
            grad_norm = np.linalg.norm(C[k])
            if self.stopping_criteria == 'gradient_norm' and grad_norm < self.tol:
                break
            if self.stopping_criteria == 'point_diff' and k > 0 and np.linalg.norm(X[k] - X[k-1]) < self.tol:
                break

            v = np.linalg.solve(beta[k], -C[k])
            d = v / np.linalg.norm(v)
            
            #line search
            golden_section = QuadraticCurveFitting(lambda alpha: self.f(X[k] + alpha * d))
            alfa, _ = golden_section.optimize()
            self.LS_function_evaluation+=_
                
            X.append(X[k] + alfa * d)
            C.append(self.grad_f(X[k+1]))

            p = X[k+1] - X[k]
            q = C[k+1] - C[k]

            if k % n == 0:
                beta.append(np.eye(n))

            D = np.outer(q, q) / np.dot(q, p)
            E = np.dot(np.dot(beta[k], p), np.dot(p.T, beta[k])) / np.dot(np.dot(p.T, beta[k]), p)
            beta.append(beta[k] + D - E)

            k += 1

        optimum_point = X[k]
        optimum_value = self.f(optimum_point)
        func_eval = self.f.get_eval_count() 
        return optimum_point, optimum_value, func_eval ,self.LS_function_evaluation,k
    
    def hessian(self, x):
        hess = np.zeros((len(x), len(x)))
        for i in range(len(x)):
            for j in range(len(x)):
                hess[i, j] = np.gradient(np.gradient(self.f(x), x[j]), x[i])
        return hess


class FletcherReeves:
    def __init__(self, f, grad_f, tol=1e-6, tol_ls=1e-6, max_iter=100, stopping_criteria='gradient_norm'):
        self.f = FunctionWithEvalCounter(f)
        self.grad_f = FunctionWithEvalCounter(grad_f)
        self.tol = tol
        self.tol_ls = tol_ls
        self.max_iter = max_iter
        self.stopping_criteria = stopping_criteria
        self.LS_function_evaluation=0

    def optimize(self, x0):
        self.f.reset()
        self.grad_f.reset()

        X = [x0]
        C = [self.grad_f(x0)]
        n = len(x0)
        k = 0
        d = -C[k]

        while k < self.max_iter:
            grad_norm = np.linalg.norm(C[k])
            if self.stopping_criteria == 'gradient_norm' and grad_norm < self.tol:
                break
            if self.stopping_criteria == 'point_diff' and k > 0 and np.linalg.norm(X[k] - X[k-1]) < self.tol:
                break

            if  k == 1:
                d = -C[k]
            try:
                beta_k = (np.linalg.norm(C[k]) / np.linalg.norm(C[k-1]))**2
                d = -C[k] + beta_k * d
            except:
                print("Division by zero")
                beta_k = (np.linalg.norm(C[k-1]) / np.linalg.norm(C[k-2]))**2
                d = -C[k] + beta_k * d

            #line search
            golden_section = GoldenSection(lambda alpha: self.f(X[k] + alpha * d))
            alfa, _ = golden_section.optimize()
            self.LS_function_evaluation+=_
            X.append(X[k] + alfa * d)
            C.append(self.grad_f(X[k+1]))
            k += 1

        optimum_point = X[k]
        optimum_value = self.f(optimum_point)
        func_eval = self.f.get_eval_count() 
        return optimum_point, optimum_value, func_eval,self.LS_function_evaluation,k

import numpy as np
from line_search import QuadraticCurveFitting

class Powell:
    def __init__(self, f, tol=1e-6, max_iter=100, stopping_criteria='gradient_norm'):
        self.f = FunctionWithEvalCounter(f)
        self.tol = tol
        self.max_iter = max_iter
        self.stopping_criteria = stopping_criteria

    def optimize(self, x0):
        self.f.reset()

        n = len(x0)
        X = x0.copy()
        U = np.eye(n)
        iter_count = 0

        for k in range(self.max_iter):
            iter_count += 1
            X_prev = X.copy()
            for i in range(n):
                d = U[:, i]
                golden_section = QuadraticCurveFitting(lambda alpha: self.f(X[k] + alpha * d))
                alfa, _ = golden_section.optimize()
                X = X + alfa * d

            d = X - X_prev
            golden_section = QuadraticCurveFitting(lambda alpha: self.f(X[k] + alpha * d))
            alfa, _ = golden_section.optimize()
            X_new = X + alfa * d

            if self.stopping_criteria == 'point_diff' and np.linalg.norm(X_new - X_prev) < self.tol:
                break
            if self.stopping_criteria == 'gradient_norm' and np.linalg.norm(X_new - X_prev) < self.tol:
                break

            U[:, 0:n-1] = U[:, 1:n]
            U[:, -1] = d / np.linalg.norm(d)
            X = X_new

        optimum_point = X
        optimum_value = self.f(X)
        func_eval = self.f.get_eval_count()

        return optimum_point, optimum_value, func_eval

import numpy as np

class NelderMead:
    def __init__(self, f, tol=1e-6, max_iter=100, stopping_criteria='point_diff'):
        self.f = FunctionWithEvalCounter(f)
        self.tol = tol
        self.max_iter = max_iter
        self.stopping_criteria = stopping_criteria

    def optimize(self, x0):
        self.f.reset()

        n = len(x0)
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5

        # Initial simplex
        simplex = [x0]
        for i in range(n):
            x = x0.copy()
            x[i] = x0[i] + 0.05 if x0[i] != 0 else 0.00025
            simplex.append(x)

        simplex = np.array(simplex)

        for k in range(self.max_iter):
            simplex = sorted(simplex, key=lambda x: self.f(x))

            if self.stopping_criteria == 'point_diff' and np.max(np.abs(simplex[0] - simplex[1:])) < self.tol:
                break

            x0 = np.mean(simplex[:-1], axis=0)
            xr = x0 + alpha * (x0 - simplex[-1])
            fr = self.f(xr)

            if self.f(simplex[0]) <= fr < self.f(simplex[-2]):
                simplex[-1] = xr
            elif fr < self.f(simplex[0]):
                xe = x0 + gamma * (xr - x0)
                fe = self.f(xe)
                if fe < fr:
                    simplex[-1] = xe
                else:
                    simplex[-1] = xr
            else:
                xc = x0 + rho * (simplex[-1] - x0)
                fc = self.f(xc)
                if fc < self.f(simplex[-1]):
                    simplex[-1] = xc
                else:
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])

            if self.stopping_criteria == 'gradient_norm' and np.max(np.abs(simplex[0] - simplex[1:])) < self.tol:
                break

        optimum_point = simplex[0]
        optimum_value = self.f(optimum_point)
        func_eval = self.f.get_eval_count()

        return optimum_point, optimum_value, func_eval
# Example function to optimize
def quadratic(x):
    return np.dot(x, x)

# Example gradient of the quadratic function
def grad_quadratic(x):
    return 2 * x

if __name__ == "__main__":
    optimizer = BFGS(quadratic, grad_quadratic, stopping_criteria='point_diff')
    x0 = np.array([100.0, 100.0])
    optimum_point, optimum_value, func_eval,ls_fe,k = optimizer.optimize(x0)
    print("Optimum Point:", optimum_point)
    print("Optimum Value:", optimum_value)
    print("Function Evaluations:", func_eval)
    print("Line Search Function Evaluations:", ls_fe)
    print("Iterations:",k)


