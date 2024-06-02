import numpy as np
import pandas as pd
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
class OptimizationLogger:
    def __init__(self, optimizer_name, function_name,line_search_name, initial_point):
        self.optimizer_name = optimizer_name
        self.line_search_name = line_search_name
        self.initial_point = initial_point
        self.function_name = function_name  
        self.records = []

    def log(self,**kwargs):
        self.records.append({
            **kwargs
        })

    def get_dataframe(self):
        df = pd.DataFrame(self.records)
        df.attrs['function_name'] = self.function_name
        df.attrs['optimizer_name'] = self.optimizer_name
        df.attrs['line_search_name'] = self.line_search_name
        df.attrs['initial_point'] = self.initial_point
        return df

    def construct_filename(self):
        initial_point_str = str(self.initial_point).replace(" ", ",")
        filename = f"{self.function_name}_{self.optimizer_name}_{self.line_search_name}_{initial_point_str}.csv"
        return filename.replace(" ", "_")

    def save_to_file(self):
        df = self.get_dataframe()
        filename = self.construct_filename()
        print(filename)
        df.to_csv(filename, index=False)
class BFGS:
    def __init__(self, f, grad_f=None, tol=1e-4,tol_ls=1e-4,max_iter=1000, stopping_criteria='point_diff', optimizer_name='BFGS', line_search_name='GoldenSection',function_name='f'):
        self.f = FunctionWithEvalCounter(f)
        self.grad_f = FunctionWithEvalCounter(grad_f)
        self.tol = tol
        self.tol_ls = tol_ls
        self.max_iter = max_iter
        self.optimizer_name = optimizer_name
        self.line_search_name = line_search_name
        self.function_name=function_name
        if(self.line_search_name=='GoldenSection'):
            self.line_search_method=GoldenSection
        elif(self.line_search_name=='QuadraticCurveFitting'):
            self.line_search_method=QuadraticCurveFitting
        

        self.stopping_criteria = stopping_criteria
        self.LS_function_evaluation=0
    def optimize(self, x0):
        self.f.reset()
        self.grad_f.reset()
        self.logger = OptimizationLogger(self.optimizer_name,self.function_name, self.line_search_name, x0)

        X = [x0]
        C = [self.grad_f(x0)]
        n = len(x0)
        beta = [np.eye(n)]
        k = 0

        while k < self.max_iter:
            grad_norm = np.linalg.norm(C[k])
            if self.stopping_criteria == 'gradient_norm' and grad_norm < self.tol:
                break
            if self.stopping_criteria == 'point_diff' and k > 0 and np.linalg.norm(X[k] - X[k-1]) < self.tol and grad_norm < self.tol:
                break

            v = np.linalg.solve(beta[k], -C[k])
            d = v / np.linalg.norm(v)
            
            #line search
            golden_section =  self.line_search_method(lambda alpha: self.f(X[k] + alpha * d))
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
            self.logger.log(iteration=k, point=X[k], func_eval=self.f.get_eval_count(),line_search_evals=self.LS_function_evaluation)

        optimum_point = X[k]
        func_eval = self.f.get_eval_count() 
        optimum_value = self.f(optimum_point)
        return optimum_point, optimum_value, func_eval,self.LS_function_evaluation, k, self.logger.get_dataframe()  
      
    def hessian(self, x):
        hess = np.zeros((len(x), len(x)))
        for i in range(len(x)):
            for j in range(len(x)):
                hess[i, j] = np.gradient(np.gradient(self.f(x), x[j]), x[i])
        return hess


class FletcherReeves:
    def __init__(self, f, grad_f=None, tol=1e-6,tol_ls=1e-6 ,max_iter=1000, stopping_criteria='point_diff', optimizer_name='FletcherReeves', line_search_name='GoldenSection',function_name='f'):
        self.f = FunctionWithEvalCounter(f)
        self.grad_f = FunctionWithEvalCounter(grad_f)
        self.tol = tol
        self.tol_ls = tol_ls
        self.max_iter = max_iter
        self.stopping_criteria = stopping_criteria
        self.LS_function_evaluation=0
        self.optimizer_name = optimizer_name
        self.line_search_name = line_search_name
        if(self.line_search_name=='GoldenSection'):
            self.line_search_method=GoldenSection
        elif(self.line_search_name=='QuadraticCurveFitting'):
            self.line_search_method=QuadraticCurveFitting

    def optimize(self, x0):
        self.f.reset()
        self.grad_f.reset()
        self.f.reset()
        self.logger = OptimizationLogger(self.optimizer_name,self.function_name, self.line_search_name, x0)

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
            golden_section = self.line_search_method(lambda alpha: self.f(X[k] + alpha * d))
            alfa, _ = golden_section.optimize()
            self.LS_function_evaluation+=_
            X.append(X[k] + alfa * d)
            C.append(self.grad_f(X[k+1]))
            k += 1
            self.logger.log(iteration=k, point=X[k], func_eval=self.f.get_eval_count(),line_search_evals=self.LS_function_evaluation)

        optimum_point = X[k]
        optimum_value = self.f(optimum_point)
        func_eval = self.f.get_eval_count() 
        return optimum_point, optimum_value, func_eval, self.LS_function_evaluation, k, self.logger.get_dataframe()

import numpy as np
from line_search import QuadraticCurveFitting

class Powell:
    def __init__(self, f, grad_f=None, tol=1e-12,tol_ls=1e-12 ,max_iter=100, stopping_criteria='point_diff', optimizer_name='Powell', line_search_name='GoldenSection',function_name='f'):
        self.f = FunctionWithEvalCounter(f)
        self.tol = tol
        self.max_iter = max_iter
        self.stopping_criteria = stopping_criteria
        self.optimizer_name = optimizer_name
        self.line_search_name = line_search_name
        self.function_name=function_name
        self.LS_function_evaluation=0
        self.tol_ls=tol_ls
        self.logger = None
        self.ls_fe=0
        if(self.line_search_name=='GoldenSection'):
            self.line_search_method=GoldenSection
        elif(self.line_search_name=='QuadraticCurveFitting'):
            self.line_search_method=QuadraticCurveFitting
    def optimize(self, x0):
        self.f.reset()
        max_cycle = self.max_iter
        X = [x0]
        n = len(x0)
        j = 0
        ep = self.tol
        s = np.eye(n)
        d = np.zeros((n, n+1, max_cycle))
        d[:, 0, 0] = s[:, -1]
        for t in range(1, n+1):
            d[:, t, 0] = s[:, t-1]

        self.logger = OptimizationLogger('Powell', 'f', 'GoldenSection', x0)

        for cycle in range(max_cycle):
            print(f' *****  cycle: {cycle + 1} *****')
            if cycle > 0:
                k = j - n 
                s[:, 0] = X[ j] - X[ k]
                d[:, n, cycle] = s[:, 0]
                for t in range(1, n):
                    d[:, t, cycle] = d[:, t+1, cycle-1]

                    if (cycle ) % n == 0:
                        s = np.eye(n)
                        d[:, 0, cycle] = s[:, -1]
                        for t in range(1, n+1):
                            d[:, t, cycle] = s[:, t-1]

                    if np.linalg.norm(d[:, t, cycle]) > 1:
                        d[:, t, cycle] = d[:, t, cycle] / np.linalg.norm(d[:, t, cycle])

                d[:, 0, cycle] = d[:, n, cycle]
                D = d[:, 0, cycle]

            for i in range(n+1):
                i = i + j
                F = self.f(X[i])
                D=d[:, i-j, cycle]
                x_minus_ep = X[i] - ep * D
                x_plus_ep = X[i] + ep * D
                F_minus_ep = self.f(x_minus_ep)
                F_plus_ep = self.f(x_plus_ep)
                if F_plus_ep < F:
                    f_alpha = (lambda alpha: self.f(X [i] + alpha * d[:, i-j, cycle]))
                else:
                    f_alpha = (lambda alpha: self.f(X[ i] - alpha * d[:, i-j, cycle]))
  

                golden_section =  self.line_search_method(f_alpha)
                alfa, _ = golden_section.optimize()
                self.LS_function_evaluation+=_

                if F_plus_ep < F:
                    X.append(X[i] + alfa * d[:, i-j, cycle])
                else:
                    X.append( X[i] - alfa * d[:, i-j, cycle])

                if F_plus_ep > F and F_minus_ep > F and np.linalg.norm(X[i+1] - X[i]) < self.tol:
                    print('Problem solved')
                    break
            self.logger.log(iteration=i, point=X[i], func_eval=self.f.get_eval_count(),line_search_evals=self.LS_function_evaluation)
            j = i+1
            if F_plus_ep > F and F_minus_ep > F and np.linalg.norm(X[i+1] - X[i]) < self.tol:
                print('Problem solved')
                break


        optimum_point = X[-1]
        k = cycle
        func_eval = self.f.get_eval_count()
        optimum_value = self.f(optimum_point)
        return optimum_point,optimum_value , func_eval, self.ls_fe, k, self.logger.get_dataframe()



class NelderMead:
    def __init__(self, f, grad_f=None, tol=1e-12, max_iter=100, stopping_criteria='point_diff', optimizer_name='Nelder-Mead', line_search_name='None',function_name='f'):
        self.f = FunctionWithEvalCounter(f)
        self.tol = tol
        self.max_iter = max_iter
        self.stopping_criteria = stopping_criteria
        self.optimizer_name = optimizer_name
        self.function_name=function_name
        self.line_search_name = line_search_name
        self.logger = None
        self.ls_fe = 0

    def optimize(self, x0):
        self.f.reset()
        self.logger = OptimizationLogger(self.optimizer_name,self.function_name, self.line_search_name, x0)

        n = len(x0)
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        X=[x0]

        # Initial simplex
        simplex = [x0]
        for i in range(n):
            x = x0.copy()
            x[i] = x0[i] + 0.05 if x0[i] != 0 else 0.00025
            simplex.append(x)

        simplex = np.array(simplex)

        for k in range(self.max_iter):
            simplex = sorted(simplex, key=lambda x: self.f(x))
            point = None
            value = None
            if k % n == 0:
                point = simplex[0]
                value = self.f(point)

            x0 = np.mean(simplex[:-1], axis=0)
            xr = x0 + alpha * (x0 - simplex[-1])
            fr = self.f(xr)
            if self.f(simplex[0]) <= fr < self.f(simplex[-2]):
                simplex[-1] = xr
                operation='reflection'
            elif fr < self.f(simplex[0]):
                xe = x0 + gamma * (xr - x0)
                fe = self.f(xe)
                if fe < fr:
                    simplex[-1] = xe
                    operation='expansion'
                else:
                    simplex[-1] = xr
                    operation='reflection'
            else:
                xc = x0 + rho * (simplex[-1] - x0)
                fc = self.f(xc)
                if fc < self.f(simplex[-1]):
                    simplex[-1] = xc
                    operation='contraction'
                else:
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    operation='shrink'
            if self.stopping_criteria == 'point_diff' and np.linalg.norm(simplex[0] -X[k]) < self.tol:
                break
            X.append(simplex[0])

            self.logger.log(iteration=k, point=simplex[0], func_eval=self.f.get_eval_count(),operation=operation)

        optimum_point = simplex[0]
        optimum_value = self.f(optimum_point)
        func_eval = self.f.get_eval_count()

        return optimum_point, optimum_value, func_eval, self.ls_fe, k, self.logger.get_dataframe()




# Example function to optimize
def quadratic(x):
    return np.dot(x, x)

# Example gradient of the quadratic function
def grad_quadratic(x):
    return 2 * x
from functions import f_1, grad_f1, f_2, grad_f2
if __name__ == "__main__":
    optimizer = NelderMead(f_1, grad_f1,line_search_name="GoldenSection",stopping_criteria='point_diff',max_iter=5000,function_name="f1")
    x0 = np.array([0 ,0,0])
    optimum_point, optimum_value, func_eval,ls_fe,k,df = optimizer.optimize(x0)
    print("Optimum Point:", optimum_point)
    print("Optimum Value:", optimum_value)
    print("Function Evaluations:", func_eval)
    print("Line Search Function Evaluations:", ls_fe)
    print("Iterations:",k)
    optimizer.logger.save_to_file()


