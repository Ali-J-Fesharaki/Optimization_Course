import numpy as np
import pandas as pd
from line_search import GoldenSection, QuadraticCurveFitting
from functions import f_1, grad_f1, f_2, grad_f2
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

