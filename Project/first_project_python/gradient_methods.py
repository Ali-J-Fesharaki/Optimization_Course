import numpy as np
import pandas as pd
from line_search import GoldenSection , QuadraticCurveFitting
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
class FletcherReeves:
    def __init__(self, f, grad_f=None, tol=1e-4,tol_ls=1e-4 ,max_iter=1000, stopping_criteria='point_diff', optimizer_name='FletcherReeves', line_search_name='GoldenSection',function_name='f'):
        self.f = FunctionWithEvalCounter(f)
        self.grad_f = FunctionWithEvalCounter(grad_f)
        self.tol = tol
        self.tol_ls = tol_ls
        self.max_iter = max_iter
        self.stopping_criteria = stopping_criteria
        self.LS_function_evaluation=0
        self.optimizer_name = optimizer_name
        self.function_name=function_name
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

            if  k == 1 or k% n == 0:
                d = -C[k]
            try:
                beta_k = (np.linalg.norm(C[k]) / np.linalg.norm(C[k-1]))**2
                d = -C[k] + beta_k * d
            except:
                print("Division by zero")
                beta_k = (np.linalg.norm(C[k-1]) / np.linalg.norm(C[k-2]))**2
                d = -C[k] + beta_k * d

            #line search
            golden_section = self.line_search_method(lambda alpha: self.f(X[k] + alpha * d),tol=self.tol_ls)
            alfa, _ = golden_section.optimize()
            self.LS_function_evaluation+=_
            X.append(X[k] + alfa * d)
            C.append(self.grad_f(X[k+1]))
            k += 1
            self.logger.log(iteration=k, point=X[k], func_eval=self.f.get_eval_count(),line_search_evals=self.LS_function_evaluation)

        optimum_point = X[k]
        optimum_value = self.f(optimum_point)
        func_eval = self.f.get_eval_count() 
        self.logger.save_to_file()
        return optimum_point, optimum_value, func_eval, self.LS_function_evaluation, k, self.logger.get_dataframe()

import numpy as np
from line_search import QuadraticCurveFitting