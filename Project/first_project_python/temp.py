import numpy as np
import sympy as sp
import pandas as pd
from functions import f_1, grad_f1, f_2, grad_f2
from line_search import GoldenSection, QuadraticCurveFitting
def golden_search(f, tol, N):
    # Implement golden section search here
    # Placeholder implementation
    return 1.0, 0  # Return alpha and function evaluations count

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
    def construct_filename(self):
        initial_point_str = str(self.initial_point).replace(" ", ",")
        filename = f"{self.function_name}_{self.optimizer_name}_{self.line_search_name}_{initial_point_str}.csv"
        return filename.replace(" ", "_")

    def save_to_file(self):
        df = self.get_dataframe()
        filename = self.construct_filename()
        print(filename)
        df.to_csv(filename, index=False)
class Powell:
    def __init__(self, f, grad_f=None, tol=1e-4,tol_ls=1e-4 ,max_iter=100, stopping_criteria='point_diff', optimizer_name='Powell', line_search_name='GoldenSection',function_name='f'):
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
                for t in range(n-2, n):
                    d[:, t, cycle] = d[:, t+1, cycle-1]

                    if (cycle ) % n == 0:
                        s = np.eye(n)
                        d[:, 0, cycle] = s[:, -1]
                        for t in range(1, n+1):
                            d[:, t, cycle] = s[:, t-1]

                    if np.linalg.norm(d[:, t, cycle]) > 1:
                        d[:, t, cycle] = d[:, t, cycle] / np.linalg.norm(d[:, t, cycle])

                d[:, 0, cycle] = d[:, n, cycle]

            for i in range(n+1):
                i = i + j
                F = self.f(X[i])
                x_minus_ep = X[i] - ep * d[:, i-j, cycle]
                x_plus_ep = X[i] + ep * d[:, i-j, cycle]
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
            j = i
            if F_plus_ep > F and F_minus_ep > F and np.linalg.norm(X[i+1] - X[i]) < self.tol:
                print('Problem solved')
                break


        optimum_point = X[-1]
        k = cycle
        func_eval = self.f.get_eval_count()
        optimum_value = self.f(optimum_point)
        return optimum_point,optimum_value , func_eval, self.ls_fe, k, self.logger.get_dataframe()

if __name__ == "__main__":
    optimizer = Powell(f_2, grad_f2,line_search_name="GoldenSection",stopping_criteria='point_diff',max_iter=2000,function_name="f1")
    x0 = np.array([0 ,0])
    optimum_point, optimum_value, func_eval,ls_fe,k,df = optimizer.optimize(x0)
    print("Optimum Point:", optimum_point)
    print("Optimum Value:", optimum_value)
    print("Function Evaluations:", func_eval)
    print("Line Search Function Evaluations:", ls_fe)
    print("Iterations:",k)
    optimizer.logger.save_to_file()