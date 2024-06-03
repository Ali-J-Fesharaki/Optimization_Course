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
import numpy as np
import pandas as pd

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
    def __init__(self, optimizer_name, function_name, line_search_name, initial_point):
        self.optimizer_name = optimizer_name
        self.line_search_name = line_search_name
        self.initial_point = initial_point
        self.function_name = function_name  
        self.records = []

    def log(self, **kwargs):
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

class NelderMead:
    def __init__(self, f, tol=1e-12, max_iter=1000, optimizer_name='NelderMead', function_name='f'):
        self.f = FunctionWithEvalCounter(f)
        self.tol = tol
        self.max_iter = max_iter
        self.alpha_R = 1
        self.alpha_E = 2
        self.alpha_C = 0.5
        self.alpha_S = 0.5
        self.optimizer_name = optimizer_name
        self.function_name = function_name
        self.line_search_name = 'None'  # Nelder-Mead does not use line search
        self.simplex_list=[]

    def optimize(self, x0):
        self.f.reset()
        self.logger = OptimizationLogger(self.optimizer_name, self.function_name, self.line_search_name, x0)
        
        n = len(x0)
        simplex = np.zeros((n, n+1))
        simplex[:, 0] = x0
        for i in range(1, n+1):
            simplex[:, i] = x0 + self.alpha_S * np.eye(n)[i-1]
        
        f_values = np.apply_along_axis(self.f, 0, simplex)
        iter_count = 0
        prev_centroid = np.mean(simplex[:, :-1], axis=1)

        while iter_count < self.max_iter:
            indices = np.argsort(f_values)
            simplex = simplex[:, indices]
            f_values = f_values[indices]

            self.simplex_list.append(simplex) 
            print (self.simplex_list)

            xc = np.mean(simplex[:, :-1], axis=1)

            xr = xc + self.alpha_R * (xc - simplex[:, -1])
            fxr = self.f(xr)
            self.operation = 'reflection'
            dist=0
            for i in range(n):
                dist+=np.linalg.norm(simplex[:,i]-xc)
            dist/=n
            if dist<self.tol:
                break
            prev_centroid = np.mean(simplex[:, :-1], axis=1)
            if f_values[0] <= fxr < f_values[-2]:
                simplex[:, -1] = xr
                f_values[-1] = fxr
            elif fxr < f_values[0]:
                xe = xc + self.alpha_E * (xr - xc)
                fxe = self.f(xe)
                if fxe < fxr:
                    simplex[:, -1] = xe
                    f_values[-1] = fxe
                    self.operation = 'expansion'
                else:
                    simplex[:, -1] = xr
                    f_values[-1] = fxr
            else:
                if fxr < f_values[-1]:
                    xq = xc + self.alpha_C * (xr - xc)
                    fxq = self.f(xq)
                    if fxq < fxr:
                        simplex[:, -1] = xq
                        f_values[-1] = fxq
                        self.operation = 'outside_contraction'
                    else:
                        for i in range(1, n+1):
                            simplex[:, i] = simplex[:, 0] + self.alpha_S * (simplex[:, i] - simplex[:, 0])
                        f_values = np.apply_along_axis(self.f, 0, simplex)
                        self.operation = 'shrinkage'
                else:
                    xq = xc + self.alpha_C * (simplex[:, -1] - xc)
                    fxq = self.f(xq)
                    if fxq < f_values[-1]:
                        simplex[:, -1] = xq
                        f_values[-1] = fxq
                        self.operation = 'inside_contraction'
                    else:
                        for i in range(1, n+1):
                            simplex[:, i] = simplex[:, 0] + self.alpha_S * (simplex[:, i] - simplex[:, 0])
                        f_values = np.apply_along_axis(self.f, 0, simplex)
                        self.operation = 'shrinkage'

            iter_count += 1
            self.logger.log(iteration=iter_count, point=simplex[:, 0], func_eval=self.f.get_eval_count(), operation=self.operation)

            centroid_distance = np.linalg.norm(np.mean(simplex[:, :-1], axis=1) - prev_centroid)
            if centroid_distance < self.tol:
                break
            prev_centroid = np.mean(simplex[:, :-1], axis=1)

        optimum_point = simplex[:, 0]
        optimum_value = self.f(optimum_point)
        func_eval = self.f.get_eval_count()
        return optimum_point, optimum_value, func_eval, iter_count, self.logger.get_dataframe()

# Example usage:
def rosenbrock(x):
    return sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

nm = NelderMead(rosenbrock)
optimum_point, optimum_value, func_eval, iter_count, log_df = nm.optimize(np.array([0.0, 0.0]))

print(f"Optimum point: {optimum_point}")
print(f"Optimum value: {optimum_value}")
print(f"Function evaluations: {func_eval}")
print(f"Iterations: {iter_count}")
print(log_df)

from functions import f_1, grad_f1, f_2, grad_f2
if __name__ == "__main__":
    optimizer = NelderMead(f_2)
    x0 = np.array([0 ,0])
    optimum_point, optimum_value, func_eval, iter_count, log_df  = optimizer.optimize(x0)
    print("Optimum Point:", optimum_point)
    print("Optimum Value:", optimum_value)
    print("Function Evaluations:", func_eval)
    optimizer.logger.save_to_file()


    """optimizer = Powell(f_1, grad_f1,line_search_name="GoldenSection",stopping_criteria='point_diff',max_iter=2000,function_name="f1")
    x0 = np.array([0 ,0,0])
    optimum_point, optimum_value, func_eval,ls_fe,k,df = optimizer.optimize(x0)
    print("Optimum Point:", optimum_point)
    print("Optimum Value:", optimum_value)
    print("Function Evaluations:", func_eval)
    print("Line Search Function Evaluations:", ls_fe)
    print("Iterations:",k)
    optimizer.logger.save_to_file()"""