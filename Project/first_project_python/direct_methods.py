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
class Powell:
    def __init__(self, f, grad_f=None, tol=1e-4,tol_ls=1e-4,max_iter=100, stopping_criteria='point_diff', optimizer_name='Powell', line_search_name='GoldenSection',function_name='f'):
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

        self.logger = OptimizationLogger(self.optimizer_name, self.function_name, self.line_search_name, x0)

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
        self.logger.save_to_file()
        return optimum_point,optimum_value , func_eval, self.ls_fe, k, self.logger.get_dataframe()



class NelderMead:
    def __init__(self, f, grad_f=None, tol=1e-4,tol_ls=1e-4,max_iter=1000, stopping_criteria='point_diff', optimizer_name='NelderMead', line_search_name='None',function_name='f'):
        self.f = FunctionWithEvalCounter(f)
        self.tol = tol
        self.max_iter = max_iter
        self.alfa_Q = -0.5
        self.delta = 0.5
        self.alfa_E = 3
        self.alfa_R = 1
        self.optimizer_name = optimizer_name
        self.function_name = function_name
        self.line_search_name = 'None'  # Nelder-Mead does not use line search
        self.LS_function_evaluation=0
        self.f_values = []
        self.operation=''
        self.simplex_list=[]

    def optimize(self, x0):
        self.f.reset()
        self.logger = OptimizationLogger(self.optimizer_name, self.function_name, self.line_search_name, x0)
        
        n = len(x0)
        simplex = np.zeros((n , n+1))
        simplex[:, 0] = x0
        for i in range(1, n+1):
            simplex[:, i] = x0 +self.delta * np.eye(n)[i-1]
        
        self.f_values = np.apply_along_axis(self.f, 0, simplex)
        iter_count = 0
        xc = np.mean(simplex[:,:-1], axis=1)

        while iter_count < self.max_iter:
            indices = np.argsort(self.f_values)
            simplex = simplex[:,indices]

            #self.simplex_list.append(simplex) 
            #print (self.simplex_list)

            self.f_values = self.f_values[indices]



            xc = np.mean(simplex[:,:-1], axis=1)
            xr = (1+self.alfa_R)*xc - self.alfa_R*simplex[:,-1]
            fxr = self.f(xr)
            self.operation='reflection'
            self.logger.log(iteration=iter_count, point=simplex[:,0], func_eval=self.f.get_eval_count(), operation=self.operation)
            dist=0
            for i in range (n+1):
                dist += np.linalg.norm(simplex[:,i]-xc)
            if (dist/(n+1))<self.tol:
                break
            if self.f_values[0] <= fxr and fxr< self.f_values[-2]:
                simplex[:,-1] = xr
                self.f_values[-1] = fxr
            elif fxr < self.f_values[0]:
                xe =(1+self.alfa_E)*xc - self.alfa_E*simplex[:,-1]
                fxe = self.f(xe)
                if fxe < fxr:
                    simplex[:,-1] = xe
                    self.f_values[-1] = fxe
                    self.operation='expansion'
                    self.logger.log(iteration=iter_count, point=simplex[:,0], func_eval=self.f.get_eval_count(), operation=self.operation)

                else:
                    simplex[:,-1] = xr
                    self.f_values[-1] = fxr
            elif self.f_values[-2]<=fxr and fxr < self.f_values[-1]:
                xq = (1+self.alfa_Q)*xr - self.alfa_Q*xc
                fxq = self.f(xq)
                if fxq < fxr:
                    simplex[:,-1] = xq
                    self.f_values[-1] = fxq
                    self.operation='outside_contraction'  
                    self.logger.log(iteration=iter_count, point=simplex[:,0], func_eval=self.f.get_eval_count(), operation=self.operation)
                else:
                    for i in range(1, len(simplex)):
                        simplex[:,i] = simplex[:,0] + self.delta * (simplex[:,i] - simplex[:,0])
                    self.f_values = np.apply_along_axis(self.f, 0, simplex)
                    self.operation='shrinkage'
                    self.logger.log(iteration=iter_count, point=simplex[:,0], func_eval=self.f.get_eval_count(), operation=self.operation)
            else:
                xq=(1+self.alfa_Q)*xc-self.alfa_Q*simplex[:,-1]
                fxq=self.f(xq)
                if fxq<self.f_values[-1]:
                    simplex[:,-1]=xq
                    self.f_values[-1]=fxq
                    self.operation='inside_contraction'
                    self.logger.log(iteration=iter_count, point=simplex[:,0], func_eval=self.f.get_eval_count(), operation=self.operation)

                else:
                    for i in range(1, len(simplex)):
                        simplex[:,i] = simplex[:,0] + self.delta * (simplex[:,i] - simplex[:,0])
                    self.f_values = np.apply_along_axis(self.f, 0, simplex)
                    self.operation='shrinkage'
                    self.logger.log(iteration=iter_count, point=simplex[:,0], func_eval=self.f.get_eval_count(), operation=self.operation)


            iter_count += 1
            

        optimum_point = simplex[:,0]
        optimum_value = self.f(optimum_point)
        func_eval = self.f.get_eval_count()
        self.logger.save_to_file()
        return optimum_point, optimum_value, func_eval, self.LS_function_evaluation,iter_count, self.logger.get_dataframe()