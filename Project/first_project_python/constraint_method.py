from direct_methods import  simplex_method
import numpy as np


class quadratic_interior_extended_penalty:
    def __init__(self, f,ineq_constraints,eq_constraints,grad_f=None, tol=1e-4,tol_ls=1e-4,max_iter=1000,epsilon=1e-3, stopping_criteria='point_diff', optimizer_name='quadratic_extended_penalty', line_search_name='None',function_name='f'):
        self.f = f
        self.constraints = ineq_constraints
        self.grad_f = grad_f
        self.tol = tol
        self.tol_ls = tol_ls
        self.max_iter = max_iter
        self.stopping_criteria = stopping_criteria
        self.optimizer_name = optimizer_name
        self.line_search_name = line_search_name
        self.function_name = function_name
        self.epsilon = epsilon
        self.penalities = self.convert_constraint_to_penalty(ineq_constraints,self.epsilon)
        self.f_penalty = self.create_penalty_function

    def convert_constraint_to_penalty(self, ineq_constraints,epsilon):
        penalities = []
        for constraint in ineq_constraints:
            penalities.append(lambda x: -1/constraint(x)  if constraint(x) <= epsilon else ((-1/epsilon)*(constraint(x)/epsilon)**2-3*(constraint(x)/epsilon)+3))

        return penalities
    def create_penalty_function(self,x):
        return self.f(x) + sum([penalty(x) for penalty in self.penalities])
    
    class extrior_penalty:
        def __init__(self, f,ineq_constraints,eq_constraints,grad_f=None, tol=1e-4,tol_ls=1e-4,max_iter=1000, stopping_criteria='point_diff', optimizer_name='quadratic_extended_penalty', line_search_name='None',function_name='f'):
            self.f = f
            self.constraints = ineq_constraints
            self.grad_f = grad_f
            self.tol = tol
            self.tol_ls = tol_ls
            self.max_iter = max_iter
            self.stopping_criteria = stopping_criteria
            self.optimizer_name = optimizer_name
            self.line_search_name = line_search_name
            self.function_name = function_name
            self.f_penalty = self.create_penalty_function(f,ineq_constraints,eq_constraints)
            self.rk=10

        def create_penalty_function(self,x):
            return self.f(x) +sum([self.rk*max(0,constraint(x))**2 for constraint in self.constraints])+sum([self.rk*constraint(x)**2 for constraint in self.eq_constraints])
        def optimize(self,x):
            optimum_point_old = x
            for i in range(self.max_iter):
                optimum_point, optimum_value, func_eval, ls,iter_count, df = simplex_method(self.f_penalty,x)
                if self.stopping_criteria == 'point_diff':
                    if np.linalg.norm(optimum_point_old-optimum_point) < self.tol:
                        break
                self.rk = self.rk*10
            return x