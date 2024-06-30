from direct_methods import  NelderMead
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
    
class Extrior_Penalty:
    def __init__(self, f,ineq_constraints=[(lambda x:0)],eq_constraints=[(lambda x:0)],grad_f=None, tol=1e-2,tol_ls=1e-4,max_iter=1000, stopping_criteria='point_diff', optimizer_name='Extrior_penalty', line_search_name='None',function_name='f'):
        self.f = f
        self.ineq_constraints = ineq_constraints
        self.eq_constraints=eq_constraints
        self.grad_f = grad_f
        self.tol = tol
        self.tol_ls = tol_ls
        self.max_iter = max_iter
        self.stopping_criteria = stopping_criteria
        self.optimizer_name = optimizer_name
        self.line_search_name = line_search_name
        self.function_name = function_name
        self.rk=10
        self.f_penalty = self.create_penalty_function()
        

    def create_penalty_function(self):
        print("*******************************************************")
        print("*******************************************************")
        print(self.rk)
        print("*******************************************************")
        print("*******************************************************")
        return lambda x:(self.f(x) +self.rk*sum([max(0,constraint(x))**2 for constraint in self.ineq_constraints])+sum([self.rk*constraint(x)**2 for constraint in self.eq_constraints]))
    def optimize(self,x):
        optimum_point_old = x
        for i in range(self.max_iter):
            self.f_penalty=self.create_penalty_function()
            optimizer=NelderMead(self.f_penalty)
            optimum_point, optimum_value, func_eval, ls,iter_count, df = optimizer.optimize(x)
            if self.stopping_criteria == 'point_diff':
                if np.linalg.norm(optimum_point_old-optimum_point) < self.tol:
                    break
            self.rk = self.rk*10
            optimum_point_old = optimum_point
        return x

class Augmented_Lagrangian:
    def __init__(self, f,ineq_constraints=[(lambda x:0)],eq_constraints=[(lambda x:0)],grad_f=None, tol=1e-2,tol_ls=1e-4,max_iter=1000, stopping_criteria='point_diff', optimizer_name='Augmented_Lagrange', line_search_name='None',function_name='f'):
        self.f = f
        self.ineq_constraints = ineq_constraints
        self.eq_constraints=eq_constraints
        self.grad_f = grad_f
        self.tol = tol
        self.tol_ls = tol_ls
        self.max_iter = max_iter
        self.stopping_criteria = stopping_criteria
        self.optimizer_name = optimizer_name
        self.line_search_name = line_search_name
        self.function_name = function_name
        self.rk=1
        self.landa=np.zeros(len(self.ineq_constraints)+len(self.eq_constraints))        
        self.m=len(self.ineq_constraints)
        self.f_penalty = self.create_penalty_function()
    
    def create_alpha(self,x):
        return 1

    def create_penalty_function(self):
        return lambda x:(self.f(x) +sum([self.landa[i]*max(-(self.landa[i]/2*self.rk),constraint(x)) for i,constraint in enumerate(self.ineq_constraints)])+
                            sum([self.landa[i+self.m]*constraint(x) for i,constraint in enumerate(self.eq_constraints)])+
                            self.rk*sum([max(-(self.landa[i]/2*self.rk),constraint(x))**2 for i,constraint in enumerate(self.ineq_constraints)])+
                            self.rk*sum([constraint(x)**2 for constraint in self.eq_constraints]))
    def optimize(self,x):
        optimum_point_old = x
        for i in range(self.max_iter):
            self.f_penalty=self.create_penalty_function()
            optimizer=NelderMead(self.f_penalty)
            optimum_point, optimum_value, func_eval, ls,iter_count, df = optimizer.optimize(x)

            for i,constraint in enumerate(self.ineq_constraints):
                self.landa[i]+=2*self.rk*max(-(self.landa[i]/2*self.rk),constraint(optimum_point))

            for i,constraint in enumerate(self.eq_constraints):
                self.landa[i+self.m]+=2*self.rk*constraint(optimum_point)

            if self.stopping_criteria == 'point_diff':
                if np.linalg.norm(optimum_point_old-optimum_point) < self.tol:
                    break
            optimum_point_old = optimum_point
        return x

        
def objective_function(x):
    x1, x2, x3, x4, x5 = x
    obj_value = np.exp(x1 * x2 * x3 * x4 * x5) - 0.5 * (x1**3 + x2**3 + 1)**2
    return obj_value

def constraint1(x):
    x1, x2, x3, x4, x5 = x
    return x1**2 + x2**2 + x3**2 + x4**2 + x5**2 - 10

def constraint2(x):
    x1, x2, x3, x4, x5 = x
    return x2 * x3 - 5 * x4 * x5

def constraint3(x):
    x1, x2, x3, x4, x5 = x
    return x1**3 + x2**3 + 1     
 
# def objective_function(x):
#     x1, x2 = x
#     return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2

# # Define the constraints
# def constraint1(x):
#     x1, x2 = x
#     return -((x1 + 2)**2 - x2)

# def constraint2(x):
#     x1, x2 = x
#     return -(-4 * x1 + 10 * x2)

if(__name__=='__main__'):
    optimizer=Augmented_Lagrangian(objective_function,eq_constraints=[constraint1,constraint2,constraint3])
    # optimizer = Augmented_Lagrangian(objective_function,ineq_constraints=[constraint1,constraint2])
    #x = np.array([3, 4])# Wow when i wanted to type the numbers of x the copilot autocomplete the numbers Wow.....
    x=np.array([-1.8,1.7,1.9,-0.8,-0.8])
    print(optimizer.optimize(x))