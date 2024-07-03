# from direct_methods import NelderMead
# from gradient_methods import FletcherReeves
# from functions import f_1 ,f_2,grad_f1,grad_f2

# if(__name__=="__main__"):
#     optimum_point, optimum_value, func_eval,ls,iter_count, df=FletcherReeves(f_1,grad_f=grad_f1,tol_ls=1e-4,tol=1e-4).optimize([0,0,0])
#     print("Optimum point:",optimum_point)
#     print("Optimum value:",optimum_value)
#     print("Function evaluation count:",func_eval)
#     print("Line search evaluation count:",ls)
#     print("Number of iterations:",iter_count)
import sympy as sp
import inspect
import re
# Define the symbolic variables
x1, x2, x3, x4, x5 = sp.symbols('x1 x2 x3 x4 x5')
x = [x1, x2, x3, x4, x5]

# Define the objective function using sympy
def objective_function(x):
    x1, x2, x3, x4, x5 = x
    obj_value = sp.exp(x1 * x2 * x3 * x4 * x5) - 0.5 * (x1**3 + x2**3 + 1)**2
    return obj_value

# Define the constraints using sympy
def constraint1(x):
    x1, x2, x3, x4, x5 = x
    return x1**2 + x2**2 + x3**2 + x4**2 + x5**2 - 10

def constraint2(x):
    x1, x2, x3, x4, x5 = x
    return x2 * x3 - 5 * x4 * x5

def constraint3(x):
    x1, x2, x3, x4, x5 = x
    return x1**3 + x2**3 + 1

# Create symbolic expressions
objective_expr = objective_function(x)
constraint1_expr = constraint1(x)
constraint2_expr = constraint2(x)
constraint3_expr = constraint3(x)

def python_to_matlab(func, func_name, var_names):
    func_str = inspect.getsource(func)
    func_str = re.sub(r'^def .*?\(', f'def {func_name}(', func_str, flags=re.DOTALL)
    func_str = func_str.replace('return ', '').replace('\n', '')
    for i, var in enumerate(var_names):
        func_str = func_str.replace(f'x{i+1}', var)
    return func_str

# Variable names for MATLAB
var_names = ['x(1)', 'x(2)', 'x(3)', 'x(4)', 'x(5)']

# Convert Python functions to MATLAB strings
objective_str = python_to_matlab(objective_function, 'objective_function', var_names)
constraint1_str = python_to_matlab(constraint1, 'constraint1', var_names)
constraint2_str = python_to_matlab(constraint2, 'constraint2', var_names)
constraint3_str = python_to_matlab(constraint3, 'constraint3', var_names)

# Combine constraints into one string
constraints_str = f"""
function [c, ceq] = confun(x)
    c = zeros(2, 1);
    ceq = zeros(1, 1);
    c(1) = {constraint1_str};
    c(2) = {constraint2_str};
    ceq(1) = {constraint3_str};
end
"""

# Objective function string
objective_func_str = f"""
function f = objfun(x)
    {objective_str}
end
"""

# Combine all MATLAB code
matlab_code_str = objective_func_str + "\n" + constraints_str

# Print MATLAB code
print(matlab_code_str)