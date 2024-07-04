import numpy as np

def f_1(x):
    return (4 - x[0])**2 + (4 - x[1])**2 + 45 * (x[1] - x[0]**2)**2 + 45 * (x[2] - x[1]**2)**2

def grad_f1(x):
    grad = np.zeros(3)
    grad[0] = 2.0 * x[0] - 180.0 * x[0] * (-x[0]**2 + x[1]) - 8.0
    grad[1] = -90.0 * x[0]**2 + 92.0 * x[1] - 180.0 * x[1] * (x[2] - x[1]**2) - 8.0
    grad[2] = -90.0 * x[1]**2 + 90.0 * x[2]
    return grad
def f_2(x):
    return ((x[0] * x[1] - x[0] + 1.5)**2 + 
            (x[0] * x[1]**2 - x[0] + 2.25)**2 + 
            (x[0] * x[1]**3 - x[0] + 2.625)**2)

def grad_f2(x):
    grad = np.zeros(2)
    grad[0] = (2.0 * (x[1]**2 - 1.0) * (x[0] * x[1]**2 - x[0] + 2.25) + 
               2.0 * (x[1]**3 - 1.0) * (x[0] * x[1]**3 - x[0] + 2.625) + 
               2.0 * (x[1] - 1.0) * (x[0] * x[1] - x[0] + 1.5))
    grad[1] = (2.0 * x[0] * (x[0] * x[1] - x[0] + 1.5) + 
               6.0 * x[0] * x[1]**2 * (x[0] * x[1]**3 - x[0] + 2.625) + 
               4.0 * x[0] * x[1] * (x[0] * x[1]**2 - x[0] + 2.25))
    return grad
def constrained_f1(x):
    x1, x2, x3, x4, x5 = x
    obj_value = np.exp(x1 * x2 * x3 * x4 * x5) - 0.5 * (x1**3 + x2**3 + 1)**2
    return obj_value

def constraint1_f1(x):
    x1, x2, x3, x4, x5 = x
    return x1**2 + x2**2 + x3**2 + x4**2 + x5**2 - 10

def constraint2_f1(x):
    x1, x2, x3, x4, x5 = x
    return x2 * x3 - 5 * x4 * x5

def constraint3_f1(x):
    x1, x2, x3, x4, x5 = x
    return x1**3 + x2**3 + 1     
 
def constrained_f2(x):
    x1, x2 = x
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2

# Define the constraints
def constraint1_f2(x):
    x1, x2 = x
    return -((x1 + 2)**2 - x2)

def constraint2_f2(x):
    x1, x2 = x
    return -(-4 * x1 + 10 * x2)
if (__name__ == "__main__"):
    x1 = np.array([ 0,0])
    x2 = np.array([ 0,0.5])
    print(f_2(x1))
    print(f_2(x2))