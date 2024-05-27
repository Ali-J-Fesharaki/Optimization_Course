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
