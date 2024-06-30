import numpy as np
import concurrent.futures
from gradient_methods import FletcherReeves
from direct_methods import Powell, NelderMead
from hessian_methods import BFGS  
from functions import f_1, grad_f1 ,f_2, grad_f2

def run_optimization(optimizer_class, initial_point, function, grad_function=None):
    optimizer = optimizer_class(f=function, grad_f=grad_function)
    result = optimizer.optimize(initial_point)
    return result

def main():
    num_points = 10  # Number of random initial points
    dimension = 2  # Dimension of the initial points
    initial_points = [np.random.uniform(-4.5, 4.5, dimension) for _ in range(num_points)]
    
    optimizers = [
        (FletcherReeves, f_2, grad_f2),
        (Powell, f_2, None),
        (NelderMead, f_2, None),
        (BFGS, f_2, grad_f2)
    ]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for initial_point in initial_points:
            for optimizer_class, function, grad_function in optimizers:
                futures.append(executor.submit(run_optimization, optimizer_class, initial_point, function, grad_function))
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print(result)
if __name__ == '__main__':
    main()
