import numpy as np
from tabulate import tabulate

class Golden_Quadratic:
    def __init__(self, f, interval=(0,1), accuracy=1e-5, max_iter=50):
        self.f = f
        self.__interval = interval
        self.__accuracy = accuracy
        self.max_iter = max_iter
        self.golden_optimizer = GoldenSection(self.f, self.__interval, self.__accuracy, 1)
        self.quadratic_optimizer = QuadraticCurveFitting(self.f, self.__interval, self.__accuracy, 1)
        self.iter=0
        self.golden_itr=0   
        self.quadratic_itr=0
    def optimize(self):
        while((self.__interval[1]-self.__interval[0])>=self.__accuracy and self.iter<self.max_iter):
            
            x_optimal,golden_itr = self.golden_optimizer.optimize()
            self.golden_itr+=golden_itr
            self.__interval=self.golden_optimizer.get_interval()
            #self.golden_optimizer.set_interval(self.__interval)
            self.quadratic_optimizer.set_interval(self.__interval)
            x_optimal_quad, curve_fit_iter = self.quadratic_optimizer.optimize()
            self.quadratic_itr+=curve_fit_iter
            self.__interval=self.golden_optimizer.get_interval()
            self.iter+=1
        return x_optimal, 2*self.golden_itr+ 3*self.quadratic_itr

class GoldenSection:
    def __init__(self, f, interval=(0,1), accuracy=1e-6, max_iter=10000):
        self.f = f
        self.__interval = interval
        self.__accuracy = accuracy
        self.max_iter = max_iter

    def get_interval(self):
        return self.__interval

    def set_interval(self, interval):
        self.__interval = interval
    
    def get_accuracy(self):
        return self.__accuracy

    def optimize(self):
        a, b = self.__interval
        x1 = a + 0.382 * (b - a)
        x2 = a + 0.618 * (b - a)

        f_x1 = self.f(x1)
        f_x2 = self.f(x2)

        k = 0
        iterations_data = []
        while abs(b - a) > self.__accuracy and k < self.max_iter:
            k += 1
            if f_x1 < f_x2:
                b = x2
                x2 = x1
                x1 = a + 0.382 * (b - a)

                f_x2 = f_x1
                f_x1 = self.f(x1)
            else:
                a = x1
                x1 = x2
                x2 = a + 0.618 * (b - a)

                f_x1 = f_x2
                f_x2 = self.f(x2)

            self.__interval = [a, b]  # Update the interval

            iterations_data.append([k, a, b, x1, x2, f_x1, f_x2])

        headers = ["Iteration", "a", "b", "x1", "x2", "f(x1)", "f(x2)"]
        print(tabulate(iterations_data, headers=headers, tablefmt="grid", floatfmt=".6f"))

        if abs(b - a) <= self.__accuracy:
            print("Optimization converged.")
        else:
            print("Maximum number of iterations reached.")

        return (a + b) / 2,k

class QuadraticCurveFitting:
    def __init__(self, f, interval=(0,1), accuracy=1e-5, max_iter=100):
        self.f = f
        self.__interval = interval
        self.__accuracy = accuracy
        self.max_iter = max_iter
    
    def set_interval(self, interval):
        self.__interval = interval

    def get_interval(self):
        return self.__interval

    def get_accuracy(self):
        return self.__accuracy

    def optimize(self):
        x_lower, x_upper = self.__interval
        x_mid = (x_lower + x_upper) / 2

        F_lower = self.f(x_lower)
        F_upper = self.f(x_upper)
        F_mid = self.f(x_mid)

        x_prev = x_lower
        x_opt = x_upper

        k = 0
        iterations_data = []
        while abs(x_prev - x_opt) > self.__accuracy and k < self.max_iter:
            k += 1
            X = np.array([
                [1, x_lower, x_lower ** 2],
                [1, x_upper, x_upper ** 2],
                [1, x_mid, x_mid ** 2]
            ])

            F = np.array([F_lower, F_upper, F_mid])
            try:
                a = np.linalg.solve(X, F)
            except:
                print("Singular matrix. Cannot solve.", X, F)

            x_prev = x_opt
            x_opt = -a[1] / (2 * a[2])
            F_x_opt = self.f(x_opt)

            if x_opt - x_mid > 0:
                if F_x_opt - F_mid > 0:
                    x_upper = x_opt
                    F_upper = F_x_opt
                else:
                    x_lower = x_mid
                    x_mid = x_opt

                    F_lower = F_mid
                    F_mid = F_x_opt
            else:
                if F_x_opt - F_mid > 0:
                    x_lower = x_opt
                    F_lower = F_x_opt
                else:
                    x_upper = x_mid
                    x_mid = x_opt

                    F_upper = F_mid
                    F_mid = F_x_opt

            self.__interval = (x_lower, x_upper)  # Update the interval

            iterations_data.append([k, x_lower, x_upper, x_mid, F_lower, F_upper, F_mid])

        headers = ["Iteration", "x_lower", "x_upper", "x_mid", "F_lower", "F_upper", "F_mid"]
        print(tabulate(iterations_data, headers=headers, tablefmt="grid",  floatfmt=".6f"))

        if abs(x_prev - x_opt) <= self.__accuracy:
            print("Optimization converged.")
        else:
            print("Maximum number of iterations reached.")

        return x_opt, k

# Define test cases
class FunctionWithEvalCounter:
    def __init__(self, f):
        self.f = f
        self.eval_count = 0

    def __call__(self, x):
        self.eval_count += 1
        return self.f(x)

    def reset(self):
        self.eval_count = 0

    def get_eval_count(self):
        return self.eval_count
def test_quadratic_curve_fitting():
    test_cases = [
        {
            "description": "Basic Quadratic Function with Standard Interval",
            "function": lambda x: (x - 2)**2,
            "interval": (0, 5)
        },
        {
            "description": "Quadratic Function with Roots Outside Interval",
            "function": lambda x: (x - 10)**2,
            "interval": (0, 5)
        },
        {
            "description": "Quadratic Function with Minimum Inside Interval but not at Midpoint",
            "function": lambda x: (x - 1)**2 + 3,
            "interval": (0, 2)
        },
        {
            "description": "Quadratic Function with Singular Matrix Check (edge case)",
            "function": lambda x: 0,
            "interval": (0, 0)
        },
        {
            "description": "Quadratic Function with Narrow Interval",
            "function": lambda x: (x - 1.5)**2 + 1,
            "interval": (1, 2)
        },
        {
            "description": "Quadratic Function with Large Interval",
            "function": lambda x: (x - 3)**2,
            "interval": (-100, 100)
        }
    ]

    for case in test_cases:
        print(f"Testing: {case['description']}")
        f = FunctionWithEvalCounter(case["function"])
        optimizer = QuadraticCurveFitting(f, interval=case["interval"])
        x_opt, iterations = optimizer.optimize()
        print(f"Optimal x: {x_opt}, found in {iterations} iterations\n")
        print("="*50, "\n")

# Initial interval
if __name__ == "__main__":
    test_quadratic_curve_fitting()
