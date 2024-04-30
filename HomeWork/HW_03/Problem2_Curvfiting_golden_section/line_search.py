import numpy as np
from tabulate import tabulate
class Golden_Quadratic:
    def __init__(self, f, interval, accuracy, max_iter=100):
        self.f = f
        self.interval = interval
        self.accuracy = accuracy
        self.max_iter = max_iter

    def optimize(self):
        golden_optimizer = GoldenSection(self.f, self.interval, self.accuracy, 1)
        x_optimal, golden_iter = golden_optimizer.optimize()

        interval_new = [self.interval[0], x_optimal]
        quadratic_optimizer = QuadraticCurveFitting(self.f, interval_new, self.accuracy, self.max_iter)
        x_optimal_quad, curve_fit_iter = quadratic_optimizer.optimize()

        return x_optimal_quad, golden_iter+ curve_fit_iter

class GoldenSection:
    def __init__(self, f, interval, accuracy, max_iter=100):
        self.f = f
        self.interval = interval
        self.accuracy = accuracy
        self.max_iter = max_iter


    def optimize(self):
        a, b = self.interval
        x1 = a + 0.382 * (b - a)
        x2 = a + 0.618 * (b - a)

        f_x1 = self.f(x1)
        f_x2 = self.f(x2)

        k = 0
        iterations_data = []
        while abs(b - a) > self.accuracy and k < self.max_iter:
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

            iterations_data.append([k, a, b, x1, x2, f_x1, f_x2])

        headers = ["Iteration", "a", "b", "x1", "x2", "f(x1)", "f(x2)"]
        print(tabulate(iterations_data, headers=headers, tablefmt="grid", floatfmt=".6f"))

        if abs(b - a) <= self.accuracy:
            print("Optimization converged.")
        else:
            print("Maximum number of iterations reached.")

        return (a + b) / 2, k
class QuadraticCurveFitting:
    def __init__(self, f, interval, accuracy, max_iter=100):
        self.f = f
        self.interval = interval
        self.accuracy = accuracy
        self.max_iter = max_iter

    def optimize(self):
        x_lower, x_upper = self.interval
        x_mid = (x_lower + x_upper) / 2

        F_lower = self.f(x_lower)
        F_upper = self.f(x_upper)
        F_mid = self.f(x_mid)

        x_prev = x_lower
        x_opt = x_upper

        k = 0
        iterations_data = []
        while abs(x_prev - x_opt) > self.accuracy and k < self.max_iter:
            k += 1
            X = np.array([
                [1, x_lower, x_lower ** 2],
                [1, x_upper, x_upper ** 2],
                [1, x_mid, x_mid ** 2]
            ])

            F = np.array([F_lower, F_upper, F_mid])
            a = np.linalg.solve(X, F)

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

            iterations_data.append([k, x_lower, x_upper, x_mid, F_lower, F_upper, F_mid])

        headers = ["Iteration", "x_lower", "x_upper", "x_mid", "F_lower", "F_upper", "F_mid"]
        print(tabulate(iterations_data, headers=headers, tablefmt="grid",  floatfmt=".6f"))

        if abs(x_prev - x_opt) <= self.accuracy:
            print("Optimization converged.")
        else:
            print("Maximum number of iterations reached.")

        return x_opt, k

# Example usage:
def f(x):
    return x ** 2 - 4 * x + 4
if(__name__ == '__main__'):
    interval = [0, 3]
    accuracy = 1e-5
    n_iter = 100  # Set the maximum number of iterations

    print("Running Quadratic Curve Fitting Optimization:")
    print("-" * 80)
    optimizer = QuadraticCurveFitting(f, interval, accuracy, max_iter=n_iter)
    opt_point, fej_qc = optimizer.optimize()
    print("-" * 80)
    print("Optimal point:", opt_point)
    print("Number of function evaluations:", fej_qc)
