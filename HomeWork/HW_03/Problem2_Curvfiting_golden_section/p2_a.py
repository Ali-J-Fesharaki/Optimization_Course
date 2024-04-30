from line_search import QuadraticCurveFitting
def f(x):
    return x ** 3 - 9*x

interval = [1, 2]
accuracy = 1e-5


optimizer = QuadraticCurveFitting(f, interval, accuracy)
opt_point, fej_qc = optimizer.optimize()
print("Optimal point:", opt_point)
print("Number of function evaluations:", fej_qc)