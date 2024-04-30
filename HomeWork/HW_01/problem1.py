import numpy as np
import matplotlib.pyplot as plt

# Objective Function
def f(x1, x2):
    return x1**2 + x2**2 - 3*x1*x2

# Constraint
def g(x1, x2):
    return x1**2 + x2**2 - 6

# Meshgrid
x1_vals = np.arange(-5, 5.1, 0.1)
x2_vals = np.arange(-5, 5.1, 0.1)
x1, x2 = np.meshgrid(x1_vals, x2_vals)
z = f(x1, x2)


cv = np.arange(-15, 20.4, 0.4)

# Plot
cont1 = plt.contour(x1, x2, f(x1, x2), cv)
plt.colorbar()
plt.contour(x1, x2, g(x1, x2), levels=[0], colors='k', linestyles='-')
plt.legend(['Objective', 'Constraint'])
plt.axis('equal')


# Interactive part

while True:
    clicked_point = plt.ginput(n=1, timeout=0, show_clicks=True, mouse_add=1, mouse_pop=3, mouse_stop=2)
    if not clicked_point:
        break
    x, y = clicked_point[0]
    plt.scatter(x, y, color='red')
    print(f"Clicked point coordinates: ({x}, {y})")

plt.show()
