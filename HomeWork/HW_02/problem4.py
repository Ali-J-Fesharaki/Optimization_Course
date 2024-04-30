import math

# Problem 4
# Take user input for the accuracy and initial interval
accuracy = float(input('Please enter accuracy of uncertainty: '))
a = float(input('Please enter the minimum of initial interval: '))
b = float(input('Please enter the maximum of initial interval: '))

# Define the function f(x)
def f(x):
    return x**3 - 9*x

# Define the required length for termination condition
L = 2 * accuracy

# Define the golden ratio
alpha = (math.sqrt(5) - 1) / 2

# Initialize the two initial points
landa = a + (1 - alpha) * (b - a)
miu = a + alpha * (b - a)

# Evaluate the function at the initial points
flanda = f(landa)
fmiu = f(miu)

# Initialize the length of the interval
Length = b - a

# Initialize the iteration counter
k = 0

# Perform the golden section search algorithm
while Length > L:
    if flanda > fmiu:
        # Update the interval and function value based on the comparison
        a = landa
        b = b
        landa = miu
        miu = a + alpha * (b - a)
        Length = b - a
        flanda = fmiu
        fmiu = f(miu)
    elif flanda <= fmiu:
        # Update the interval and function value based on the comparison
        a = a
        b = miu
        miu = landa
        landa = a + (1 - alpha) * (b - a)
        Length = b - a
        fmiu = flanda
        flanda = f(landa)
    # Increment the iteration counter
    k += 1

# Print the results
print('The iteration number of k is', k)
print('The optimal solution lies in the interval ({}, {})'.format(a, b))
