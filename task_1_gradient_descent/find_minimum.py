import matplotlib.pyplot as plt
from random import randint

ITER_MAX = 1e4
MAX_VAL = 1e7
# for f(x) - quadratic function
# X_AXIS_LEN = 10
# for g(x) - polynomial function
X_AXIS_LEN = 3


def getGradientValue(x: float):
    """
    Quadratic function gradient:
    f'(x) = 2*x + 3
    Polynomial function gradient:
    g'(x) = 4*x**3 - 10*x - 3
    """
    # for f(x) - quadratic function
    return 2 * x + 3
    # for g(x) - polynomial function
    # return 4*x**3 - 10*x - 3


def getFunctionValue(x: float):
    """
    Quadratic function:
    f(x) = x**2 + 3*x + 7
    Polynomial function:
    g(x) = x**4 - 5*x**2 - 3*x
    """
    # for f(x) - quadratic function
    return 7 + 3 * x + 2 * x ** 2
    # for g(x) - polynomial function
    # return x**4 - 5*x**2 - 3*x


def plotGraph(delta: float = 0.05):
    x = -X_AXIS_LEN
    x_values = []
    y_values = []

    while x <= X_AXIS_LEN:
        x_values.append(x)
        y_values.append(getFunctionValue(x))
        x += delta

    plt.plot(x_values, y_values)
    plt.ylabel('Y AXIS')
    plt.show()


def findMinimum(
        starting_point: float,
        beta: float = 0.15,
        error_margin: float = 0.05
):
    """
    Function finds local minimum of a function defined in a
    getFunctionValue().
    Polynomial function optimization works properly for small starting point
    values. For greater values - out of range case.
    difference between beta values can be best observed for values:
    0.9, 0.6, 0.1 - for f(x)
    0.2, 0.15, 0.01 - for g(x)
    """

    x = starting_point
    iteration_num = 0
    found_x_values = []
    found_y_values = []
    while (abs(getGradientValue(x)) > error_margin
            and iteration_num < ITER_MAX):
        delta = getGradientValue(x)

        if abs(x) > MAX_VAL:
            return None

        found_x_values.append(x)
        found_y_values.append(getFunctionValue(x))
        x -= delta * beta
        iteration_num += 1

    plt.plot(found_x_values, found_y_values)
    return x


starting_x = randint(-X_AXIS_LEN, X_AXIS_LEN)
local_minimum = findMinimum(starting_x, 0.1)

print(f"Starting x: {starting_x}")
if local_minimum is None:
    print("Local minimum has not been found due to the large numbers")
else:
    print(f"Local minimum found in: {local_minimum}")

plotGraph()
