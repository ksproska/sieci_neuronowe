import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(1 - sigmoid(x))


def tanh(x):
    return 2 / (1 + np.exp(-2 * x)) - 1


def tanh_derivative(x):
    return 1 - (tanh(x) ** 2)


def relu(x):
    return np.vectorize(lambda v: max(v, 0.0))(x)



@np.vectorize
def relu_derivative(x):
    if x > 0:
        return 1
    if x == 0:
        return 0.5
    return 0
