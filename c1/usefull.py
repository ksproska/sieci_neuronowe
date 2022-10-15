import unittest
from math import sqrt

import numpy as np
import random
import matplotlib.pyplot as plt

def get_dimensions(n):
    tempSqrt = sqrt(n)
    divisors = []
    currentDiv = 1
    for currentDiv in range(n):
        if n % float(currentDiv + 1) == 0:
            divisors.append(currentDiv + 1)
    # print divisors this is to ensure that we're choosing well
    hIndex = min(range(len(divisors)), key=lambda i: abs(divisors[i] - sqrt(n)))

    if divisors[hIndex] * divisors[hIndex] == n:
        return divisors[hIndex], divisors[hIndex]
    else:
        wIndex = hIndex + 1
        return divisors[hIndex], divisors[wIndex]


# from perceptron_experiments import get_dimensions

class AllPlots:
    def __init__(self, all_count, title):
        self.dimensions = get_dimensions(all_count)
        d = self.dimensions[::-1]
        plt.rcParams["figure.figsize"] = (d[0]*4, d[1]*4)
        f, self.axis = plt.subplots(*self.dimensions)
        f.suptitle(title)
        f.tight_layout(pad=5)
        self.order_numb = 0
        self.current = CurrentPlot(self.axis[self.order_numb // self.dimensions[1], self.order_numb % self.dimensions[1]])

    def next(self):
        self.order_numb = (self.order_numb + 1) % (self.dimensions[0] * self.dimensions[1])
        self.current = CurrentPlot(self.axis[self.order_numb // self.dimensions[1], self.order_numb % self.dimensions[1]])


class CurrentPlot:
    def __init__(self, plot_current):
        self.plot_current = plot_current
        self.plot_current.axhline(y=0, color="k")
        self.plot_current.axvline(x=0, color="k")
        self.plot_current.grid(True)
        # plt.sca(self.plot_current)
        plt.ylim(-1, 1)
        plt.xlim(-1, 1)

    def scatter(self, xs, ys, groups):
        self.plot_current.scatter(xs, ys, c=groups)

    def plot_line(self, start, end, y_func):
        x_vals = np.linspace(start, end, 10)
        self.plot_current.plot(x_vals, y_func(x_vals))

    def set_title(self, title):
        self.plot_current.set_title(title)


def delta_w(d, y, x):
    diff = d - y
    delt = diff @ x.T
    return delt


def delta_w_adaline(d, y, x):
    diff = d - y
    delt = diff @ x.T
    return -2 * delt


def get_random_weights(length, min_val, max_val):
    return np.array([[random.uniform(min_val, max_val) for _ in range(length)]]).T


def sign_bipolar(x):
    if x < 0:
         return -1
    return 1


def reproduce_x_times(input, num_to_merge):
    output = input.T
    for i in range(num_to_merge - 1):
        output = np.concatenate((output, input.T))
    return output.T


def randomize(input):
    rand_matrix = np.random.rand(input.shape[0], input.shape[1]) / 10 - 0.05
    rand_matrix[0, :] = 0
    return input + rand_matrix


def get_random_except_first_row(shape):
    rand_matrix = np.random.rand(shape[0], shape[1]) / 10 - 0.05
    rand_matrix[0, :] = 0
    return rand_matrix


def count_cost(weights, xs):
    multip = xs @ weights
    return multip


def apply_func(current_cost, func):
    estimated = np.array([func(x) for x in current_cost])
    return estimated


def unipolar(teta, z):
    return int(z > teta)


def bipolar(teta, z):
    return 2 * (z > teta) - 1


def calculate_new_weight(old, alfa, delta_weight):
    return old + alfa * delta_weight


class TestStringMethods(unittest.TestCase):
    def test_merge_multiple_times(self):
        output = reproduce_x_times(np.array([[0, 1, 0], [0, 0, 1]]), 2)
        np.testing.assert_equal(output, np.array([[0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1]]))

        output = reproduce_x_times(np.array([[0, 1, 0], [0, 0, 1]]), 3)
        np.testing.assert_equal(output, np.array([[0, 1, 0, 0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1, 0, 0, 1]]))

    def test_count_new_weight(self):
        np.testing.assert_equal(calculate_new_weight(np.array([0.5, 0]), 0.5, np.array([-1, 0])), np.array([0, 0]))

    def test_cost(self):
        np.testing.assert_equal(count_cost(np.array([[0, 1, 0], [0, 0, 1]]), np.array([0.5, 0])), np.array([0, 0.5, 0]))
        np.testing.assert_equal(count_cost(np.array([[0, 1, 0], [0, 0, 1]]), np.array([1, 1])), np.array([0, 1, 1]))

    def test_estimate(self):
        np.testing.assert_equal(apply_func(np.array([0, 0.5, 0]), lambda x: unipolar(0.2, x)), np.array([0, 1, 0]))
        np.testing.assert_equal(apply_func(np.array([0, 0.5, 0]), lambda x: bipolar(0.2, x)), np.array([-1, 1, -1]))

    def test_delta_w(self):
        np.testing.assert_equal(delta_w(np.array([0, 0, 0]), np.array([0, 1, 0]), np.array([[0, 1, 0], [0, 0, 1]])),
                                np.array([-1, 0]))

    def test_get_random_weights(self):
        w = get_random_weights(3)
        self.assertEqual(w.shape, (1,3))
        # for x in w:
        #     self.assertLessEqual(0, x)
        #     self.assertLessEqual(x, 0.2)

    def test_unipolar(self):
        self.assertEqual(unipolar(0.1, 0.2), 1)
        self.assertEqual(unipolar(0, -0.2), 0)

    def test_bipolar(self):
        self.assertEqual(bipolar(0.1, 0.2), 1)
        self.assertEqual(bipolar(0, -0.2), -1)


if __name__ == '__main__':
    unittest.main()
