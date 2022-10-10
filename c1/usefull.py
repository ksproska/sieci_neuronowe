import unittest
import numpy as np
import random


def delta_w(d, y, x):
    diff = d - y
    delt = diff @ x.T
    return delt


def delta_w_adaline(d, y, x):
    diff = d - y
    delt = diff @ x.T
    return -2 * delt


def get_random_weights(length, min_val=0.01, max_val=0.2):
    # return np.random.randn(1, length) * min_val
    # ToDo gotowa funkcja w numpy -> np.random.rand(1,2) / 10
    return np.array([random.uniform(min_val, max_val) for _ in range(length)])


def sign_bipolar(x):
    if x < 0:
        return -1
    return 1


def merge_multiple_times(input, num_to_merge):
    output = input.T
    for i in range(num_to_merge - 1):
        output = np.concatenate((output, input.T))
    return output.T


def randomize(input):
    rand_matrix = np.random.rand(input.shape[0], input.shape[1]) / 10 - 0.05
    rand_matrix[0, :] = 0
    return input + rand_matrix


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
        output = merge_multiple_times(np.array([[0, 1, 0], [0, 0, 1]]), 2)
        np.testing.assert_equal(output, np.array([[0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1]]))

        output = merge_multiple_times(np.array([[0, 1, 0], [0, 0, 1]]), 3)
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
