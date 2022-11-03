"""
X_train: (60000, 28, 28)
Y_train: (60000,)
X_test:  (10000, 28, 28)
Y_test:  (10000,)
"""
# https://www.askpython.com/python/examples/load-and-plot-mnist-dataset-in-python

# from keras.datasets import mnist
# (train_X, train_y), (test_X, test_y) = mnist.load_data()

import numpy as np
import random


def reproduce_x_times(input, num_to_merge):
    output = input.T
    for i in range(num_to_merge - 1):
        output = np.concatenate((output, input.T))
    return output.T


def get_random(shape):
    rand_matrix = np.random.rand(shape[0], shape[1]) / 10 - 0.05
    return rand_matrix


x_unipolar = np.array(
    [
        [0, 0, 1, 1],
        [0, 1, 0, 1]
    ]
)
d_bipolar = np.array([[0, 0, 0, 1]])

repetitions = int((60000 + 10000) / 4)


def get_random_weights(length, min_val, max_val):
    return np.array([[random.uniform(min_val, max_val) for _ in range(length)]]).T


x_all = reproduce_x_times(x_unipolar, repetitions)
d_all = reproduce_x_times(d_bipolar, repetitions)
x_all = x_all + get_random(x_all.shape)

x_train, x_test = x_all[:, :60000].T, x_all[:, 60000:].T
d_train, d_test = d_all[:, :60000].T, d_all[:, 60000:].T
# print(x_train.shape, d_train.shape)
