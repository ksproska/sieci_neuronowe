"""
Aplikacja powinna być napisana na tyle ogólnie, aby była możliwość:
b) użycia od 2-4 warstw,
c) użycia różnych funkcji aktywacji w warstwach ukrytych (sigmoidalna, tanh, ReLU),
d) użycia warstwy softmax (na Rys. 7. smax) w warstwie wyjściowej,
e) zmiany sposobu inicjalizowania wag (w tym ćwiczeniu przyjmujemy, że wagi będą
inicjalizowane z rozkładu normalnego ze zmiennym odchyleniem standardowym),
f) Zmiany liczby neuronów w warstwach ukrytych,
g) przerwania uczenia i ponownego rozpoczęcia nauki od poprzednich wartości wag.
"""

import math
from enum import Enum
import numpy as np
from keras.datasets import mnist


@np.vectorize
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(1 - sigmoid(x))


@np.vectorize
def tanh(x):
    return 2 / (1 + math.exp(-2 * x)) - 1


def tanh_derivative(x):
    return 1 - (tanh(x) ^ 2)


@np.vectorize
def relu(x):
    if x >= 0:
        return x
    return 0


def relu_derivative(x):
    if x > 0:
        return 1
    if x == 0:
        return 0.5
    return 0


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def negative_log_likelyhood(y, d):
    return -np.log(y) * d


def get_d_matrix(y):
    d = np.zeros((10, y.shape[0]))
    for i in range(d.shape[1]):
        d[int(y[i][0])][i] = 1
    return d


"""
X_train: (60000, 28, 28)
Y_train: (60000,)
X_test:  (10000, 28, 28)
Y_test:  (10000,)
"""


class MLP:
    def __init__(self, neuron_counts=[10, 20], activation_fun=sigmoid, fun_derivative=sigmoid_derivative):
        self.activation_fun = activation_fun
        self.activation_fun_derivative = fun_derivative

        (train_X, train_y), (test_X, test_y) = mnist.load_data()
        train_X = train_X.reshape((train_X.shape[0], -1))
        train_y = train_y.reshape((train_y.shape[0], -1))
        test_X = test_X.reshape((test_X.shape[0], -1))
        test_y = test_y.reshape((test_y.shape[0], -1))
        (self.train_X, self.train_y), (self.test_X, self.test_y) = (train_X, train_y), (test_X, test_y)

        self.neuron_counts = [train_X.shape[1]] + neuron_counts + [10]
        self.all_weights = [get_random(self.get_weights_matrix_shape(i)) for i in range(len(self.neuron_counts) - 1)]
        self.all_bs = [get_random(self.get_b_matrix_shape(i)) for i in range(len(self.neuron_counts) - 1)]

    def __str__(self):
        neurons = "            ".join(str(x) for x in self.neuron_counts)
        weights = "      ".join(str(x.shape) for x in self.all_weights)
        bs = "    ".join(str(x.shape) for x in self.all_bs)
        return f'Neuron counts: {neurons}\n' \
               f'Weights:           {weights}\n' \
               f'Bs:               {bs}'

    def get_weights_matrix_shape(self, first_inx):
        return [self.neuron_counts[first_inx + 1], self.neuron_counts[first_inx]]

    def get_a_matrix_shape(self, inx):
        return [self.neuron_counts[inx], 60000]

    def get_b_matrix_shape(self, first_inx):
        return [self.neuron_counts[first_inx + 1], 60000]

    def count_one_step(self):
        f = self.activation_fun
        f_prim = self.activation_fun_derivative
        x = self.train_X.T
        # x = np.zeros((60000, 28 * 28)).T
        # self.train_y = np.zeros((60000, 1))
        a_all = [x]
        z_all = []

        for j in range(len(self.neuron_counts) - 1):
            W = self.all_weights[j]
            b = self.all_bs[j]
            z = W @ a_all[j] + b
            z_all.append(z)
            a = f(z)
            a_all.append(a)
        y = softmax(a_all[-1])
        d = get_d_matrix(self.train_y)
        err = y - d
        err_all = [err]
        # print(self.neuron_counts)
        # print(err.shape)

        for j in range(len(self.neuron_counts) - 1 - 1, 1, -1):
            # print(j)
            W = self.all_weights[j]
            err = err_all[0]
            z = z_all[j - 1]
            # print(W.T.shape, err.shape, f_prim(z).shape)
            new_err = (W.T @ err) * f_prim(z)
            err_all.insert(0, new_err)

        for j in range(len(self.neuron_counts) - 1):
            W = self.all_weights[j]
            a = a_all[j]
            b = self.all_bs[j]
            # print(W.shape, a.shape, b.shape, a.shape)
            delta_w = -f_prim(W @ a + b) @ a.T
            delta_b = -f_prim(W @ a + b)
            self.all_weights[j] += delta_w
            self.all_bs[j] += delta_b
        # print(self.train_y[i][0], list(y.T[0]))


def get_random(shape):
    rand_matrix = np.random.rand(*shape)
    return rand_matrix


def main():
    mlp = MLP()
    print(mlp)
    for i in range(len(mlp.all_weights)):
        print("W", mlp.all_weights[i].shape)
        print("a", mlp.get_a_matrix_shape(i))
        print("b", mlp.all_bs[i].shape)
        print()

    mlp.count_one_step()
    # print(np.array([[0], [7]]).shape)
    # print(get_d_matrix(np.array([[0], [7]])))


if __name__ == '__main__':
    main()
