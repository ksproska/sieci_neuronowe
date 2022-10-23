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


@np.vectorize
def tanh(x):
    return 2 / (1 + math.exp(-2 * x)) - 1


@np.vectorize
def relu(x):
    if x >= 0:
        return x
    return 0


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


"""
X_train: (60000, 28, 28)
Y_train: (60000,)
X_test:  (10000, 28, 28)
Y_test:  (10000,)
"""


class MLP:
    def __init__(self, neuron_counts=[10, 20], activation_fun=sigmoid):
        self.activation_fun = activation_fun

        (train_X, train_y), (test_X, test_y) = mnist.load_data()
        train_X = train_X.reshape((train_X.shape[0], -1))
        train_y = train_y.reshape((train_y.shape[0], -1))
        test_X = test_X.reshape((test_X.shape[0], -1))
        test_y = test_y.reshape((test_y.shape[0], -1))
        (self.train_X, self.train_y), (self.test_X, self.test_y) = (train_X, train_y), (test_X, test_y)

        self.neuron_counts = [train_X.shape[1]] + neuron_counts + [10]
        self.all_weights = [get_random(self.get_weights_matrix_shape(i)) for i in range(len(self.neuron_counts) - 1)]
        self.all_bs = [get_random(self.get_b_matrix_shape(i)) for i in range(len(self.neuron_counts) - 1)]

    def get_weights_matrix_shape(self, first_inx):
        return [self.neuron_counts[first_inx + 1], self.neuron_counts[first_inx]]

    def get_a_matrix_shape(self, inx):
        return [self.neuron_counts[inx], 1]

    def get_b_matrix_shape(self, first_inx):
        return [self.neuron_counts[first_inx + 1], 1]

    def count_one_step(self):
        f = self.activation_fun
        for i in range(self.train_X.shape[0]):
            x = self.train_X[i, :]
            a = x
            for j in range(len(self.neuron_counts) - 1 - 1):
                W = self.all_weights[j]
                b = self.all_bs[j]
                z = W @ a + b
                a = f(z)
            y = softmax(a)


def get_random(shape):
    rand_matrix = np.random.rand(*shape)
    return rand_matrix


def main():
    mlp = MLP()
    mlp.count_one_step()
    # for i in range(len(mlp.all_weights)):
    #     print("W", mlp.all_weights[i].shape)
    #     print("a", mlp.get_a_matrix_shape(i))
    #     print("b", mlp.all_bs[i].shape)
    #     print()


if __name__ == '__main__':
    main()
