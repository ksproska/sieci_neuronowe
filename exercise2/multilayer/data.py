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

import numpy as np
# from keras.datasets import mnist
from temp import x_train, d_train, x_test, d_test

from activation_functions import sigmoid, sigmoid_derivative


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def negative_log_likelyhood(y, d):
    return -np.log(y) * d


def get_d_matrix(y, dim):
    d = np.zeros((dim, y.shape[0]))
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
    def __init__(self, neuron_counts=[10, 5], activation_fun=sigmoid, fun_derivative=sigmoid_derivative):
        self.activation_fun = activation_fun
        self.activation_fun_derivative = fun_derivative

        # (train_X, train_y), (test_X, test_y) = mnist.load_data()
        # train_X = train_X.reshape((train_X.shape[0], -1))
        # train_y = train_y.reshape((train_y.shape[0], -1))
        # test_X = test_X.reshape((test_X.shape[0], -1))
        # test_y = test_y.reshape((test_y.shape[0], -1))
        # (self.train_X, self.train_y), (self.test_X, self.test_y) = (train_X, train_y), (test_X, test_y)
        (self.train_X, self.train_y), (self.test_X, self.test_y) = (x_train, d_train), (x_test, d_test)

        self.neuron_counts = [self.train_X.shape[1]] + neuron_counts + [2]
        self.all_weights = [get_random(self.get_weights_matrix_shape(i)) for i in range(len(self.neuron_counts) - 1)]
        self.all_bs = [get_random(self.get_b_matrix_shape(i)) for i in range(len(self.neuron_counts) - 1)]

    def __str__(self):
        neurons = "            ".join(str(x) for x in self.neuron_counts)
        weights = "      ".join(str(x.shape) for x in self.all_weights)
        xs = "   ".join(str(self.get_a_matrix_shape(i)) for i in range(len(self.neuron_counts) - 1))
        return f'Neuron counts: {neurons}\n' \
               f'Weights:           {weights}\n' \
               f'Xs:               {xs}'

    def get_weights_matrix_shape(self, first_inx):
        return [self.neuron_counts[first_inx + 1], self.neuron_counts[first_inx]]

    def get_a_matrix_shape(self, inx):
        return [self.neuron_counts[inx], self.train_X.shape[0]]

    def get_b_matrix_shape(self, first_inx):
        return [1, self.train_X.shape[0]]

    def count_one_step(self):
        f = self.activation_fun
        f_prim = self.activation_fun_derivative
        x = self.train_X.T
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
        d = get_d_matrix(self.train_y, y.shape[0])
        # print(d)
        err = d - y  # y - d ?
        err_all = [err]

        for j in range(len(self.neuron_counts) - 1 - 1, 1, -1):
            W = self.all_weights[j]
            err = err_all[0]
            z = z_all[j - 1]
            new_err = (W.T @ err) * f_prim(z)
            err_all.insert(0, new_err)

        for j in range(len(self.neuron_counts) - 1):
            W = self.all_weights[j]
            a = a_all[j]
            b = self.all_bs[j]
            z = z_all[j]
            # print("z", z.shape, "a", a.shape)
            delta_w = z @ a.T / z.shape[1]
            delta_b = np.sum(z) / z.shape[1]
            self.all_weights[j] -= delta_w * 0.01
            self.all_bs[j] -= delta_b * 0.01

        return y


#     def get_d_for(self, x):
#         f = self.activation_fun
#         x = x.T
#         a_all = [x]
#         z_all = []

#         for j in range(len(self.neuron_counts) - 1):
#             W = self.all_weights[j]
#             b = self.all_bs[j]
#             z = W @ a_all[j] + b
#             z_all.append(z)
#             a = f(z)
#             a_all.append(a)
#         return softmax(a_all[-1])


def get_random(shape):
    rand_matrix = np.random.rand(*shape)
    return rand_matrix


def main():
    np.set_printoptions(precision=3, suppress=True)
    mlp = MLP()
    print(mlp)
    for i in range(len(mlp.all_weights)):
        print("W", mlp.all_weights[i].shape)
        print("a", mlp.get_a_matrix_shape(i))
        print("b", mlp.all_bs[i].shape)
        print()

    for i in range(300):
        y = mlp.count_one_step()
        if i % 10 == 0:
            print(i, mlp.test_y[:4].T[0], "\n", y.T[:4].T)
        # print(mlp.all_weights[0].T)c
    # print(np.array([[0], [7]]).shape)
    # print(get_d_matrix(np.array([[0], [7]])))


if __name__ == '__main__':
    main()
