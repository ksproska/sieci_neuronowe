import datetime
import random
import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from keras.datasets import mnist
from activation_functions import *
from utils import *

plt.style.use('ggplot')

colors = 'bgrcmyk'


class MLP:
    def __init__(self, learning_rate, neurons_in_hidden_layers, activation_fun, w_range, output_classes_numb=10):
        self.iterations = 0
        self.w_range = w_range
        self.activation_fun = activation_fun
        self.activation_fun_derivative = {
            relu: relu_derivative,
            sigmoid: sigmoid_derivative,
            tanh: tanh_derivative
        }[activation_fun]
        self.learning_rate = learning_rate

        (train_X, train_y), (test_X, test_y) = mnist.load_data()
        train_X = train_X.reshape((train_X.shape[0], -1))
        train_y = train_y.reshape((train_y.shape[0], -1))
        test_X = test_X.reshape((test_X.shape[0], -1))
        test_y = test_y.reshape((test_y.shape[0], -1))
        (self.train_X, self.train_y), (self.test_X, self.test_y) = (train_X / 255, train_y), (test_X / 255, test_y)

        self.neurons_in_layers = [self.train_X.shape[1]] + neurons_in_hidden_layers + [output_classes_numb]
        self.all_weights = [np.random.randn(*self.get_weights_matrix_shape(i)) for i in
                            range(len(self.neurons_in_layers) - 1)]
        self.all_bs = [(random.random() * 2 - 1) * w_range for _ in range(len(self.neurons_in_layers) - 1)]
        self.d_train = get_d_matrix(self.train_y, output_classes_numb)
        self.d_test = get_d_matrix(self.test_y, output_classes_numb)

    def get_weights_matrix_shape(self, first_inx):
        return [self.neurons_in_layers[first_inx + 1], self.neurons_in_layers[first_inx]]

    def count_one_step(self):
        self.iterations += 1
        f = self.activation_fun
        f_prim = self.activation_fun_derivative
        x = self.train_X.T
        a_all = [x]
        z_all = []

        for j in range(len(self.neurons_in_layers) - 1):
            w = self.all_weights[j]
            b = self.all_bs[j]
            z = w @ a_all[j] + b
            z_all.append(z)
            a = f(z)
            a_all.append(a)
        a_all[-1] = softmax(z_all[-1])
        y = a_all[-1]

        dz = y - self.d_train
        dz_all = [dz]

        for j in range(len(self.neurons_in_layers) - 2, 0, -1):
            prev_w = self.all_weights[j]
            prev_dz = dz_all[-1]
            dz = (prev_w.T @ prev_dz) * f_prim(z_all[j - 1])
            dz_all.append(dz)

        dz_all = dz_all[::-1]
        for j in range(len(self.neurons_in_layers) - 1):
            a = a_all[j]
            dz = dz_all[j]
            dw = dz @ a.T / x.shape[1]
            db = np.sum(dz, axis=1, keepdims=True) / x.shape[1]
            self.all_weights[j] = self.all_weights[j] - dw * self.learning_rate
            self.all_bs[j] = self.all_bs[j] - db * self.learning_rate

        return y

    def get_predictions(self, x, d):
        f = self.activation_fun
        a = x.T
        z = None

        for j in range(len(self.neurons_in_layers) - 1):
            w = self.all_weights[j]
            b = self.all_bs[j]
            z = w @ a + b
            a = f(z)
        y = softmax(z)

        best = np.argmax(y, axis=0)
        labels = np.argmax(d, axis=0)
        return calculate_loss(y, d), np.mean(best == labels)

    def __str__(self):
        return f"{self.learning_rate} {self.neurons_in_layers[1:-1]} {self.activation_fun.__name__} {-self.w_range}:{self.w_range}"


def run_simulation(mlps, iterations):
    fig_path = f'plots/fig_{datetime.datetime.now()}.png'
    for current, i in enumerate(iterations):
        smart_iterator = tqdm(range(i), colour='BLUE')
        smart_iterator.set_postfix_str(f'{current}/{len(iterations)}')
        for _ in smart_iterator:
            for mlp in mlps.keys():
                mlp.count_one_step()
                loss_train, prediction_train = mlp.get_predictions(mlp.train_X, mlp.d_train)
                loss_test, prediction_test = mlp.get_predictions(mlp.test_X, mlp.d_test)
                mlps[mlp][0].append(prediction_test)
                mlps[mlp][1].append(prediction_train)
                mlps[mlp][2].append(loss_test)
                mlps[mlp][3].append(loss_train)

        fig, axis = plt.subplots(1, 2)
        fig.set_size_inches(10, 5)
        for j, mlp in enumerate(mlps.keys()):
            plt.sca(axis[0])
            plt.plot(mlps[mlp][0], label=str(mlp) + " test", c=colors[j])
            # plt.plot(mlps[mlp][1], "--", label=str(mlp) + " train", c=colors[j])
            plt.sca(axis[1])
            plt.plot(mlps[mlp][2], label=str(mlp) + " test", c=colors[j])
            # plt.plot(mlps[mlp][3], "--", label=str(mlp) + " train", c=colors[j])

        plt.sca(axis[0])
        plt.ylim(0, 1)
        plt.xlabel("iterations")
        plt.ylabel("accuracy")
        plt.legend()

        plt.sca(axis[1])
        plt.xlabel("iterations")
        plt.ylabel("accuracy")
        plt.legend()

        plt.savefig(fig_path)
        plt.show()


def main():
    mlps = {
        MLP(20.0, [], tanh, 0.01): [[], [], [], []],
        # MLP(1.3, [70], tanh, 0.01): [[], [], [], []],
        # MLP(1.3, [70], tanh, 0.1): [[], [], [], []],
        # MLP(1.3, [70], tanh, 10): [[], [], [], []],
        # MLP(1.3, [70], tanh, 20): [[], [], [], []],
    }
    iterations = [20] * 30
    run_simulation(mlps, iterations)


if __name__ == '__main__':
    main()
