import datetime
import inspect
import math
import random
import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from keras.datasets import mnist
import regex

from activation_functions import *
from utils import *

plt.style.use('ggplot')

colors = 'bgrcmyk'


class MLP:
    def __init__(self, learning_rate, neurons_in_hidden_layers, activation_fun, w_range, batch_size,
                 output_classes_numb=10):
        self.iterations = 0
        self.batch_size = batch_size
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

        self.standard_dict = {
            "batch size: ": f"{self.batch_size}",
            "weights ranges: ": f"{-self.w_range}:{self.w_range}",
            "activation function: ": f"{self.activation_fun.__name__}",
            "neurons: ": f"{self.neurons_in_layers}",
            "learning rate: ": f"{self.learning_rate}",
        }
        self.show_dict = self.standard_dict.copy()

    def diff_params_str(self, other):
        diffrent_params = []

    def get_weights_matrix_shape(self, first_inx):
        return [self.neurons_in_layers[first_inx + 1], self.neurons_in_layers[first_inx]]

    def train_one_epoch(self):
        self.iterations += 1
        f = self.activation_fun
        f_prim = self.activation_fun_derivative
        all_x = self.train_X.T
        all_d = self.d_train
        numb_of_samples = all_x.shape[1]
        batch_count = np.ceil(numb_of_samples / self.batch_size)

        for batch_numb in range(int(batch_count)):
            x = all_x[:, batch_numb * self.batch_size:min((batch_numb + 1) * self.batch_size, numb_of_samples)]
            d = all_d[:, batch_numb * self.batch_size:min((batch_numb + 1) * self.batch_size, numb_of_samples)]

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

            dz = y - d
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
                dw = dz @ a.T / numb_of_samples
                db = np.sum(dz, axis=1, keepdims=True) / numb_of_samples
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

    def update_params(self, **kwargs):
        for k, v in kwargs.items():
            self.__dict__[k] = v

    def __str__(self):
        output = ""
        for param, val in self.show_dict.items():
            if val != "":
                output += param + val + " "
        return output
        # return f"{self.learning_rate} {self.neurons_in_layers[1:-1]} {self.activation_fun.__name__} " \
        #        f"{-self.w_range}:{self.w_range} {self.batch_size}"


def run_simulation(mlps, iterations, show_train=False):
    figure_title = ""
    if len(list(mlps.keys())) == 1:
        for param in list(mlps.keys())[0].show_dict:
            list(mlps.keys())[0].show_dict[param] = ""
    else:
        for mlp in mlps:
            for mlp2 in mlps:
                if mlp != mlp2:
                    for param in mlp.show_dict:
                        if mlp.standard_dict[param] == mlp2.standard_dict[param]:
                            mlp.show_dict[param] = ""
                            mlp2.show_dict[param] = ""
    for param in list(mlps.keys())[0].show_dict:
        val = list(mlps.keys())[0].show_dict[param]
        if val == "":
            figure_title += param + list(mlps.keys())[0].standard_dict[param] + "\n"

    fig_path = f'plots/fig_{datetime.datetime.now()}.png'
    for current, i in enumerate(iterations):
        smart_iterator = tqdm(range(i), colour='BLUE')
        smart_iterator.set_postfix_str(f'{current}/{len(iterations)}')
        for _ in smart_iterator:
            for mlp in mlps.keys():
                mlp.train_one_epoch()
                loss_train, prediction_train = mlp.get_predictions(mlp.train_X, mlp.d_train)
                loss_test, prediction_test = mlp.get_predictions(mlp.test_X, mlp.d_test)
                mlps[mlp][0].append(prediction_test)
                mlps[mlp][1].append(prediction_train)
                mlps[mlp][2].append(loss_test)
                mlps[mlp][3].append(loss_train)
            smart_iterator.set_postfix_str(", ".join([f"{np.round(mlps[x][0][-1] * 100, 2)}%" for x in list(mlps.keys())]))

        fig, axis = plt.subplots(1, 2)
        plt.subplots_adjust(top=0.73)
        fig.set_size_inches(11, 6)
        for j, mlp in enumerate(mlps.keys()):
            print(mlp)
            plt.sca(axis[0])
            plt.plot(mlps[mlp][0], label=str(mlp), c=colors[j])
            if show_train:
                plt.plot(mlps[mlp][1], "--", c=colors[j])
            plt.sca(axis[1])
            plt.plot(mlps[mlp][2], label=str(mlp), c=colors[j])
            if show_train:
                plt.plot(mlps[mlp][3], "--", c=colors[j])

        plt.sca(axis[0])
        plt.title("accuracy\n" + figure_title)
        plt.ylim(0, 1)
        plt.xlabel("iterations")
        plt.ylabel("accuracy")
        plt.legend()

        plt.sca(axis[1])
        plt.title("loss\n" + figure_title)
        plt.xlabel("iterations")
        plt.ylabel("loss")
        plt.legend()

        plt.savefig(fig_path)
        plt.show()


def main():
    mlps = {
        MLP(10.0, [150, 100, 50], tanh, 0.01, 64): [[], [], [], []],
        # MLP(5.0, [150, 100, 50], tanh, 0.01, 64): [[], [], [], []],
        # MLP(1.3, [70], tanh, 0.01): [[], [], [], []],
        # MLP(1.3, [70], tanh, 0.1): [[], [], [], []],
        # MLP(1.3, [70], tanh, 10): [[], [], [], []],
        # MLP(1.3, [70], tanh, 20): [[], [], [], []],
    }

    iterations = [1] * 60
    run_simulation(mlps, iterations)


if __name__ == '__main__':
    main()
