import datetime
import random
import time

import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.datasets import mnist
from activation_functions import *
from utils import *

plt.style.use('ggplot')


class MLP:
    def __init__(self, learning_rate, neurons_in_hidden_layers, activation_fun, output_classes_numb=10):
        self.iterations = 0
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
        self.all_bs = [random.random() for i in range(len(self.neurons_in_layers) - 1)]
        self.d = get_d_matrix(self.train_y, output_classes_numb)
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

        dz = y - self.d
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

    def get_predictions(self):
        f = self.activation_fun
        x = self.test_X.T
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

        a = np.zeros(y.shape)
        for row in range(y.shape[1]):
            m = np.argmax(y[:, row])
            a[m][row] = 1
        diff = a - self.d_test
        diff = np.abs(diff).astype(int)
        count = np.count_nonzero(diff) / 2
        return (y.shape[1] - count) / y.shape[1]

    def __str__(self):
        return f"{self.learning_rate} {self.neurons_in_layers[1:-1]} {self.activation_fun.__name__}"


def run_simulation(mlps, iterations):
    fig_path = f'plots/fig_{datetime.datetime.now()}.png'
    for i in iterations:
        smart_iterator = tqdm(range(i), colour='BLUE')
        smart_iterator.set_postfix_str(f'{i}/{len(iterations)}')
        for _ in smart_iterator:
            for mlp in mlps.keys():
                mlp.count_one_step()
                prediction = mlp.get_predictions()
                mlps[mlp].append(prediction)
        for mlp in mlps.keys():
            plt.plot(mlps[mlp], label=str(mlp))
        plt.ylim(0, 1)
        plt.xlabel("iterations")
        plt.ylabel("accuracy")
        plt.legend()
        plt.savefig(fig_path)
        plt.show()


def main():
    mlps = {
        MLP(1.3, [5], tanh): [],
        MLP(1.3, [10], tanh): [],
        MLP(1.3, [70], tanh): [],
        MLP(1.3, [150], tanh): [],
    }
    iterations = [10, 5, 5, 5, 10, 10]
    run_simulation(mlps, iterations)


if __name__ == '__main__':
    main()
