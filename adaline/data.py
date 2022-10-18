import random

import numpy as np
import matplotlib.pyplot as plt
import statistics

x_unipolar = np.array(
    [
        [1, 1, 1, 1],
        [0, 0, 1, 1],
        [0, 1, 0, 1]
    ]
)
d_bipolar = np.array([[-1, -1, -1, 1]])


def get_random_weights(length, min_val, max_val):
    return np.array([[random.uniform(min_val, max_val) for _ in range(length)]]).T


def sign_bipolar(x):
    if x < 0:
        return -1
    return 1


class Adaline:
    def __init__(self, x_train, d_train, x_test, d_test, alfa, allowed_error, wrange):
        self.x_train = x_train
        self.x_test = x_test
        self.d_train = d_train
        self.d_test = d_test
        self.alfa = alfa
        self.allowed_error = allowed_error
        self.wrange = wrange
        self.weights_output = []
        self.matching_percents = []
        self.epoch_nums = []

    @property
    def to_string(self):
        return f'epochs:     {self.average_epoch_count}\n' \
               f'matching:   {self.average_matching_percent}%\n' \
               f'alfa:       {self.alfa}\n' \
               f'w. range:   {self.wrange}\n' \
               f'train size: {self.x_train.shape[1]}\n' \
               f'max err:    {self.allowed_error}'

    def count(self):
        epoch_count = 0
        err = None
        weights = get_random_weights(self.x_train.shape[0], *self.wrange)
        while err is None or err > self.allowed_error and epoch_count < 100:
            epoch_count += 1
            z = self.x_train.T @ weights
            delta_root = self.d_train.T - z
            err = np.mean(np.square(delta_root))
            weights = weights + (self.alfa * self.x_train @ delta_root)

        z = self.x_test.T @ weights

        self.weights_output.append(weights)
        self.matching_percents.append(np.mean(self.d_test == np.vectorize(sign_bipolar)(z)).round() * 100)
        self.epoch_nums.append(epoch_count)

    @property
    def average_matching_percent(self):
        return statistics.mean(self.matching_percents)

    @property
    def average_epoch_count(self):
        return statistics.mean(self.epoch_nums)

    def display(self):
        plt.rcParams.update({
            'font.family': 'monospace'
        })
        plt.title(f"Adaline\n{self.to_string}", ha="left", x=-.12)
        plt.xlim(-0.25, 1.25)
        plt.ylim(-0.25, 1.25)
        plt.style.use('ggplot')
        x_range = np.arange(-2, 4)
        plt.scatter(self.x_test[1, :], self.x_test[2, :], c=self.d_test)
        for w in self.weights_output:
            plt.plot(x_range, (-w[1] * x_range - w[0]) / w[2])
        plt.show()


def reproduce_x_times(input, num_to_merge):
    output = input.T
    for i in range(num_to_merge - 1):
        output = np.concatenate((output, input.T))
    return output.T


def get_random_except_first_row(shape):
    rand_matrix = np.random.rand(shape[0], shape[1]) / 10 - 0.05
    rand_matrix[0, :] = 0
    return rand_matrix


def unipolar(teta, z):
    return int(z > teta)


def bipolar(teta, z):
    return 2 * (z > teta) - 1


def main():
    repetitions = 200
    x_all = reproduce_x_times(x_unipolar, repetitions)
    d_all = reproduce_x_times(d_bipolar, repetitions)
    x_all = x_all + get_random_except_first_row(x_all.shape)

    test_size = int(x_all.shape[1] * 0.25)
    train_size = int(x_all.shape[1] - test_size)
    x_train, x_test = x_all[:, :train_size], x_all[:, train_size:]
    d_train, d_test = d_all[:, :train_size], d_all[:, train_size:]

    perceptron = Adaline(x_train, d_train, x_test, d_test, 0.0001, 0.4, (-0.1, 0.1))
    perceptron.count()
    perceptron.display()
    print(perceptron.to_string)


if __name__ == '__main__':
    main()
