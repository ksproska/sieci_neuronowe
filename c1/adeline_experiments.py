import numpy as np
from matplotlib import pyplot as plt

from usefull import AllPlots, reproduce_x_times, get_random_except_first_row, get_random_weights, count_cost, \
    apply_func, sign_bipolar


class Adaline:
    def __init__(self, x_all, d_all, test_percent):
        test_size = int(x_all.shape[1] * test_percent)
        train_size = int(x_all.shape[1] - test_size)
        self.x_train, self.x_test = x_all[:, :train_size], x_all[:, train_size:]
        self.d_train, self.d_test = d_all[:train_size], d_all[train_size:]
        self.weights = get_random_weights(x_all.shape[0])

    def count(self, alfa, max_epoch):
        self.weights.reshape((self.x_test.shape[0], 1))
        epoch_numb = 100
        for epoch in range(epoch_numb):
            for i in range(self.x_train.shape[1]):
                z = self.weights.T @ self.x_train[:, i].reshape(3, 1)
                delta_root = self.d_train[i] - z
                self.weights = self.weights + (alfa * delta_root * self.x_train[:, i].reshape(3, 1))

        z = count_cost(self.weights, self.x_test.T)
        self.matching_percent = np.mean(self.d_test == np.vectorize(sign_bipolar)(z)).round() * 100
        self.title = f'alfa: {alfa} - {self.matching_percent}%'

    def draw(self, current_plt, title_prev):
        current_plt.set_title(title_prev + "\n" + self.title)
        current_plt.scatter(self.x_test[1, :], self.x_test[2, :], self.d_test)
        current_plt.plot_line(-1.0, 1.0, lambda x_vals: (-self.weights[0][1] * x_vals - self.weights[0][0]) / self.weights[0][2])


class AdalineExperiments:
    x_original = np.array(
        [
            [1, 1, 1, 1],
            [0, 0, 1, 1],
            [0, 1, 0, 1]
        ]
    )

    d_and = np.array([-1, -1, -1, 1])
    d_or = np.array([-1, 1, 1, 1])
    d_xor = np.array([-1, 1, 1, -1])

    d_cases = {
        "AND": d_and,
        "OR": d_or,
        "XOR": d_xor,
    }

    def __init__(self, figure_name, input_cases, alfas, all_individuals_sizes):
        self.figure_name = figure_name
        self.input_cases = input_cases
        self.alfas = alfas
        self.all_individuals_sizes = all_individuals_sizes

    def run(self):
        test_percent = 0.25
        teta = 0
        max_epoch = 1000

        all_plots = AllPlots(
            len(self.input_cases) * len(self.alfas) * len(self.all_individuals_sizes),
            self.figure_name)

        for input_case in self.input_cases:
            for alfa in self.alfas:
                for individuals_size in self.all_individuals_sizes:
                    repetitions = int(individuals_size / 4)

                    d_case = self.d_cases[input_case]
                    x_all = reproduce_x_times(self.x_original, repetitions)
                    d_all = reproduce_x_times(d_case, repetitions)
                    x_all = x_all + get_random_except_first_row(x_all.shape)

                    adaline = Adaline(x_all, d_all, test_percent)
                    adaline.count(alfa, max_epoch)

                    adaline.draw(all_plots.current, f'{input_case} - size: {individuals_size}')
                    all_plots.next()
        plt.show()


if __name__ == '__main__':
    experiment = AdalineExperiments("all test cases", ["AND", "OR", "XOR"], [0.001, 0.01], [200])
    experiment.run()
    # experiment = AdalineExperiments("unipolar - different alfas", ["AND"], [0.001, 0.01, 0.1, 1], [200])
    # experiment.run()
    # experiment = AdalineExperiments("bipolar - different alfas", ["AND"], [0.001, 0.01, 0.1, 1], [200])
    # experiment.run()
    # experiment = AdalineExperiments("unipolar vs bipolar - different alfas", ["AND"], [0.001, 0.01, 0.1, 1], [200])
    # experiment.run()
    # experiment = AdalineExperiments("different case sizes", ["AND"], [0.1], [16, 32, 50, 100])
    # experiment.run()
