import numpy as np
from matplotlib import pyplot as plt

from usefull import AllPlots, reproduce_x_times, get_random_except_first_row, get_random_weights, count_cost, sign_bipolar, apply_func


class Adaline:
    def __init__(self, x_all, d_all, test_percent):
        test_size = int(x_all.shape[1] * test_percent)
        train_size = int(x_all.shape[1] - test_size)
        self.x_train, self.x_test = x_all[:, :train_size], x_all[:, train_size:]
        self.d_train, self.d_test = d_all[:train_size], d_all[train_size:]
        self.weights = get_random_weights(x_all.shape[0])

    def count(self, alfa, allowed_error):
        epoch_numb = 1000
        err = None 
        epoch = 0

        while err is None or err > allowed_error and epoch < epoch_numb:
            epoch += 1
            z = self.x_train.T @ self.weights
            delta_root = self.d_train - z
            err = np.mean(np.square(delta_root))
            self.weights = self.weights + (alfa * self.x_train @ delta_root)

        z = count_cost(self.weights, self.x_test.T)
        self.matching_percent = np.mean(self.d_test == apply_func(z, sign_bipolar)).round() * 100

        self.title = f'alfa: {alfa} - epoch: {epoch} - {self.matching_percent}%'
        print(self.weights)

    def draw(self, current_plt, title_prev):
        current_plt.set_title(title_prev + "\n" + self.title)
        current_plt.scatter(self.x_test[1, :], self.x_test[2, :], self.d_test)
        current_plt.plot_line(0.0, 1.0, lambda x_vals: (-self.weights[1] * x_vals - self.weights[0]) / self.weights[2])


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
        allowed_error = 0.25

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
                    adaline.count(alfa, allowed_error)

                    adaline.draw(all_plots.current, f'{input_case} - size: {individuals_size}')
                    all_plots.next()
        plt.show()


if __name__ == '__main__':
    experiment = AdalineExperiments("all test cases", ["AND"], [0.001, 0.01], [100, 1000])
    experiment.run()
    # experiment = AdalineExperiments("unipolar - different alfas", ["AND"], [0.001, 0.01, 0.1, 1], [200])
    # experiment.run()
    # experiment = AdalineExperiments("bipolar - different alfas", ["AND"], [0.001, 0.01, 0.1, 1], [200])
    # experiment.run()
    # experiment = AdalineExperiments("unipolar vs bipolar - different alfas", ["AND"], [0.001, 0.01, 0.1, 1], [200])
    # experiment.run()
    # experiment = AdalineExperiments("different case sizes", ["AND"], [0.1], [16, 32, 50, 100])
    # experiment.run()
