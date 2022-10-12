from usefull import bipolar, get_random_except_first_row, get_random_weights, reproduce_x_times, unipolar, AllPlots
import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, x_all, d_all, test_percent):
        test_size = int(x_all.shape[1] * test_percent)
        train_size = int(x_all.shape[1] - test_size)
        self.x_train, self.x_test = x_all[:, :train_size], x_all[:, train_size:]
        self.d_train, self.d_test = d_all[:train_size], d_all[train_size:]
        self.weights = get_random_weights(x_all.shape[0])

    def count(self, alfa, apply_estimate_func, max_epoch):
        epoch_count = 0
        y_train = None
        while np.mean(self.d_train == y_train) < 1.00 and epoch_count < max_epoch:
            epoch_count += 1
            count = self.x_train.T @ self.weights
            y_train = apply_estimate_func(count)
            dw = (self.d_train - y_train) @ self.x_train.T
            self.weights = self.weights + alfa * dw

        count = self.x_test.T @ self.weights
        y_test = apply_estimate_func(count)
        self.matching_percent = np.mean(self.d_test == y_test) * 100
        self.title = f'epochs: {epoch_count} - alfa: {alfa} - {self.matching_percent}%'

    def draw(self, current_plt, title_prev, left):
        current_plt.set_title(title_prev + "\n" + self.title)
        current_plt.scatter(self.x_test[1, :], self.x_test[2, :], self.d_test)
        current_plt.plot_line(left, 1.0, lambda x_vals: (-self.weights[1] * x_vals - self.weights[0]) / self.weights[2])


class PerceptronExperiments:
    x_original = {
        unipolar: np.array(
            [
                [1, 1, 1, 1],
                [0, 0, 1, 1],
                [0, 1, 0, 1]
            ]
        ),
        bipolar: np.array(
            [
                [1, 1, 1, 1],
                [-1, -1, 1, 1],
                [-1, 1, -1, 1]
            ]
        ),
    }

    d_and = {
        unipolar: np.array([0, 0, 0, 1]),
        bipolar: np.array([-1, -1, -1, 1]),
    }

    d_or = {
        unipolar: np.array([0, 1, 1, 1]),
        bipolar: np.array([-1, 1, 1, 1]),
    }
    d_xor = {
        unipolar: np.array([0, 1, 1, 0]),
        bipolar: np.array([-1, 1, 1, -1]),
    }

    d_cases = {
        "AND": d_and,
        "OR": d_or,
        "XOR": d_xor,
    }

    func_types = {
        "unipolar": unipolar,
        "bipolar": bipolar,
    }

    def __init__(self, figure_name, input_cases, activation_functions, alfas, all_individuals_sizes):
        self.figure_name = figure_name
        self.input_cases = input_cases
        self.activation_functions = activation_functions
        self.alfas = alfas
        self.all_individuals_sizes = all_individuals_sizes
    
    def run(self):
        test_percent = 0.25
        teta = 0
        max_epoch = 100

        all_plots = AllPlots(len(self.input_cases) * len(self.activation_functions) * len(self.alfas) * len(self.all_individuals_sizes), self.figure_name)

        for input_case in self.input_cases:
            for input_func in self.activation_functions:
                for alfa in self.alfas:
                    for individuals_size in self.all_individuals_sizes:
                        repetitions = int(individuals_size/4)
                        apply_estimate_func = np.vectorize(lambda v: self.func_types[input_func](teta, v))

                        d_case = self.d_cases[input_case][self.func_types[input_func]]
                        x_all = reproduce_x_times(self.x_original[self.func_types[input_func]], repetitions)
                        d_all = reproduce_x_times(d_case, repetitions)
                        x_all = x_all + get_random_except_first_row(x_all.shape)

                        perceptron = Perceptron(x_all, d_all, test_percent)
                        perceptron.count(alfa, apply_estimate_func, max_epoch)

                        perceptron.draw(all_plots.current, f'{input_case} - {input_func} - size: {individuals_size}', 0.0 if input_func == "unipolar" else -1.0)
                        all_plots.next()
        plt.show()


if __name__ == '__main__':
    experiment = PerceptronExperiments("all test cases", ["AND", "OR", "XOR"], ["unipolar", "bipolar"], [0.1], [200])
    experiment.run()
    experiment = PerceptronExperiments("unipolar - different alfas", ["AND"], ["unipolar"], [0.001, 0.01, 0.1, 1], [200])
    experiment.run()
    experiment = PerceptronExperiments("bipolar - different alfas", ["AND"], ["bipolar"], [0.001, 0.01, 0.1, 1], [200])
    experiment.run()
    experiment = PerceptronExperiments("unipolar vs bipolar - different alfas", ["AND"], ["unipolar", "bipolar"], [0.001, 0.01, 0.1, 1], [200])
    experiment.run()
    experiment = PerceptronExperiments("different case sizes", ["AND"], ["unipolar", "bipolar"], [0.1], [16, 32, 50, 100])
    experiment.run()
