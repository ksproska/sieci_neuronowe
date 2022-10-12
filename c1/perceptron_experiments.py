from usefull import MyCustomPlot, bipolar, get_random_except_first_row, get_random_weights, reproduce_x_times, unipolar
import numpy as np
import matplotlib.pyplot as plt

from math import sqrt

def get_dimensions(n):
    tempSqrt = sqrt(n)
    divisors = []
    currentDiv = 1
    for currentDiv in range(n):
        if n % float(currentDiv + 1) == 0:
         divisors.append(currentDiv+1)
    #print divisors this is to ensure that we're choosing well
    hIndex = min(range(len(divisors)), key=lambda i: abs(divisors[i]-sqrt(n)))
    
    if divisors[hIndex]*divisors[hIndex] == n:
        return divisors[hIndex], divisors[hIndex]
    else:
        wIndex = hIndex + 1
        return divisors[hIndex], divisors[wIndex]


class PerceptronExperiment:
    x_original = np.array(
    [
        [1, 1, 1, 1],
        [0, 0, 1, 1],
        [0, 1, 0, 1]
    ]
    )

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
        plt.rcParams["figure.figsize"] = (10, 7.5)
        current_plt = MyCustomPlot()
        # repetitions = 200
        test_percent = 0.25

        teta = 0
        max_epoch = 100

        dimensions = get_dimensions(len(self.input_cases) * len(self.activation_functions) * len(self.alfas) * len(self.all_individuals_sizes))
        f, axis = plt.subplots(*dimensions)
        f.suptitle(self.figure_name)
        f.tight_layout(pad=3)
        # f.set_size_inches(dimensions[1]*8, dimensions[0]*8, forward=True)
        order_numb = 0
        for input_case in self.input_cases:
            for input_func in self.activation_functions:
                for alfa in self.alfas:
                    for individuals_size in self.all_individuals_sizes:
                        repetitions = int(individuals_size/4)
                        apply_estimate_func = np.vectorize(lambda v: self.func_types[input_func](teta, v))

                        d_case = self.d_cases[input_case][self.func_types[input_func]]
                        x_all = reproduce_x_times(self.x_original, repetitions)
                        d_all = reproduce_x_times(d_case, repetitions)
                        x_all = x_all + get_random_except_first_row(x_all.shape)

                        test_size = int(x_all.shape[1] * test_percent)
                        train_size = int(x_all.shape[1] - test_size)
                        x_train, x_test = x_all[:, :train_size], x_all[:, train_size:]
                        d_train, d_test = d_all[:train_size], d_all[train_size:]
                        weights = get_random_weights(x_all.shape[0])

                        epoch_count = 0
                        y_train = None
                        while np.mean(d_train == y_train) < 1.00 and epoch_count < max_epoch:
                            epoch_count += 1
                            count = x_train.T @ weights
                            y_train = apply_estimate_func(count)
                            dw = (d_train - y_train) @ x_train.T
                            weights = weights + alfa * dw

                        count = x_test.T @ weights
                        y_test = apply_estimate_func(count)
                        matching_percent = np.mean(d_test == y_test) * 100

                        current_plt.set_plot(axis[order_numb // dimensions[1], order_numb % dimensions[1]])
                        # current_plt.tight_layout(pad=5.0)
                        current_plt.set_title(f'{input_case} - {input_func} - size {individuals_size}\nepochs: {epoch_count} - alfa: {alfa} - {matching_percent}%')
                        current_plt.scatter(x_test[1, :], x_test[2, :], d_test)
                        current_plt.plot_line(0.0, 1.0, lambda x_vals: (-weights[1] * x_vals - weights[0]) / weights[2])
                        order_numb += 1
        plt.show()


if __name__ == '__main__':
    experiment = PerceptronExperiment("all test cases", ["AND", "OR", "XOR"], ["unipolar", "bipolar"], [0.1], [200])
    experiment.run()
    experiment = PerceptronExperiment("unipolar - different alfas", ["AND"], ["unipolar"], [0.001, 0.01, 0.1, 1], [200])
    experiment.run()
    experiment = PerceptronExperiment("bipolar - different alfas", ["AND"], ["bipolar"], [0.001, 0.01, 0.1, 1], [200])
    experiment.run()
    experiment = PerceptronExperiment("unipolar vs bipolar - different alfas", ["AND"], ["unipolar", "bipolar"], [0.001, 0.01, 0.1, 1], [200])
    experiment.run()
    experiment = PerceptronExperiment("different case sizes", ["AND"], ["unipolar", "bipolar"], [0.1], [16, 32, 50, 100])
    experiment.run()
