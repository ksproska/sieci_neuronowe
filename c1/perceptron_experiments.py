from c1.perceptron import Perceptron
from usefull import bipolar, get_random_except_first_row, reproduce_x_times, unipolar, AllPlots
import numpy as np
import matplotlib.pyplot as plt


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

    def __init__(self, figure_name, input_cases, activation_functions, alfas, all_individuals_sizes, weights_ranges):
        self.figure_name = figure_name
        self.input_cases = input_cases
        self.activation_functions = activation_functions
        self.alfas = alfas
        self.all_individuals_sizes = all_individuals_sizes
        self.weights_ranges = weights_ranges
    
    def run(self):
        test_percent = 0.25
        teta = 0
        max_epoch = 100

        all_plots = AllPlots(len(self.input_cases) * len(self.activation_functions) * len(self.alfas) * len(self.all_individuals_sizes) * len(self.weights_ranges), self.figure_name)

        for input_case in self.input_cases:
            for input_func in self.activation_functions:
                for alfa in self.alfas:
                    for individuals_size in self.all_individuals_sizes:
                        for wrange in self.weights_ranges:
                            repetitions = int(individuals_size/4)
                            apply_estimate_func = np.vectorize(lambda v: self.func_types[input_func](teta, v))

                            for i in range(10):
                                d_case = self.d_cases[input_case][self.func_types[input_func]]
                                x_all = reproduce_x_times(self.x_original[self.func_types[input_func]], repetitions)
                                d_all = reproduce_x_times(d_case, repetitions)
                                x_all = x_all + get_random_except_first_row(x_all.shape)

                                perceptron = Perceptron(x_all, d_all, test_percent, wrange)
                                
                                perceptron.count(alfa, apply_estimate_func)

                                perceptron.draw(all_plots.current, f'{input_case} - {input_func} - size: {individuals_size}\nweights start range: {wrange}')
                                perceptron.draw_line(all_plots.current, 0.0 if input_func == "unipolar" else -1.0)

                            all_plots.next()
        plt.show()


if __name__ == '__main__':
    # experiment = PerceptronExperiments("all test cases", ["AND", "OR", "XOR"], ["unipolar", "bipolar"], [0.1], [200], [(0.1, 0.2)])
    # experiment.run()
    experiment = PerceptronExperiments("unipolar - different alfas", ["AND"], ["unipolar"], [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 0.8, 1], [200], [(0.1, 0.2)])
    experiment.run()
    # experiment = PerceptronExperiments("bipolar - different alfas", ["AND"], ["bipolar"], [0.001, 0.01, 0.1, 1], [200], [(0.1, 0.2)])
    # experiment.run()
    experiment = PerceptronExperiments("unipolar vs bipolar", ["AND"], ["unipolar", "bipolar"], [0.01, 0.01, 0.01, 0.01, 0.01], [200], [(-0.1, 0.1)])
    experiment.run()
    # experiment = PerceptronExperiments("different case sizes", ["AND"], ["unipolar", "bipolar"], [0.1], [16, 32, 50, 100], [(0.1, 0.2)])
    # experiment.run()
    experiment = PerceptronExperiments("different weights start ranges", ["AND"], ["unipolar"], [0.1], [200], [(-1.0, 1.0), (-0.9, 0.9), (-0.8, 0.8), (-0.6, 0.6), (-0.5, 0.5), (-0.3, 0.3), (-0.2, 0.2), (-0.1, 0.1)])
    experiment.run()
