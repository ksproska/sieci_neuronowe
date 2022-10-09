from usefull import *
import numpy as np
import matplotlib.pyplot as plt

x_originals = {
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
    )
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

if __name__ == '__main__':
    repetitions = 200
    test_percent = 0.25
    for experiment_case in [(d_and, "AND"), (d_or, "OR"), (d_xor, "XOR")]:
        for func_type in [(unipolar, "unipolar"), (bipolar, "bipolar")]:
            alfa = 0.1
            epoce_numb = 100

            estimate_func = func_type[0]
            x_original = x_originals[estimate_func]
            x_all = randomize(merge_multiple_times(x_original, repetitions))
            d_all = merge_multiple_times(experiment_case[0][estimate_func], repetitions)

            train_size = int(x_all.shape[1] - x_all.shape[1] * test_percent)
            x_train, x_test = x_all[:, :train_size], x_all[:, train_size:]

            weights = get_random_weights(x_all.shape[0])
            y = None
            for i in range(epoce_numb):
                c = cost(weights, x_train.T)
                y = estimate(c, lambda v: estimate_func(0, v))
                dw = delta_w(d_all[:train_size], y, x_train)
                weights = calculate_new_weight(weights, alfa, dw)

            c = cost(weights, x_test.T)
            y = estimate(c, lambda v: estimate_func(0, v))
            diff = np.mean(d_all[train_size:] == y) * 100

            print(f'{experiment_case[1]}\t{func_type[1]}\t{epoce_numb}\t{diff}%')

            plt.title(f'{experiment_case[1]} - {func_type[1]} - {epoce_numb} - {diff}%')
            plt.plot(x_test[1, :], x_test[2, :], marker=".", linestyle='None')

            x_all = np.linspace(-1, 1.5, 10)
            y = (-weights[1] * x_all - weights[0]) / weights[2]
            plt.plot(x_all, y)
            plt.show()
