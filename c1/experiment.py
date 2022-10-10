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
            for alfa in [0.01, 0.1, 1]:
                epoch_numb = 100

                d_case = experiment_case[0]
                estimate_func = func_type[0]
                x_original = x_originals[estimate_func]
                x_all = randomize(merge_multiple_times(x_original, repetitions))
                d_all = merge_multiple_times(d_case[estimate_func], repetitions)

                train_size = int(x_all.shape[1] - x_all.shape[1] * test_percent)
                x_train, x_test = x_all[:, :train_size], x_all[:, train_size:]

                weights = get_random_weights(x_all.shape[0])
                y = None
                epoch_count = 0
                for i in range(epoch_numb):
                    epoch_count += 1
                    c = cost(weights, x_train.T)
                    y = estimate(c, lambda v: estimate_func(0, v))
                    dw = delta_w(d_all[:train_size], y, x_train)
                    weights = calculate_new_weight(weights, alfa, dw)
                    if 1.0 - np.mean(d_all[:train_size] == y) < 0.01:
                        break

                c = cost(weights, x_test.T)
                y = estimate(c, lambda v: estimate_func(0, v))
                diff = np.mean(d_all[train_size:] == y) * 100

                print(f'{experiment_case[1]}\t{func_type[1]}\t{epoch_count}\t{alfa}\t{diff}%')

                plt.title(f'{experiment_case[1]} - {func_type[1]} - epochs: {epoch_count} - alfa: {alfa} - match: {diff}%')
                plt.scatter(x_test[1, :], x_test[2, :])

                x_all = np.linspace(-1.25 if estimate_func == bipolar else -0.25, 1.25, 10)
                plt.plot(x_all, (-weights[1] * x_all - weights[0]) / weights[2])
                plt.axhline(y=0, color="k")
                plt.axvline(x=0, color="k")
                plt.grid(True)
                plt.show()
