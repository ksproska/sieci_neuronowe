from usefull import *
import numpy as np
import matplotlib.pyplot as plt

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

if __name__ == '__main__':
    repetitions = 200
    test_percent = 0.25
    f, axis = plt.subplots(2, 3)
    plt_y = 0
    for experiment_case in [(d_and, "AND"), (d_or, "OR"), (d_xor, "XOR")]:
        plt_x = 0
        for func_type in [(unipolar, "unipolar"), (bipolar, "bipolar")]:
            for alfa in [0.1]:
                epoch_numb = 100

                d_case = experiment_case[0]
                estimate_func = func_type[0]
                x_all = randomize(merge_multiple_times(x_original, repetitions))
                d_all = merge_multiple_times(d_case[estimate_func], repetitions)

                train_size = int(x_all.shape[1] - x_all.shape[1] * test_percent)
                x_train, x_test = x_all[:, :train_size], x_all[:, train_size:]

                weights = get_random_weights(x_all.shape[0])
                y = None
                epoch_count = 0
                for i in range(epoch_numb):
                    epoch_count += 1
                    c = count_cost(weights, x_train.T)
                    y = apply_func(c, lambda v: estimate_func(0, v))
                    dw = delta_w(d_all[:train_size], y, x_train)
                    weights = calculate_new_weight(weights, alfa, dw)
                    if 1.0 - np.mean(d_all[:train_size] == y) < 0.01:
                        break

                c = count_cost(weights, x_test.T)
                y = apply_func(c, lambda v: estimate_func(0, v))
                diff = np.mean(d_all[train_size:] == y) * 100

                axis[plt_x, plt_y].set_title(f'{experiment_case[1]} - {func_type[1]} - epochs: {epoch_count} - alfa: {alfa} - match: {diff}%')
                axis[plt_x, plt_y].scatter(x_test[1, :], x_test[2, :])

                x_all = np.linspace(0.0, 1.0, 10)
                axis[plt_x, plt_y].plot(x_all, (-weights[1] * x_all - weights[0]) / weights[2])
                axis[plt_x, plt_y].axhline(y=0, color="k")
                axis[plt_x, plt_y].axvline(x=0, color="k")
                axis[plt_x, plt_y].grid(True)
                plt_x += 1
        plt_y += 1
    plt.show()
