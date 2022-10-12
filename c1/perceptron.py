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

d_cases = {
    "AND": d_and,
    "OR": d_or,
    "XOR": d_xor,
}

func_types = {
    "unipolar": unipolar,
    "bipolar": bipolar,
}


def main():
    current_plt = CurrentPlot()
    repetitions = 200
    test_percent = 0.25

    alfa = 0.1
    teta = 0
    max_epoch = 100

    f, axis = plt.subplots(2, 3)
    f.set_size_inches(16, 9, forward=True)
    for plt_y, input_case in enumerate(["AND", "OR", "XOR"]):
        for plt_x, input_func in enumerate(["unipolar", "bipolar"]):
            apply_estimate_func = np.vectorize(lambda v: func_types[input_func](teta, v))

            d_case = d_cases[input_case][func_types[input_func]]
            x_all = reproduce_x_times(x_original, repetitions)
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

            current_plt.set_plot(axis[plt_x, plt_y])
            current_plt.set_title(f'{input_case} - {input_func} - epochs: {epoch_count} - alfa: {alfa} - {matching_percent}%')
            current_plt.scatter(x_test[1, :], x_test[2, :], d_test)
            current_plt.plot_line(0.0, 1.0, lambda x_vals: (-weights[1] * x_vals - weights[0]) / weights[2])
    plt.show()


if __name__ == '__main__':
    main()
