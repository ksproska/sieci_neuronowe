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

if __name__ == '__main__':
    repetitions = 200
    test_percent = 0.25
    f, axis = plt.subplots(2, 3)
    f.set_size_inches(16, 9, forward=True)
    for plt_y, input_case in enumerate(["AND", "OR", "XOR"]):
        for plt_x, input_func in enumerate(["unipolar", "bipolar"]):
            current_plt = axis[plt_x, plt_y]
            for alfa in [0.1]:
                epoch_numb = 100
                estimate_func = func_types[input_func]
                d_case = d_cases[input_case][estimate_func]

                x_all = randomize(merge_multiple_times(x_original, repetitions))
                d_all = merge_multiple_times(d_case, repetitions)

                train_size = int(x_all.shape[1] - x_all.shape[1] * test_percent)
                x_train, x_test = x_all[:, :train_size], x_all[:, train_size:]
                d_train, d_test = d_all[:train_size], d_all[train_size:]
                weights = get_random_weights(x_all.shape[0])

                epoch_count = 0
                y = None
                while 1.0 - np.mean(d_train == y) > 0.01 and epoch_count < 100:
                    epoch_count += 1
                    c = count_cost(weights, x_train.T)
                    y = np.vectorize(lambda v: estimate_func(0, v))(c)
                    dw = delta_w(d_train, y, x_train)
                    weights = calculate_new_weight(weights, alfa, dw)

                c = count_cost(weights, x_test.T)
                y = apply_func(c, lambda v: estimate_func(0, v))
                diff = np.mean(d_test == y) * 100

                current_plt.set_title(f'{input_case} - {input_func} - epochs: {epoch_count} - alfa: {alfa} - {diff}%')
                current_plt.scatter(x_test[1, :], x_test[2, :])

                x_all = np.linspace(0.0, 1.0, 10)
                current_plt.plot(x_all, (-weights[1] * x_all - weights[0]) / weights[2])
                current_plt.axhline(y=0, color="k")
                current_plt.axvline(x=0, color="k")
                current_plt.grid(True)
    plt.show()
