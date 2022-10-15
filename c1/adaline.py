from usefull import reproduce_x_times, get_random_weights, count_cost, apply_func, sign_bipolar, \
    get_random_except_first_row, CurrentPlot
import numpy as np
import matplotlib.pyplot as plt


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


def main():
    current_plt = CurrentPlot(plt)
    repetitions = 100
    test_percent = 0.25
    alfa = 0.001
    epoch_numb = 1000

    x_all = reproduce_x_times(x_original, repetitions)
    x_all = x_all + get_random_except_first_row(x_all.shape)
    d_all = reproduce_x_times(d_and, repetitions)

    test_size = int(x_all.shape[1] * test_percent)
    train_size = int(x_all.shape[1] - test_size)
    x_train, x_test = x_all[:, :train_size], x_all[:, train_size:]

    weights = get_random_weights(x_all.shape[0])
    d_train, d_test = d_all[:train_size], d_all[train_size:]

    allowed_error = 0.25
    err = None 
    epoch = 0

    while err is None or err > allowed_error and epoch < epoch_numb:
        epoch += 1
        z = x_train.T @ weights
        delta_root = d_train - z
        err = np.mean(np.square(delta_root))
        weights = weights + (alfa * x_train @ delta_root)

    z = count_cost(weights, x_test.T)
    matching_percent = np.mean(d_test == apply_func(z, sign_bipolar)).round() * 100

    # plot all data and show
    plt.title(f'AND - epochs: {epoch} - alfa: {alfa}\nallowed error {allowed_error} - match: {matching_percent}%')
    plt.scatter(x_test[1, :], x_test[2, :], c=d_test)
    current_plt.plot_line(0.0, 1.0, lambda x_vals: (-weights[1] * x_vals - weights[0]) / weights[2])
    plt.show()


if __name__ == '__main__':
    main()
