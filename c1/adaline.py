from usefull import randomize, reproduce_x_times, get_random_weights, count_cost, apply_func, sign_bipolar, \
    get_random_except_first_row, MyCustomPlot
import numpy as np
import matplotlib.pyplot as plt

x_original = np.array(
    [
        [1, 1, 1, 1],
        [0, 0, 1, 1],
        [0, 1, 0, 1]
    ]
)
d_and = np.array([0, 0, 0, 1])
d_or = np.array([0, 1, 1, 1])
d_xor = np.array([0, 1, 1, 0])


def main():
    current_plt = MyCustomPlot()
    current_plt.set_plot(plt)
    repetitions = 25
    test_percent = 0.25
    alfa = 0.001
    epoch_numb = 1000

    x_all = reproduce_x_times(x_original, repetitions)
    x_all = x_all + get_random_except_first_row(x_all.shape)
    d_all = reproduce_x_times(d_and, repetitions).reshape((x_all.shape[1], 1))

    test_size = int(x_all.shape[1] * test_percent)
    train_size = int(x_all.shape[1] - test_size)
    x_train, x_test = x_all[:, :train_size], x_all[:, train_size:]

    weights = get_random_weights(x_all.shape[0]).reshape((x_all.shape[0], 1))
    d_train, d_test = d_all[:train_size], d_all[train_size:]

    for i in range(epoch_numb):
        cost = x_train.T @ weights
        dw = x_train @ (d_train - cost)
        weights = weights + alfa * dw
        # current_plt.plot_line(0.0, 1.0, lambda x_vals: (-weights[1] * x_vals - weights[0]) / weights[2])
    
    cost = count_cost(weights, x_test.T)
    matching_percent = np.mean(d_test == apply_func(cost, sign_bipolar)) * 100

    # plot all data and show
    plt.title(f'AND - epochs: {epoch_numb} - alfa: {alfa} - match: {matching_percent}%')
    current_plt.scatter(x_test[1, :], x_test[2, :])
    current_plt.plot_line(0.0, 1.0, lambda x_vals: (-weights[1] * x_vals - weights[0]) / weights[2])
    plt.show()


if __name__ == '__main__':
    main()
