from usefull import randomize, merge_multiple_times, get_random_weights, count_cost, apply_func, sign_bipolar
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

if __name__ == '__main__':
    repetitions = 25
    test_percent = 0.25
    alfa = 0.001
    epoch_numb = 1000

    x_all = randomize(merge_multiple_times(x_original, repetitions))
    d_all = merge_multiple_times(d_and, repetitions).reshape((x_all.shape[1], 1))

    test_size = int(x_all.shape[1] * test_percent)
    train_size = int(x_all.shape[1] - test_size)
    x_train, x_test = x_all[:, :train_size], x_all[:, train_size:]

    weights = get_random_weights(x_all.shape[0]).reshape((x_all.shape[0], 1))
    d_train, d_test = d_all[:train_size], d_all[train_size:]

    for i in range(epoch_numb):
        cost = x_train.T @ weights
        dw = x_train @ (d_train - cost)
        weights = weights + alfa * dw
    
    cost = count_cost(weights, x_test.T)
    diff = np.mean(d_test == apply_func(cost, sign_bipolar)) * 100

    # plot all data and show
    plt.title(f'AND - epochs: {epoch_numb} - alfa: {alfa} - match: {diff}%')
    plt.scatter(x_test[1, :], x_test[2, :])
    x_all = np.linspace(0.0, 1.0, 10)
    plt.plot(x_all, (-weights[1] * x_all - weights[0]) / weights[2])
    plt.axhline(y=0, color="k")
    plt.axvline(x=0, color="k")
    plt.grid(True)
    plt.show()
