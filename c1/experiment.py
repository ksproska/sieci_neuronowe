from usefull import *
import numpy as np
import matplotlib.pyplot as plt

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
    for experiment_case in [(d_and, "and"), (d_or, "or"), (d_xor, "xor")]:
        for estimate_func in [(unipolar, "unipolar"), (bipolar, "bipolar")]:
            x = randomize(merge_multiple_times(x_original[estimate_func[0]], repetitions))
            d = merge_multiple_times(experiment_case[0][estimate_func[0]], repetitions)
            train_size = int(x.shape[1] - x.shape[1] * test_percent)
            x_train, x_test = x[:, :train_size], x[:, train_size:]
            plt.plot(x_test[1, :], x_test[2, :], marker="o", linestyle='None')

            weights = get_random_weights(3)
            alfa = 0.1
            y = None
            for i in range(100):
                c = cost(weights, x_train.T)
                y = estimate(c, lambda v: estimate_func[0](0, v))
                dw = delta_w(d[:train_size], y, x_train)
                weights = calculate_new_weight(weights, alfa, dw)

            c = cost(weights, x_test.T)
            y = estimate(c, lambda v: estimate_func[0](0, v))
            diff = np.mean(d[train_size:] == y)
            plt.title(f'{experiment_case[1]} - {estimate_func[1]} - {diff}')
            print(f'{experiment_case[1]} - {estimate_func[1]} - {diff}')

            x = np.linspace(-1, 1.5, 10)
            y = (-weights[1] * x - weights[0]) / weights[2]
            plt.plot(x, y)
            plt.show()
