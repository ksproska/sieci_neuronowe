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
    for experiment_case in [(d_and, "and"), (d_or, "or"), (d_xor, "xor")]:
        for estimate_func in [(unipolar, "unipolar"), (bipolar, "bipolar")]:
            plt.title(f'{experiment_case[1]} - {estimate_func[1]}')
            x = randomize(merge_multiple_times(x_original[estimate_func[0]], repetitions))
            plt.plot(x[1, :], x[2, :], marker="o", linestyle='None')

            d = merge_multiple_times(experiment_case[0][estimate_func[0]], repetitions)
            weights = get_random_weights(3)
            alfa = 0.1
            for i in range(100):
                c = cost(weights, x.T)
                y = estimate(c, lambda v: estimate_func[0](0, v))
                dw = delta_w(d, y, x)
                weights = calculate_new_weight(weights, alfa, dw)

            x = np.linspace(-1, 1.5, 10)
            y = (-weights[1] * x - weights[0]) / weights[2]
            plt.plot(x, y)
            plt.show()
