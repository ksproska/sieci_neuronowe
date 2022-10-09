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

d_and = np.array(
    [0, 0, 0, 1]
)
d_or = np.array(
    [0, 1, 1, 1]
)
d_xor = np.array(
    [0, 1, 1, 0]
)

if __name__ == '__main__':
    repetitions = 200
    for d_name in [(d_and, "and"), (d_or, "or"), (d_xor, "xor")]:
        plt.title(d_name[1])
        x = randomize(merge_multiple_times(x_original, repetitions))
        plt.plot(x[1, :], x[2, :], marker="o", linestyle='None')

        d = merge_multiple_times(d_name[0], repetitions)
        weights = get_random_weights(3)
        alfa = 0.1
        teta = 0
        func = lambda x: unipolar(teta, x)
        for i in range(100):
            c = cost(weights, x.T)
            y = estimate(c, func)
            dw = delta_w(d, y, x)
            weights = calculate_new_weight(weights, alfa, dw)
            print(weights)

        x = np.linspace(-0.5, 1.5, 10)
        y = (-weights[1] * x - weights[0]) / weights[2]
        plt.plot(x, y)
        plt.show()
