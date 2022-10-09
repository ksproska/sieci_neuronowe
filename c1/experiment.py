from usefull import *
import numpy as np
import matplotlib.pyplot as plt


x = np.array(
    [
    [1, 1, 1, 1],
    [0, 0, 1, 1], 
    [0, 1, 0, 1]
    ]
)

d = np.array(
    [0, 0, 0, 1]
)

if __name__ == '__main__':
    repetitions = 200
    x = randomize(merge_multiple_times(x, repetitions))
    plt.plot(x[1, :], x[2, :], marker="o", linestyle='None')

    d = merge_multiple_times(d, repetitions)
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

    x = np.linspace(0, 1.5, 10)
    y = (-weights[1] * x - weights[0])/weights[2]
    plt.plot(x, y)
    plt.show()
