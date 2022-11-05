import numpy as np


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def negative_log_likelihood(y, d):
    return -np.log(y) * d


def get_d_matrix(y, dim):
    d = np.zeros((dim, y.shape[0]))
    for i in range(d.shape[1]):
        d[int(y[i][0])][i] = 1
    return d


def calculate_loss(Y, D):
    diff = (D - Y) ** 2
    summed = np.sum(diff, axis=0) / 2
    return np.mean(summed)
