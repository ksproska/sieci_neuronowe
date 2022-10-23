#%%
import pandas as pd
from data import *
plt.style.use('ggplot')

repetitions = 200
x_all = reproduce_x_times(x_unipolar_no_teta, repetitions)
d_all = reproduce_x_times(d_unipolar, repetitions)
x_all = x_all + get_random(x_all.shape)

test_size = int(x_all.shape[1] * 0.25)
train_size = int(x_all.shape[1] - test_size)
x_train, x_test = x_all[:, :train_size], x_all[:, train_size:]
d_train, d_test = d_all[:, :train_size], d_all[:, train_size:]

x_range = np.arange(-2, 4)
experiments_numb = 10

perceptrons = []
tetas = list(np.linspace(0, 1, num=1000))[1:]

estimate_func = lambda v: unipolar(2, v)
p = Perceptron(x_train, d_train, x_test, d_test, estimate_func, 1.0, (-0.1, 0.1))
perceptrons.append(p)
p.count()
p.display()
