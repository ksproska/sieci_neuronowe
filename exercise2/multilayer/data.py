"""
Aplikacja powinna być napisana na tyle ogólnie, aby była możliwość:
b) użycia od 2-4 warstw,
c) użycia różnych funkcji aktywacji w warstwach ukrytych (sigmoidalna, tanh, ReLU),
d) użycia warstwy softmax (na Rys. 7. smax) w warstwie wyjściowej,
e) zmiany sposobu inicjalizowania wag (w tym ćwiczeniu przyjmujemy, że wagi będą
inicjalizowane z rozkładu normalnego ze zmiennym odchyleniem standardowym),
f) Zmiany liczby neuronów w warstwach ukrytych,
g) przerwania uczenia i ponownego rozpoczęcia nauki od poprzednich wartości wag.
"""

import math
from enum import Enum
import numpy as np

@np.vectorize
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

@np.vectorize
def tanh(x):
    return 2 / (1 + math.exp(-2*x)) - 1

@np.vectorize
def ReLU(x):
    if x >= 0:
        return x
    return 0

class ActivationFunction(Enum):
    SIGMOID = sigmoid
    TANH = tanh
    ReLU = ReLU

"""
X_train: (60000, 28, 28)
Y_train: (60000,)
X_test:  (10000, 28, 28)
Y_test:  (10000,)
"""
from keras.datasets import mnist
class MLP:
    def __init__(self, neuron_counts=[10, 20], activation_fun: ActivationFunction = ActivationFunction.SIGMOID):
        self.activation_fun = activation_fun
        self.neuron_counts = [28 * 28] + neuron_counts + [28 * 28]
        # (self.train_X, self.train_y), (self.test_X, self.test_y) = mnist.load_data()
        self.all_weights = [get_random(self.get_weights_matrix_shape(i)) for i in range(len(self.neuron_counts) - 1)]
        self.all_bs = [get_random(self.get_b_matrix_shape(i)) for i in range(len(self.neuron_counts) - 1)]
    
    def get_weights_matrix_shape(self, first_inx):
        return [self.neuron_counts[first_inx + 1], self.neuron_counts[first_inx]]
    
    def get_a_matrix_shape(self, inx):
        return [self.neuron_counts[inx], 1]
    
    def get_b_matrix_shape(self, first_inx):
        return [self.neuron_counts[first_inx + 1], 1]

    def count_one_step(self):
        x = None
        a = x
        f = self.activation_fun
        for i in range(len(self.neuron_counts) - 1):
            W = self.all_weights[i]
            b = self.all_bs[i]
            z = W @ a + b
            a = f(z)


def get_random(shape):
    rand_matrix = np.random.rand(*shape)
    return rand_matrix

def main():
    mlp = MLP()
    for i in range(len(mlp.all_weights)):
        print("W", mlp.all_weights[i].shape)
        print("a", mlp.get_a_matrix_shape(i))
        print("b",mlp.all_bs[i].shape)
        print()

if __name__ == '__main__':
    main()
