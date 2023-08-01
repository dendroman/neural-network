import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# define batch of 3 of 4 inputs which could be from a sensor or the output from the previous layer
X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

# generate spiral data
X, y = spiral_data(100, 3)

# define a layer class
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # initialize weights and biases
        # n inputs, n outputs. multiply by 0.1 to keep values small
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))  # 1 row, n_neurons columns

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

layer1 = Layer_Dense(4, 5)  # 4 inputs, 5 neurons

# define activation function as a class instance
activation1 = Activation_ReLU()

# forward pass and use X as input
layer1.forward(X)

# forward pass and use layer1.output as input
# changing negative values to 0
activation1.forward(layer1.output)
