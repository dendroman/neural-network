import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

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


class Activation_Softmax:
    def forward(self, inputs):
        # get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

# generate spiral data
X, y = spiral_data(samples=100, classes=3)

# create dense layer with 2 input features (x, y) and 3 output values
dense1 = Layer_Dense(2, 3)
# create ReLU activation (to be used with dense layer):
activation1 = Activation_ReLU()

# create second dense layer with 3 input features from previous layer and 3 output values
dense2 = Layer_Dense(3, 3)
# create softmax activation (to be used with dense layer):
activation2 = Activation_Softmax()

# make a forward pass of our training data through this layer
dense1.forward(X)
# make a forward pass through activation function
activation1.forward(dense1.output)

# make a forward pass through second dense layer
dense2.forward(activation1.output)
# make a forward pass through activation function
activation2.forward(dense2.output)

# let's see output of the first few samples:
print(activation2.output[:5])