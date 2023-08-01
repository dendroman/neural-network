import numpy as np

np.random.seed(0)

# define batch of 3 of 4 inputs which could be from a sensor or the output from the previous layer
X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

# define a layer class
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # initialize weights and biases
        # n inputs, n outputs. multiply by 0.1 to keep values small
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))  # 1 row, n_neurons columns

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

layer1 = Layer_Dense(4, 5)  # 4 inputs, 5 neurons
layer2 = Layer_Dense(5, 2)  # 5 inputs, 2 neurons

# forward pass and use X as input
layer1.forward(X)
print(layer1.output)

# forward pass and use layer1.output as input
layer2.forward(layer1.output)
print(layer2.output)