import numpy as np

# define 4 inputs which could be from a sensor or the output from the previous layer
inputs = [1, 2, 3, 2.5]

# define 4 weights for the 3 neurons in the layer
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

# define 3 biases for the 3 neurons in the layer
biases = [2, 3, 0.5]

# do a dot product of weights and inputs and add the bias
# (or a matrix multiplication)
output = np.dot(weights, inputs) + biases

# print the output
print(output)
