import numpy as np

# passed in gradient from the next layer
# for the purpose of this example we're going to use
# an array of an incremental gradient values
dvalues = np.array([[1., 1., 1.],
                    [2., 2., 2.],
                    [3., 3., 3.]])


# we have 3 sets of weights - one set for each neuron
# we have 4 inputs, thus 4 weights
# recall that we keep weights transposed
weights = np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T


# we calculate the gradient for each input
# we transposed weights, so we need to transpose it back
# because weight matrix saved as rows not columns
dinputs = np.dot(dvalues, weights.T)

# let's print gradients to see if they make sense
print('Gradients: ', dinputs)
