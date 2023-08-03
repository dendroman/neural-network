import numpy as np

# passed in gradient from the next layer
# for the purpose of this example we're going to use
# an array of an incremental gradient values
dvalues = np.array([[1., 1., 1.],
                    [2., 2., 2.],
                    [3., 3., 3.]])


# we have 3 sets of inputs - samples
inputs = np.array([[1, 2, 3, 2.5],
                   [2.0, 5.0, -1.0, 2.0],
                   [-1.5, 2.7, 3.3, -0.8]])

dweights = np.dot(inputs.T, dvalues)

print("dweights:", dweights)
