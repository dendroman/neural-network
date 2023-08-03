import numpy as np

# passed in gradient from the next layer
# for the purpose of this example we're going to use
# an array of an incremental gradient values
dvalues = np.array([[1., 1., 1.],
                    [2., 2., 2.],
                    [3., 3., 3.]])


# one bias for each neuron
# biases are the row vector with a shape (1, neurons)
bias = np.array([[2, 3, 0.5]])

# dbiases - sum values, do this over samples (first axis), keepdims
# since this by default will produce a plain list
dbiases = np.sum(dvalues, axis=0, keepdims=True)

print("dbiases:", dbiases)

