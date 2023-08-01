import numpy as np

layer_otputs = [[4.8, 1.21, 2.385],
                [8.9, -1.81, 0.2],
                [1.41, 1.051, 0.026]]

exp_values = np.exp(layer_otputs)
norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

print('normalized exponentiated values:')
print(norm_values)
