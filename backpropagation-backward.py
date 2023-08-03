import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# define a layer class
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # initialize weights and biases
        # n inputs, n outputs. multiply by 0.1 to keep values small
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))  # 1 row, n_neurons columns

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs

    def backward(self, dvalues):
        # gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        # since we need to modify the original variable, we make a copy of the values first
        self.dinputs = dvalues.copy()

        # zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0


class Activation_Softmax:
    def forward(self, inputs):
        # get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class Loss:
    def calculate(self, output, y):
        # calculate sample losses
        sample_losses = self.forward(output, y)
        # calculate mean loss
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        # number of samples in a batch
        samples = len(y_pred)
        # clip data to prevent division by 0
        # clip both sides to not drag mean towards any value (bias)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        # probabilities for target values -
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        # mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        # losses
        negative_log_likelihoods = -np.log(correct_confidences)

        return negative_log_likelihoods

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

loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y)

print("Loss: ", loss)