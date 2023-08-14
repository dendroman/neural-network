import numpy as np
import nnfs
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data
from timeit import timeit
from nnfs.datasets import sine_data

nnfs.init()

# define a layer class


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1=0, weight_regularizer_l2=0, bias_regularizer_l1=0, bias_regularizer_l2=0):
        # initialize weights and biases
        # n inputs, n outputs. multiply by 0.1 to keep values small
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))  # 1 row, n_neurons columns

        # set regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs

    def backward(self, dvalues):
        # gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # gradients on regularization
        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1

        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights

        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1

        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        # gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)


class Layer_Dropout:
    # init
    def __init__(self, rate):
        # store rate, we invert it as for example for dropout of 0.1 we need success rate of 0.9
        self.rate = 1 - rate

    # forward pass
    def forward(self, inputs):
        # save input values
        self.inputs = inputs
        # generate and save scaled mask
        self.binary_mask = np.random.binomial(
            1, self.rate, size=inputs.shape) / self.rate
        # apply mask to output values
        self.output = inputs * self.binary_mask

    # backward pass
    def backward(self, dvalues):
        # gradient on values
        self.dinputs = dvalues * self.binary_mask


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
        # formula: exp(x_i) / sum(exp(x))
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # normalize them for each sample
        # this is done by dividing each value by the sum of all values
        # the sum of all values now equals 1
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        # create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # flatten output array
            single_output = single_output.reshape(-1, 1)

            # calculate jacobian matrix of the output
            # jacobian matrix is the matrix of all first-order partial derivatives of a vector-valued function
            jacobian_matrix = np.diagflat(
                single_output) - np.dot(single_output, single_output.T)

            # calculate sample-wise gradient and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


class Activation_Sigmoid:

    # forward pass
    def forward(self, inputs):
        # save input and calculate/save output of the sigmoid function
        # sigmoid(x) = 1 / (1 + exp(-x))
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    # backward pass
    def backward(self, dvalues):
        # derivative - calculates from output of the sigmoid function
        self.dinputs = dvalues * (1 - self.output) * self.output


class Activation_Linear:

    # forward pass
    def forward(self, inputs):
        # just remember values
        self.inputs = inputs
        self.output = inputs

    # backward pass
    def backward(self, dvalues):
        # derivative is 1, 1 * dvalues = dvalues - the chain rule
        self.dinputs = dvalues.copy()


class Loss:
    def calculate(self, output, y):
        # calculate sample losses
        sample_losses = self.forward(output, y)
        # calculate mean loss
        data_loss = np.mean(sample_losses)
        return data_loss

    def regularization_loss(self, layer):
        # 0 by default
        regularization_loss = 0

        # calculate regularization loss
        # L1 regularization - weights
        # calculate only when factor greater than 0
        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * \
                np.sum(np.abs(layer.weights))

        # L2 regularization - weights
        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * \
                np.sum(layer.weights * layer.weights)

        # L1 regularization - biases
        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * \
                np.sum(np.abs(layer.biases))

        # L2 regularization - biases
        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * \
                np.sum(layer.biases * layer.biases)

        return regularization_loss


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

    def backward(self, dvalues, y_true):
        # number of samples
        samples = len(dvalues)
        # number of labels in every sample
        # we'll use the first sample to count them
        labels = len(dvalues[0])

        # if labels are sparse, turn them into one-hot vector
        # sparse means that the labels are integers
        # and non-sparse are one-hot vectors
        if len(y_true.shape) == 1:
            # np.eye returns a 2D array with 1.0s at the diagonal and 0.0s elsewhere
            # diagonal being index of the label
            y_true = np.eye(labels)[y_true]

        # calculate gradient
        # formula: -y_true / y_pred
        self.dinputs = -y_true / dvalues
        # normalize gradient
        # this means that all the values of the gradient will be divided by the number of samples
        # so the larger the batch, the smaller the gradient
        self.dinputs = self.dinputs / samples


class Loss_BinaryCrossentropy(Loss):
    # forward pass
    def forward(self, y_pred, y_true):
        # clip data to prevent division by 0
        # clip both sides to not drag mean towards any value (bias)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        # calculate sample-wise loss
        sample_losses = -(y_true*np.log(y_pred_clipped) +
                          (1-y_true)*np.log(1-y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)

        return sample_losses

    # backward pass
    def backward(self, dvalues, y_true):

        # number of samples
        samples = len(dvalues)
        # number of outputs in every sample
        # we'll use the first sample to count them
        outputs = len(dvalues[0])

        # clip data to prevent division by 0
        # clip both sides to not drag mean towards any value (bias)
        clipped_dvalues = np.clip(dvalues, 1e-7, 1-1e-7)

        # calculate gradient
        self.dinputs = -(y_true/clipped_dvalues -
                         (1-y_true)/(1-clipped_dvalues)) / outputs

        # normalize gradient
        self.dinputs = self.dinputs / samples


class Loss_MeanSquaredError(Loss):  # L2 loss
    # forward pass
    def forward(self, y_pred, y_true):
        # calculate loss
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
        # return losses
        return sample_losses

    # backward pass
    def backward(self, dvalues, y_true):
        # number of samples
        samples = len(dvalues)
        # number of outputs in every sample
        # we'll use the first sample to count them
        outputs = len(dvalues[0])

        # gradient on values
        self.dinputs = -2 * (y_true - dvalues) / outputs
        # normalize gradient
        self.dinputs = self.dinputs / samples


class Loss_MeanAbsoluteError(Loss):  # L1 loss

    # forward pass
    def forward(self, y_pred, y_true):

        # calculate loss
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)

        # return losses
        return sample_losses

    # backward pass
    def backward(self, dvalues, y_true):

        # number of samples
        samples = len(dvalues)
        # number of outputs in every sample
        # we'll use the first sample to count them
        outputs = len(dvalues[0])

        # calculate gradient
        self.dinputs = np.sign(y_true - dvalues) / outputs
        # normalize gradient
        self.dinputs = self.dinputs / samples

# softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step


class Activation_Softmax_Loss_CategoricalCrossentropy():
    # creates activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    # forward pass
    def forward(self, inputs, y_true):
        # output layer's activation function
        self.activation.forward(inputs)
        # set the output
        self.output = self.activation.output
        # calculate and return loss value
        return self.loss.calculate(self.output, y_true)

    # backward pass
    def backward(self, dvalues, y_true):
        # number of samples
        samples = len(dvalues)

        # if labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # copy so we can safely modify
        self.dinputs = dvalues.copy()

        # calculate gradient
        self.dinputs[range(samples), y_true] -= 1

        # normalize gradient
        self.dinputs = self.dinputs / samples


class Optimizer_SGD:
    # initialize optimizer - set settings,
    # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=1.0, decay=0.0, momentum=0.0):
        self.learning_rate = learning_rate
        # learning rate with decay
        self.current_learning_rate = learning_rate
        # learning rate decay
        self.decay = decay
        # iteration counter
        self.iterations = 0
        # momentum
        self.momentum = momentum

    # call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # update parameters
    def update_params(self, layer):
        # if we use momentum
        if self.momentum:
            # If layer does not contain momentum arrays, create them
            # filled with zeros
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                # If there is no momentum array for weights
                # The array doesn't exist for biases yet either.
                layer.bias_momentums = np.zeros_like(layer.biases)
            # Build weight updates with momentum - take previous
            # updates multiplied by retain factor and update with
            # current gradients
            weight_updates = \
                self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            # Build bias updates
            bias_updates = \
                self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        else:
            weight_updates = -self.current_learning_rate * \
                layer.dweights
            bias_updates = -self.current_learning_rate * \
                layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates

    # call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1


class Optimizer_Adagrad:
    # initialize optimizer - set settings,
    # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=1.0, decay=0.0, epsilon=1e-7):
        self.learning_rate = learning_rate
        # learning rate with decay
        self.current_learning_rate = learning_rate
        # learning rate decay
        self.decay = decay
        # iteration counter
        self.iterations = 0
        # epsilon
        self.epsilon = epsilon

    # call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # update parameters
    def update_params(self, layer):
        # If layer does not contain momentum arrays, create them
        # filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            # If there is no momentum array for weights
            # The array doesn't exist for biases yet either.
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * \
            layer.dweights / \
            (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
            layer.dbiases / \
            (np.sqrt(layer.bias_cache) + self.epsilon)

    # call once after any parameter updates

    def post_update_params(self):
        self.iterations += 1

# RMSprop optimizer


class Optimizer_RMSprop:

    # Initialize optimizer - set settings
    # learning rate of 0.001 is default for this optimizer
    def __init__(self, learning_rate=0.001, decay=0.0, epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    # Call once before any parameter updates
    def pre_update_params(self):

        # If decay is set, update the learning rate
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If layer does not contain cache arrays, create ones filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        layer.weight_cache = self.rho * layer.weight_cache + \
            (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + \
            (1 - self.rho) * layer.dbiases**2

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * \
            layer.dweights / \
            (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
            layer.dbiases / \
            (np.sqrt(layer.bias_cache) + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

# Adam optimizer


class Optimizer_Adam:

    # Initialize optimizer - set settings
    # learning rate of 0.001 is default for this optimizer
    # beta_1 and beta_2 are the momentum hyperparameters
    # they calibrate the decay rates of past gradients
    # beta_1 = 0.9, beta_2 = 0.999 are default values
    def __init__(self, learning_rate=0.001, decay=0.0, epsilon=1e-7,
                 beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    # Call once before any parameter updates
    def pre_update_params(self):

        # If decay is set, update the learning rate
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If layer does not contain cache arrays, create ones filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update momentum  with current gradients
        layer.weight_momentums = self.beta_1 * \
            layer.weight_momentums + \
            (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * \
            layer.bias_momentums + \
            (1 - self.beta_1) * layer.dbiases

        # Get corrected momentum
        # self.iteration is 0 at first pass
        # and we need to start with 1 here
        weight_momentums_corrected = layer.weight_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))

        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + \
            (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + \
            (1 - self.beta_2) * layer.dbiases**2

        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * \
            weight_momentums_corrected / \
            (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
            bias_momentums_corrected / \
            (np.sqrt(bias_cache_corrected) + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1


# create dataset
X, y = sine_data()

# create Dense layer with 1 input feature and 64 output values
dense1 = Layer_Dense(1, 64)

# create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()

# create second Dense layer with 64 input features (as we take output
# of previous layer here) and 64 output value
dense2 = Layer_Dense(64, 64)
# create ReLU activation (to be used with Dense layer):
activation2 = Activation_ReLU()

# create third Dense layer with 64 input features (as we take output
# of previous layer here) and 1 output value
dense3 = Layer_Dense(64, 1)

# create linear activation:
activation3 = Activation_Linear()

# Create loss function
loss_function = Loss_MeanSquaredError()

# Create optimizer
optimizer = Optimizer_Adam(learning_rate=0.005, decay=1e-3)

# accuracy precision for accuracy calculation
# there are no really accuracy here, this is just to track the progress
# but we can simulate/approximate it. We'll calculate it by checking how many values
# have a diff to ground truth less than given precision
# we'll calculate this precision as a fraction of standard deviation of all the ground
# truth values
accuracy_precision = np.std(y) / 250

# Train in loop
for epoch in range(10001):
    # perform a forward pass of our training data through this layer
    dense1.forward(X)

    # perform a forward pass through activation function
    # takes the output of first dense layer here
    activation1.forward(dense1.output)

    # perform a forward pass through second Dense layer
    # takes outputs of activation function of first layer as inputs
    dense2.forward(activation1.output)

    # perform a forward pass through activation function
    # takes the output of second dense layer here
    activation2.forward(dense2.output)

    # perform a forward pass through third Dense layer
    # takes outputs of activation function of second layer as inputs
    dense3.forward(activation2.output)

    # perform a forward pass through activation function
    # takes the output of third dense layer here
    activation3.forward(dense3.output)

    # calculate the data loss
    data_loss = loss_function.calculate(activation3.output, y)

    # calculate regularization penalty
    regularization_loss = loss_function.regularization_loss(
        dense1) + loss_function.regularization_loss(dense2) \
        + loss_function.regularization_loss(dense3)

    # calculate overall loss
    loss = data_loss + regularization_loss

    # calculate accuracy from output of activation2 and targets
    # to calculate it we're taking absolute difference between
    # predictions and ground truth values and compare if differences
    # are lower than given precision value
    predictions = activation3.output
    accuracy = np.mean(np.absolute(predictions - y) < accuracy_precision)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f} (' +
              f'data_loss: {data_loss:.3f}, ' +
              f'reg_loss: {regularization_loss:.3f}), ' +
              f'lr: {optimizer.current_learning_rate}')

    # Backward pass
    loss_function.backward(activation3.output, y)
    activation3.backward(loss_function.dinputs)
    dense3.backward(activation3.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.post_update_params()


# Validate the model with test data
X_test, y_test = sine_data()

dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
dense3.forward(activation2.output)
activation3.forward(dense3.output)

plt.plot(X_test, y_test)
plt.plot(X_test, activation3.output)
plt.show()
