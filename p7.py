import math

# example values
softmax_output = [0.7, 0.1, 0.2]

# target (ground-truth) class for one-hot encoding
target_output = [1, 0, 0]

# calculate loss per sample
loss = -(math.log(softmax_output[0]) * target_output[0] +
         math.log(softmax_output[0]) * target_output[1] +
         math.log(softmax_output[0]) * target_output[2])

# print it
print(loss)

# possible to simplify because of one-hot encoding
loss = -math.log(softmax_output[0])
print(loss)

# as higher the confidence x, the lower the loss
print(-math.log(0.7))
print(-math.log(0.5))