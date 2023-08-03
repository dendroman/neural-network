import numpy as np

# example layer output
z = np.array([[1, 2, -3, -4],
                [2, -7, -1, 3],
                [-1, 2, 5, -1]])

dvalues = np.array([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12]])

# ReLU activation's derivative
# with the chained rule applied
drelu = dvalues.copy()
drelu[z <= 0] = 0

print(drelu)
