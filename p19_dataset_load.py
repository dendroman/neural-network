import numpy as np
import cv2
import os
import nnfs

nnfs.init()

# Loads a MNIST dataset


def load_mnist_dataset(dataset, path):
    # Scan all the directories and create a list of labels
    labels = os.listdir(os.path.join(path, dataset))
    # Create lists for samples and labels
    X = []
    y = []
    # For each label folder
    for label in labels:
        # And for each image in given folder
        for file in os.listdir(os.path.join(path, dataset, label)):
            # Read the image
            image = cv2.imread(os.path.join(
                path, dataset, label, file
            ), cv2.IMREAD_UNCHANGED)

            # And append it and a label to the lists
            X.append(image)
            y.append(label)

    # Convert the data to proper numpy arrays and return
    return np.array(X), np.array(y).astype('uint8')

def create_data_mnist(path):
    # Load both sets separately
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)

    # And return all the data
    return X, y, X_test, y_test

def generate_dataset():
    # Create dataset
    X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')

    # Scale features
    X = (X.astype(np.float32) - 127.5) / 127.5
    X_test = (X_test.astype(np.float32) - 127.5) / 127.5

    print(X.min(), X.max())

    print(X.shape)

    # Reshape the vectors
    X = X.reshape(X.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # shuffle the training dataset
    keys = np.array(range(X.shape[0]))
    np.random.shuffle(keys)

    X = X[keys]
    y = y[keys]

    return X, y, X_test, y_test

