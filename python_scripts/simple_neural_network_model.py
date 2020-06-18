import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# input layer => 1st hidden layer
X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

# A = X・W + B
A = np.dot(X, W1) + B1
Z1 = sigmoid(A)
print(Z1)

# 1st hidden layer => 2nd hidden layer
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

# A = Z・W + B
A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(Z1)
print(Z2)