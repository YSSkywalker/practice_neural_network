import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def identity_function(x):
    return x

# input layer => 1st hidden layer
X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

# A = X・W + B
A1 = np.dot(X, W1) + B1
Z1 = sigmoid(A1)
print(Z1)

# 1st hidden layer => 2nd hidden layer
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

# A = Z・W + B
A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)
print(Z2)

# 2nd hidden layer => output layer
W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

# A = Z・W + B
A3 = np.dot(Z2, W3) + B3
Z3 = identity_function(A3)
print(Z3)