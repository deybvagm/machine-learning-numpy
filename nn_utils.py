import numpy as np


def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x) * np.sqrt(2/n_x)
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * np.sqrt(2/n_h)
    b2 = np.zeros((1, 1))
    parameters = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }
    return parameters


def sigmoid(Z):
    return 1. / (1+np.exp(-Z))


def relu(Z):
    return np.maximum(0, Z)


def forward_propagation(parameters, X):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)

    W2 = parameters["W2"]
    b2 = parameters["b2"]
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2)
    return A2, cache


def backward_propagation(X, Y, cache):
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2) = cache
    dZ2 = A2 - Y
    dW2 = 1./m * np.dot(dZ2, A2.T)
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {
        'dW1': dW1,
        'db1': db1,
        'dW2': dW2,
        'db2': db2
    }

    return gradients


def compute_cost(A2, Y):
    m = Y.shape[1]
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1-A2), 1-Y)
    cost = -1. / m * np.nansum(logprobs)
    return cost


def update_parameters(parameters, gradients, learning_rate):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    dW1 = gradients['dW1']
    db1 = gradients['db1']
    dW2 = gradients['dW2']
    db2 = gradients['db2']

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters['W1'] = W1
    parameters['b1'] = b1
    parameters['W2'] = W2
    parameters['b2'] = b2

    return parameters


def predict(X, parameters):
    Y_pred, _ = forward_propagation(parameters, X)
    Y_pred = (Y_pred > 0.5)
    return Y_pred


def accuracy(Y, Y_pred):
    return 1 - np.mean(np.abs(Y - Y_pred))