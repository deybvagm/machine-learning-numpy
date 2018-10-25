import numpy as np
import scipy
from scipy import ndimage
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1/(1+np.exp(-z))


def initialize_parameters(dim):
    w = np.zeros((dim, 1))
    b = 0
    return w, b


def forward_propagation(parameters, X):
    w, b = parameters
    z = np.dot(w.T, X) + b
    a = sigmoid(z)
    return a


def compute_cost(A, Y):
    m = Y.shape[1]
    logprobs = np.multiply(np.log(A), Y) + np.multiply(np.log(1-A), 1-Y)
    cost = -1. / m * np.nansum(logprobs)
    return cost


def backward_propagation(X, A, Y):
    m = A.shape[1]
    dw = 1/m * np.dot(X, (A - Y).T)
    db = 1/m * np.sum(A - Y)
    return dw, db


def update_parameters(parameters, grads, learning_rate):
    w, b = parameters
    dw, db = grads
    w = w - learning_rate * dw
    b = b - learning_rate * db
    return w, b


def predict(parameters, X):
    A = forward_propagation(parameters, X)
    Y_prediction = (A > 0.5)
    return Y_prediction


def plot_cost_over_time(costs):
    plt.plot(costs)
    plt.xlabel("iterations")
    plt.ylabel("cost")
    plt.show()


def calculate_accuracy(Y, Y_pred):
    accuracy = 1 - np.mean(np.abs(Y - Y_pred))
    print('Accuracy: {}'.format(accuracy))


def load_image(image_name, num_px=64):
    image_path = "images/" + image_name
    image = np.array(ndimage.imread(image_path, flatten=False))
    image_vector = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
    image_vector = image_vector / 255.
    return image, image_vector

