import numpy as np

# defining common activation functions using np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def tanh(x):
    return np.tanh(x)


def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

# defining common activation functions' derivatives using np


def sigmoid_derivative(x):
    return x * (1 - x)


def relu_derivative(x):
    return 1 * (x > 0)


def tanh_derivative(x):
    return 1 - np.power(x, 2)


def softmax_derivative(x):
    return x * (1 - x)

# defining a dictionary to map activation functions to their derivatives


activation_functions = {
    'sigmoid': (sigmoid, sigmoid_derivative),
    'relu': (relu, relu_derivative),
    'tanh': (tanh, tanh_derivative),
    'softmax': (softmax, softmax_derivative)
}
