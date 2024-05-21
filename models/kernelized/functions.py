import numpy as np

# define some common kernel functions


def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p


def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-np.linalg.norm(x-y)**2 / (2 * (sigma ** 2)))


def rbf_kernel(x, y, gamma=0.1):
    return np.exp(-gamma * np.linalg.norm(x - y) ** 2)


def sigmoid_kernel(x, y, gamma=0.1, r=0, d=3):
    return np.tanh(gamma * np.dot(x, y) + r)


def fisher_kernel(x, y, gamma=0.1, r=0, d=3):
    return np.tanh(gamma * np.dot(x, y) + r) + np.tanh(gamma * np.dot(x, y) + r)**2


def laplacian_kernel(x, y, gamma=0.1):
    return np.exp(-gamma * np.linalg.norm(x - y, 1))


def nngp_kernel(x, y, gamma=0.1):
    return np.exp(-gamma * np.linalg.norm(x - y, 2))
