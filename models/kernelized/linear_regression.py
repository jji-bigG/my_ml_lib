import numpy as np

from models.kernelized.functions import rbf_kernel


class KernelizedOLS:
    def __init__(self, kernel=rbf_kernel, lambda_=0.0):
        self.kernel = kernel
        self.lambda_ = lambda_
        self.alpha = None
        self.X = None

    def fit(self, X, y):
        self.X = X
        K = self._compute_kernel_matrix(X)
        n_samples = X.shape[0]
        self.alpha = np.linalg.inv(K + self.lambda_ * np.eye(n_samples)).dot(y)

    def predict(self, X):
        K = self._compute_kernel_matrix(X, self.X)
        return K.dot(self.alpha)

    def _compute_kernel_matrix(self, X1, X2=None):
        if X2 is None:
            X2 = X1
        n_samples_1, n_samples_2 = X1.shape[0], X2.shape[0]
        K = np.zeros((n_samples_1, n_samples_2))
        for i in range(n_samples_1):
            for j in range(n_samples_2):
                K[i, j] = self.kernel(X1[i], X2[j])
        return K


class KernelizedRidgeRegression:
    def __init__(self, kernel=rbf_kernel, lambda_=1.0):
        self.kernel = kernel
        self.lambda_ = lambda_
        self.alpha = None
        self.X = None

    def fit(self, X, y):
        self.X = X
        K = self._compute_kernel_matrix(X)
        n_samples = X.shape[0]
        self.alpha = np.linalg.inv(K + self.lambda_ * np.eye(n_samples)).dot(y)

    def predict(self, X):
        K = self._compute_kernel_matrix(X, self.X)
        return K.dot(self.alpha)

    def _compute_kernel_matrix(self, X1, X2=None):
        if X2 is None:
            X2 = X1
        n_samples_1, n_samples_2 = X1.shape[0], X2.shape[0]
        K = np.zeros((n_samples_1, n_samples_2))
        for i in range(n_samples_1):
            for j in range(n_samples_2):
                K[i, j] = self.kernel(X1[i], X2[j])
        return K
