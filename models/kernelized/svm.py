import numpy as np

# write a kernelized svm class


class KernelSVM:
    def __init__(self, kernel, C=1):
        self.kernel = kernel
        self.C = C

    def fit(self, X, y):
        n, m = X.shape
        self.X = X
        self.y = y
        self.alpha = np.zeros(n)
        self.b = 0
        self.K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                self.K[i, j] = self.kernel(X[i], X[j])
        while True:
            alpha_prev = np.copy(self.alpha)
            for i in range(n):
                s = 0
                for j in range(n):
                    s += self.alpha[j] * y[j] * self.K[i, j]
                s += self.b
                if y[i] * s < 1:
                    self.alpha[i] += 1
            self.b = 0
            for i in range(n):
                self.b += y[i]
                for j in range(n):
                    self.b -= self.alpha[j] * y[j] * self.K[i, j]
            self.b /= n
            if np.linalg.norm(self.alpha - alpha_prev) < 1e-10:
                break

    def predict(self, X):
        n = len(X)
        y = np.zeros(n)
        for i in range(n):
            s = 0
            for j in range(len(self.X)):
                s += self.alpha[j] * self.y[j] * self.kernel(X[i], self.X[j])
            y[i] = s + self.b
        return np.sign(y)
