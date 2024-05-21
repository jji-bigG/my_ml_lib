import numpy as np

from models.kernelized.functions import rbf_kernel


class KernelSVM:
    def __init__(self, kernel=rbf_kernel, C=1.0, max_iter=1000, tol=1e-3):
        self.kernel = kernel
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = None
        self.b = 0
        self.X = None
        self.y = None

    def _compute_kernel_matrix(self, X):
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self._kernel_function(X[i], X[j])
        return K

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.alpha = np.zeros(n_samples)
        self.b = 0
        self.X = X
        self.y = y
        K = self._compute_kernel_matrix(X)

        for _ in range(self.max_iter):
            alpha_prev = np.copy(self.alpha)

            for j in range(n_samples):
                i = self._select_second_alpha(j, n_samples)
                if i is None:
                    continue

                xi, xj = X[i], X[j]
                yi, yj = y[i], y[j]
                kii, kij, kjj = K[i, i], K[i, j], K[j, j]
                eta = 2.0 * kij - kii - kjj

                if eta >= 0:
                    continue

                self.alpha[j] = self.alpha[j] - \
                    (yj * (self._decision_function(xi) - yi)) / eta
                self.alpha[j] = np.clip(self.alpha[j], 0, self.C)

                if abs(self.alpha[j] - alpha_prev[j]) < self.tol:
                    continue

                self.alpha[i] = self.alpha[i] + yi * \
                    yj * (alpha_prev[j] - self.alpha[j])

                b1 = self.b - self._decision_function(xi) - yi * (
                    self.alpha[i] - alpha_prev[i]) * kii - yj * (self.alpha[j] - alpha_prev[j]) * kij
                b2 = self.b - self._decision_function(xj) - yi * (
                    self.alpha[i] - alpha_prev[i]) * kij - yj * (self.alpha[j] - alpha_prev[j]) * kjj

                if 0 < self.alpha[i] < self.C:
                    self.b = b1
                elif 0 < self.alpha[j] < self.C:
                    self.b = b2
                else:
                    self.b = (b1 + b2) / 2

            diff = np.linalg.norm(self.alpha - alpha_prev)
            if diff < self.tol:
                break

    def predict(self, X):
        y_pred = np.array([self._decision_function(x) for x in X])
        return np.sign(y_pred)

    def _decision_function(self, x):
        result = self.b
        for i in range(len(self.alpha)):
            if self.alpha[i] > 0:
                result += self.alpha[i] * self.y[i] * \
                    self._kernel_function(self.X[i], x)
        return result

    def _select_second_alpha(self, j, n_samples):
        i = np.random.randint(0, n_samples - 1)
        if i >= j:
            i += 1
        return i
