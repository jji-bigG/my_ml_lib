import numpy as np

from optimizers.loss import mse


class LinearRegression:
    def __init__(self, loss=mse, verbose=False):
        self.weights = None
        self.loss = loss
        self.verbose = verbose

    def fit(self, X, y, epochs=1000, lr=0.01):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
        self.weights = np.random.randn(X_b.shape[1], 1)

        for epoch in range(epochs):
            gradients = 2 / X_b.shape[0] * X_b.T.dot(X_b.dot(self.weights) - y)
            self.weights -= lr * gradients

            if self.verbose and epoch % 100 == 0:
                loss = self.loss(y, X_b.dot(self.weights))
                print(f"Epoch {epoch}: loss={loss}")

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
        return X_b.dot(self.weights)
