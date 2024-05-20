import numpy as np


# apply boosting on the given model
class Boosting:
    def __init__(self, model, n_estimators=10, learning_rate=0.1):
        self.model = model
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []
        self.weights = []

    def fit(self, X, y):
        n = X.shape[0]
        w = np.ones(n) / n
        for i in range(self.n_estimators):
            model = self.model()
            model.fit(X, y, sample_weight=w)
            y_pred = model.predict(X)
            error = np.sum(w * (y_pred - y) ** 2)
            if error >= 0.5:
                break
            alpha = np.log((1 - error) / error) / 2
            w *= np.exp(-alpha * y * y_pred)
            w /= np.sum(w)
            self.models.append(model)
            self.weights.append(alpha)

    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])
        return np.sign(np.dot(self.weights, predictions))

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y == y_pred)
