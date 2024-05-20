import numpy as np


# apply boosting on the given model


import numpy as np


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
            misclassified = (y_pred != y).astype(int)
            error = np.sum(w * misclassified) / np.sum(w)

            if error == 0 or error >= 0.5:
                break

            alpha = self.learning_rate * 0.5 * np.log((1 - error) / error)
            w *= np.exp(-alpha * y * y_pred)
            w /= np.sum(w)

            self.models.append(model)
            self.weights.append(alpha)

    def predict(self, X):
        if not self.models:
            return np.ones(X.shape[0])

        model_preds = np.array([model.predict(X) for model in self.models])
        weighted_preds = np.dot(self.weights, model_preds)
        return np.sign(weighted_preds)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y == y_pred)
