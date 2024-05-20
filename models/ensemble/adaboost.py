import numpy as np

from models.cart import DecisionTree


class AdaBoost:
    def __init__(self, n_estimators=50, max_depth=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.models = []
        self.model_weights = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        weights = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            model = DecisionTree(max_depth=self.max_depth)
            model.fit(X, y, sample_weight=weights)
            predictions = model.predict(X)

            misclassified = (predictions != y).astype(int)
            error = np.dot(weights, misclassified) / np.sum(weights)

            if error == 0 or error >= 0.5:
                break

            alpha = 0.5 * np.log((1 - error) / error)
            weights *= np.exp(alpha * misclassified)
            weights /= np.sum(weights)

            self.models.append(model)
            self.model_weights.append(alpha)

    def predict(self, X):
        model_preds = np.array([model.predict(X) for model in self.models])
        weighted_preds = np.dot(self.model_weights, model_preds)
        return np.sign(weighted_preds)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y == y_pred)
