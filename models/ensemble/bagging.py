import numpy as np


# apply bagging on the given model
class Bagging:
    def __init__(self, model, n_estimators=10, max_samples=1.0, max_features=1.0):
        self.model = model
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.models = []
        self.features = []
        self.sample_indices = []
        self.feature_indices = []

    def fit(self, X, y):
        for i in range(self.n_estimators):
            sample_indices = np.random.choice(X.shape[0], int(
                self.max_samples * X.shape[0]), replace=True)
            feature_indices = np.random.choice(X.shape[1], int(
                self.max_features * X.shape[1]), replace=False)
            self.sample_indices.append(sample_indices)
            self.feature_indices.append(feature_indices)
            X_sample = X[sample_indices][:, feature_indices]
            y_sample = y[sample_indices]
            model = self.model()
            model.fit(X_sample, y_sample)
            self.models.append(model)

    def predict(self, X):
        predictions = []
        for i in range(self.n_estimators):
            X_sample = X[:, self.feature_indices[i]]
            predictions.append(self.models[i].predict(X_sample))
        return np.mean(predictions, axis=0)

    def evaluate(self, X, y, metric):
        y_pred = self.predict(X)
        return metric(y, y_pred)
