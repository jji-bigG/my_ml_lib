from models.ensemble.bagging import Bagging
from models.cart import DecisionTree
import numpy as np


class RandomForest:
    def __init__(self, n_estimators=10, max_samples=1.0, max_features=1.0, max_depth=None):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.max_depth = max_depth
        self.models = Bagging(
            model=lambda: DecisionTree(max_depth=self.max_depth),
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            max_features=self.max_features
        )

    def fit(self, X, y):
        self.models.fit(X, y)

    def predict(self, X):
        return self.models.predict(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y == y_pred)
