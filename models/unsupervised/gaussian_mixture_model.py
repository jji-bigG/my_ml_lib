# gaussian mixture model: a close neighbor for k-means
import numpy as np


class GMM:
    def __init__(self, clusters) -> None:
        self.clusters = clusters
        self.weights = None
        self.means = None
        self.covs = None

    def fit(self, X, max_iter=100):
        self.weights = np.ones(self.clusters) / self.clusters
        self.means = X[np.random.choice(
            X.shape[0], self.clusters, replace=False)]
        self.covs = np.array([np.eye(X.shape[1])
                             for _ in range(self.clusters)])
        for _ in range(max_iter):
            # E-step
            likelihoods = np.array([self.weights[i] * self._gaussian(
                X, self.means[i], self.covs[i]) for i in range(self.clusters)]).T
            responsibilities = likelihoods / \
                np.sum(likelihoods, axis=1)[:, None]
            # M-step
            N = np.sum(responsibilities, axis=0)
            self.weights = N / X.shape[0]
            self.means = np.array([np.sum(
                responsibilities[:, i][:, None] * X, axis=0) / N[i] for i in range(self.clusters)])
            self.covs = np.array([np.dot((responsibilities[:, i][:, None] * (
                X - self.means[i])).T, (X - self.means[i])) / N[i] for i in range(self.clusters)])

    def _gaussian(self, X, mean, cov):
        return np.exp(-0.5 * np.sum((X - mean) @ np.linalg.pinv(cov) * (X - mean), axis=1)) / np.sqrt(np.linalg.det(cov) * (2 * np.pi) ** X.shape[1])

    def predict(self, X):
        likelihoods = np.array([self.weights[i] * self._gaussian(X,
                               self.means[i], self.covs[i]) for i in range(self.clusters)]).T
        return np.argmax(likelihoods, axis=1)
