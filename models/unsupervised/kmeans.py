import numpy as np


class KMeans:
    def __init__(self, clusters) -> None:
        self.clusters = clusters
        self.centroids = None

    def fit(self, X, max_iter=100):
        self.centroids = X[np.random.choice(
            X.shape[0], self.clusters, replace=False)]
        for _ in range(max_iter):
            distances = np.linalg.norm(X[:, None] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            new_centroids = np.array(
                [X[labels == i].mean(axis=0) for i in range(self.clusters)])
            if np.all(new_centroids == self.centroids):
                break
            self.centroids = new_centroids

    def predict(self, X):
        distances = np.linalg.norm(X[:, None] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)
