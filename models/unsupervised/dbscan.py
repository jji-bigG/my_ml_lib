import numpy as np


class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def fit(self, X):
        self.labels_ = -1 * np.ones(X.shape[0])
        cluster_id = 0

        for i in range(X.shape[0]):
            if self.labels_[i] == -1:
                if self._expand_cluster(X, i, cluster_id):
                    cluster_id += 1

    def _expand_cluster(self, X, point_idx, cluster_id):
        neighbors = self._region_query(X, point_idx)
        if len(neighbors) < self.min_samples:
            self.labels_[point_idx] = -1
            return False
        else:
            self.labels_[point_idx] = cluster_id
            i = 0
            while i < len(neighbors):
                neighbor_idx = neighbors[i]
                if self.labels_[neighbor_idx] == -1:
                    self.labels_[neighbor_idx] = cluster_id
                    new_neighbors = self._region_query(X, neighbor_idx)
                    if len(new_neighbors) >= self.min_samples:
                        neighbors.extend(new_neighbors)
                i += 1
            return True

    def _region_query(self, X, point_idx):
        distances = np.linalg.norm(X - X[point_idx], axis=1)
        return list(np.where(distances <= self.eps)[0])

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
