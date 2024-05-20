import unittest
import numpy as np

from models.unsupervised.dbscan import DBSCAN


class TestDBSCAN(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.X = np.vstack([
            np.random.multivariate_normal([0, 0], np.eye(2), 50),
            np.random.multivariate_normal([5, 5], np.eye(2), 50),
            np.random.multivariate_normal([10, 10], np.eye(2), 50)
        ])
        self.dbscan = DBSCAN(eps=1.5, min_samples=5)

    def test_initialization(self):
        self.assertEqual(self.dbscan.eps, 1.5)
        self.assertEqual(self.dbscan.min_samples, 5)
        self.assertIsNone(self.dbscan.labels_)

    def test_fit(self):
        self.dbscan.fit(self.X)
        self.assertEqual(len(self.dbscan.labels_), self.X.shape[0])
        # Some points are clustered
        self.assertTrue(np.any(self.dbscan.labels_ != -1))

    def test_predict(self):
        labels = self.dbscan.fit_predict(self.X)
        self.assertEqual(labels.shape[0], self.X.shape[0])
        self.assertTrue(np.any(labels != -1))  # Some points are clustered

    def test_noisy_points(self):
        self.dbscan = DBSCAN(eps=0.1, min_samples=5)
        self.dbscan.fit(self.X)
        # All points should be noise
        self.assertTrue(np.all(self.dbscan.labels_ == -1))

    def test_clusters_found(self):
        self.dbscan = DBSCAN(eps=1.5, min_samples=5)
        labels = self.dbscan.fit_predict(self.X)
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remove noise label if present
        self.assertEqual(len(unique_labels), 3)  # There should be 3 clusters


if __name__ == '__main__':
    unittest.main()
