import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal

from models.unsupervised.kmeans import KMeans


class TestKMeans(unittest.TestCase):

    def setUp(self):
        # Set up a small dataset for testing
        np.random.seed(42)
        self.X = np.vstack([
            np.random.multivariate_normal([0, 0], np.eye(2), 50),
            np.random.multivariate_normal([5, 5], np.eye(2), 50)
        ])
        self.kmeans = KMeans(clusters=2)

    def test_initialization(self):
        # Test initialization of KMeans parameters
        self.assertEqual(self.kmeans.clusters, 2)
        self.assertIsNone(self.kmeans.centroids)

    def test_fit(self):
        # Test fitting the KMeans model to the data
        self.kmeans.fit(self.X)
        self.assertEqual(len(self.kmeans.centroids), 2)
        self.assertEqual(self.kmeans.centroids.shape, (2, 2))

    def test_predict(self):
        # Test predicting the cluster assignments
        self.kmeans.fit(self.X)
        predictions = self.kmeans.predict(self.X)
        self.assertEqual(predictions.shape[0], self.X.shape[0])
        self.assertTrue(np.all(np.unique(predictions) < self.kmeans.clusters))

    def test_fit_predict(self):
        # Test fit and predict together
        self.kmeans.fit(self.X)
        predictions = self.kmeans.predict(self.X)
        self.assertEqual(predictions.shape, (self.X.shape[0],))
        # Check if there are at least some points assigned to each cluster
        unique, counts = np.unique(predictions, return_counts=True)
        self.assertEqual(len(unique), self.kmeans.clusters)

    def test_convergence(self):
        # Test if the algorithm converges
        initial_centroids = self.kmeans.centroids
        self.kmeans.fit(self.X)
        self.assertFalse(np.all(initial_centroids == self.kmeans.centroids))

    def test_predict_new_points(self):
        # Test predicting cluster for new points
        self.kmeans.fit(self.X)
        new_points = np.array([[0, 0], [5, 5], [2.5, 2.5]])
        predictions = self.kmeans.predict(new_points)
        self.assertEqual(predictions.shape[0], new_points.shape[0])
        self.assertTrue(np.all(np.unique(predictions) < self.kmeans.clusters))


if __name__ == '__main__':
    unittest.main()
