import unittest
import numpy as np

from models.unsupervised.gaussian_mixture_model import GMM


class TestGMM(unittest.TestCase):

    def setUp(self):
        # Set up a small dataset for testing
        np.random.seed(42)
        self.X = np.vstack([
            np.random.multivariate_normal([0, 0], np.eye(2), 50),
            np.random.multivariate_normal([5, 5], np.eye(2), 50)
        ])
        self.gmm = GMM(clusters=2)

    def test_initialization(self):
        # Test initialization of GMM parameters
        self.assertEqual(self.gmm.clusters, 2)
        self.assertIsNone(self.gmm.weights)
        self.assertIsNone(self.gmm.means)
        self.assertIsNone(self.gmm.covs)

    def test_fit(self):
        # Test fitting the GMM model to the data
        self.gmm.fit(self.X)
        self.assertEqual(len(self.gmm.weights), 2)
        self.assertEqual(len(self.gmm.means), 2)
        self.assertEqual(len(self.gmm.covs), 2)
        for cov in self.gmm.covs:
            self.assertEqual(cov.shape, (2, 2))

    def test_predict(self):
        # Test predicting the cluster assignments
        self.gmm.fit(self.X)
        predictions = self.gmm.predict(self.X)
        self.assertEqual(predictions.shape[0], self.X.shape[0])
        self.assertTrue(np.all(np.unique(predictions) < self.gmm.clusters))

    def test_fit_predict(self):
        # Test fit and predict together
        self.gmm.fit(self.X)
        predictions = self.gmm.predict(self.X)
        self.assertEqual(predictions.shape, (self.X.shape[0],))
        # Check if there are at least some points assigned to each cluster
        unique, counts = np.unique(predictions, return_counts=True)
        self.assertEqual(len(unique), self.gmm.clusters)

    def test_gaussian(self):
        # Test the _gaussian function
        mean = np.array([0, 0])
        cov = np.eye(2)
        values = self.gmm._gaussian(self.X, mean, cov)
        self.assertEqual(values.shape, (self.X.shape[0],))
        self.assertTrue(np.all(values >= 0))

    def test_weights_sum_to_one(self):
        # Test that the weights sum to one after fitting
        self.gmm.fit(self.X)
        self.assertAlmostEqual(np.sum(self.gmm.weights), 1.0)


if __name__ == '__main__':
    unittest.main()
