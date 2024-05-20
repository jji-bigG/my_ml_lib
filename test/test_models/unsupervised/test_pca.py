import unittest
import numpy as np

from models.unsupervised.principal_components import PCA


class TestPCA(unittest.TestCase):

    def setUp(self):
        # Set up a small dataset for testing
        np.random.seed(42)
        self.X = np.array([[2.5, 2.4],
                           [0.5, 0.7],
                           [2.2, 2.9],
                           [1.9, 2.2],
                           [3.1, 3.0],
                           [2.3, 2.7],
                           [2, 1.6],
                           [1, 1.1],
                           [1.5, 1.6],
                           [1.1, 0.9]])
        self.pca = PCA(n_components=2)

    def test_initialization(self):
        # Test initialization of PCA parameters
        self.assertEqual(self.pca.n_components, 2)
        self.assertIsNone(self.pca.components)
        self.assertIsNone(self.pca.mean)

    def test_fit(self):
        # Test fitting the PCA model to the data
        self.pca.fit(self.X)
        self.assertEqual(self.pca.components.shape, (2, 2))
        self.assertEqual(self.pca.mean.shape, (2,))

    def test_transform(self):
        # Test transforming the data
        self.pca.fit(self.X)
        transformed = self.pca.transform(self.X)
        self.assertEqual(transformed.shape, (self.X.shape[0], 2))

    def test_fit_transform(self):
        # Test fit and transform together
        transformed = self.pca.fit_transform(self.X)
        self.assertEqual(transformed.shape, (self.X.shape[0], 2))

    def test_variance_preservation(self):
        # Test if the explained variance is preserved
        self.pca.fit(self.X)
        transformed = self.pca.transform(self.X)
        variance_original = np.var(self.X, axis=0).sum()
        variance_transformed = np.var(transformed, axis=0).sum()
        self.assertAlmostEqual(
            variance_original, variance_transformed, places=5)


if __name__ == '__main__':
    unittest.main()
