import unittest
import numpy as np
from models.kernelized.functions import linear_kernel, polynomial_kernel, gaussian_kernel, rbf_kernel
from models.kernelized.svm import KernelSVM


class TestKernelSVM(unittest.TestCase):
    def setUp(self):
        # Simple dataset for testing
        self.X = np.array([[1, 2], [2, 3], [3, 3], [2, 1], [3, 2]])
        self.y = np.array([1, 1, 1, -1, -1])
        self.model = KernelSVM(kernel=rbf_kernel, C=1.0,
                               max_iter=1000, tol=1e-3)

    def test_initialization(self):
        self.assertEqual(self.model.C, 1.0)
        self.assertEqual(self.model.max_iter, 1000)
        self.assertEqual(self.model.tol, 1e-3)
        self.assertEqual(self.model.kernel, rbf_kernel)

    def test_linear_kernel(self):
        result = linear_kernel(np.array([1, 2]), np.array([2, 3]))
        self.assertEqual(result, 8)

    def test_polynomial_kernel(self):
        result = polynomial_kernel(np.array([1, 2]), np.array([2, 3]), p=2)
        self.assertEqual(result, 81)

    def test_gaussian_kernel(self):
        result = gaussian_kernel(np.array([1, 2]), np.array([2, 3]), sigma=1.0)
        self.assertAlmostEqual(result, np.exp(-2.0 / 2), places=5)

    def test_rbf_kernel(self):
        result = rbf_kernel(np.array([1, 2]), np.array([2, 3]), gamma=0.1)
        self.assertAlmostEqual(result, np.exp(-0.1 * 2.0), places=5)

    def test_training(self):
        self.model.fit(self.X, self.y)
        self.assertIsNotNone(self.model.alpha)
        self.assertIsNotNone(self.model.b)

    def test_prediction(self):
        self.model.fit(self.X, self.y)
        predictions = self.model.predict(self.X)
        self.assertEqual(predictions.tolist(), self.y.tolist())


if __name__ == '__main__':
    unittest.main()
