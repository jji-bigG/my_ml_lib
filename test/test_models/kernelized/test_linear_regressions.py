import unittest
import numpy as np
from models.kernelized.functions import linear_kernel, polynomial_kernel, gaussian_kernel, rbf_kernel
from models.kernelized.linear_regression import KernelizedOLS, KernelizedRidgeRegression


class TestKernelizedRegression(unittest.TestCase):
    def setUp(self):
        # Simple dataset for testing
        self.X = np.array([[1, 2], [2, 3], [3, 3], [2, 1], [3, 2]])
        self.y = np.array([1, 2, 3, 4, 5])

    def test_kernel_functions(self):
        # Linear Kernel
        result = linear_kernel(np.array([1, 2]), np.array([2, 3]))
        self.assertEqual(result, 8)

        # Polynomial Kernel
        result = polynomial_kernel(np.array([1, 2]), np.array([2, 3]), p=2)
        self.assertEqual(result, 81)

        # Gaussian Kernel
        result = gaussian_kernel(np.array([1, 2]), np.array([2, 3]), sigma=1.0)
        self.assertAlmostEqual(result, np.exp(-2.0 / 2), places=5)

        # RBF Kernel
        result = rbf_kernel(np.array([1, 2]), np.array([2, 3]), gamma=0.1)
        self.assertAlmostEqual(result, np.exp(-0.1 * 2.0), places=5)

    def test_kernelized_ols(self):
        model = KernelizedOLS(kernel=rbf_kernel, lambda_=0.0)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)

        # Simple check: predictions should be close to the actual values
        for pred, actual in zip(predictions, self.y):
            self.assertAlmostEqual(pred, actual, places=1)

    def test_kernelized_ridge_regression(self):
        model = KernelizedRidgeRegression(kernel=rbf_kernel, lambda_=1.0)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)

        # Simple check: predictions should be close to the actual values
        for pred, actual in zip(predictions, self.y):
            self.assertAlmostEqual(pred, actual, places=1)


if __name__ == '__main__':
    unittest.main()
