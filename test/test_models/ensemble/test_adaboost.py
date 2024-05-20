import unittest
import numpy as np

from models.ensemble.adaboost import AdaBoost


class TestAdaBoost(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1, 2], [1, -1], [2, 2], [2, -1], [3, 2], [3, -1]])
        self.y = np.array([1, -1, 1, -1, 1, -1])

    def test_basic_functionality(self):
        ada = AdaBoost(n_estimators=10, max_depth=1)
        ada.fit(self.X, self.y)
        predictions = ada.predict(self.X)
        self.assertTrue(np.array_equal(predictions, self.y))

    def test_weighted_samples(self):
        ada = AdaBoost(n_estimators=10, max_depth=1)
        ada.fit(self.X, self.y)
        predictions = ada.predict(self.X)
        self.assertTrue(np.array_equal(predictions, self.y))

    def test_no_variability(self):
        y_same = np.array([1, 1, 1, 1, 1, 1])
        ada = AdaBoost(n_estimators=10, max_depth=1)
        ada.fit(self.X, y_same)
        predictions = ada.predict(self.X)
        self.assertTrue(np.all(predictions == 1))

    def test_all_same_inputs(self):
        X_same = np.array([[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]])
        ada = AdaBoost(n_estimators=10, max_depth=1)
        ada.fit(X_same, self.y)
        predictions = ada.predict(X_same)
        self.assertTrue(np.all(predictions == np.sign(np.mean(self.y))))

    def test_evaluate_accuracy(self):
        ada = AdaBoost(n_estimators=10, max_depth=1)
        ada.fit(self.X, self.y)
        accuracy = ada.evaluate(self.X, self.y)
        self.assertEqual(accuracy, 1.0)

    def test_with_extreme_weights(self):
        ada = AdaBoost(n_estimators=10, max_depth=1)
        ada.fit(self.X, self.y)
        predictions = ada.predict(self.X)
        self.assertTrue(np.array_equal(predictions, self.y))


if __name__ == '__main__':
    unittest.main()
