import unittest
import numpy as np

from models.cart import DecisionTree
from models.ensemble.boosting import Boosting


class TestBoosting(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1, 2], [1, -1], [2, 2], [2, -1], [3, 2], [3, -1]])
        self.y = np.array([1, -1, 1, -1, 1, -1])

    def test_basic_functionality(self):
        boosting = Boosting(model=lambda: DecisionTree(
            max_depth=1), n_estimators=10, learning_rate=0.1)
        boosting.fit(self.X, self.y)
        predictions = boosting.predict(self.X)
        self.assertTrue(np.array_equal(predictions, self.y))

    def test_weighted_samples(self):
        boosting = Boosting(model=lambda: DecisionTree(
            max_depth=1), n_estimators=10, learning_rate=0.1)
        boosting.fit(self.X, self.y)
        predictions = boosting.predict(self.X)
        self.assertTrue(np.array_equal(predictions, self.y))

    def test_no_variability(self):
        y_same = np.array([1, 1, 1, 1, 1, 1])
        boosting = Boosting(model=lambda: DecisionTree(
            max_depth=1), n_estimators=10, learning_rate=0.1)
        boosting.fit(self.X, y_same)
        predictions = boosting.predict(self.X)
        self.assertTrue(np.all(predictions == 1))

    def test_all_same_inputs(self):
        X_same = np.array([[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]])
        boosting = Boosting(model=lambda: DecisionTree(
            max_depth=1), n_estimators=10, learning_rate=0.1)
        boosting.fit(X_same, self.y)
        predictions = boosting.predict(X_same)
        self.assertTrue(np.all(predictions == np.sign(np.mean(self.y))))

    def test_evaluate_accuracy(self):
        boosting = Boosting(model=lambda: DecisionTree(
            max_depth=1), n_estimators=10, learning_rate=0.1)
        boosting.fit(self.X, self.y)
        accuracy = boosting.evaluate(self.X, self.y)
        self.assertEqual(accuracy, 1.0)

    def test_with_extreme_weights(self):
        boosting = Boosting(model=lambda: DecisionTree(
            max_depth=1), n_estimators=10, learning_rate=0.1)
        boosting.fit(self.X, self.y)
        predictions = boosting.predict(self.X)
        self.assertTrue(np.array_equal(predictions, self.y))


if __name__ == '__main__':
    unittest.main()
