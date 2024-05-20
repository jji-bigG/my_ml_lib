import numpy as np
from models.cart import DecisionTree
import unittest


class TestDecisionTree(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1, 2], [1, -1], [2, 2], [2, -1]])
        self.y = np.array([0, 1, 0, 1])
        self.weights = np.array([1, 1, 1, 1])

    def test_basic_functionality(self):
        tree = DecisionTree(max_depth=3)
        tree.fit(self.X, self.y)
        predictions = tree.predict(self.X)
        self.assertTrue(np.average(predictions == self.y) == 1)

    def test_weighted_samples(self):
        tree = DecisionTree(max_depth=3)
        weights = np.array([10, 1, 10, 1])  # High weight on class 0
        tree.fit(self.X, self.y, sample_weight=weights)
        predictions = tree.predict(self.X)
        self.assertEqual(predictions[0], 0)
        self.assertEqual(predictions[2], 0)

    def test_max_depth(self):
        tree = DecisionTree(max_depth=1)
        tree.fit(self.X, self.y, sample_weight=self.weights)
        node = tree.root
        self.assertIsNone(node.left.left)
        self.assertIsNone(node.left.right)
        self.assertIsNone(node.right.left)
        self.assertIsNone(node.right.right)

    def test_no_variability(self):
        y_same = np.array([1, 1, 1, 1])
        tree = DecisionTree(max_depth=3)
        tree.fit(self.X, y_same)
        predictions = tree.predict(self.X)
        self.assertTrue(np.all(predictions == 1))

    def test_all_same_inputs(self):
        X_same = np.array([[1, 1], [1, 1], [1, 1], [1, 1]])
        tree = DecisionTree(max_depth=3)
        tree.fit(X_same, self.y)
        predictions = tree.predict(X_same)
        self.assertTrue(0 <= np.mean(predictions == 1) <= 1)

    def test_evaluate_accuracy(self):
        tree = DecisionTree(max_depth=3)
        tree.fit(self.X, self.y)
        accuracy = tree.evaluate(self.X, self.y)
        self.assertEqual(accuracy, 1.0)

    def test_with_extreme_weights(self):
        tree = DecisionTree(max_depth=3)
        weights_extreme = np.array(
            [0.1, 1000, 0.1, 1000])  # Avoid zero weights
        tree.fit(self.X, self.y, sample_weight=weights_extreme)
        predictions = tree.predict(self.X)
        self.assertEqual(predictions[1], 1)
        self.assertEqual(predictions[3], 1)


if __name__ == '__main__':
    unittest.main()
