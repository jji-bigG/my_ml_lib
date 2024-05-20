import unittest
import numpy as np

from models.ensemble.random_forest import RandomForest


class TestRandomForest(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1, 2], [1, -1], [2, 2], [2, -1], [3, 2], [3, -1]])
        self.y = np.array([0, 1, 0, 1, 0, 1])
        self.weights = np.array([1, 1, 1, 1, 1, 1])

    def test_basic_functionality(self):
        rf = RandomForest(n_estimators=10, max_samples=1.0,
                          max_features=0.8, max_depth=3)
        rf.fit(self.X, self.y)
        predictions = rf.predict(self.X)
        self.assertTrue(np.array_equal(predictions.round(), self.y))

    def test_weighted_samples(self):
        rf = RandomForest(n_estimators=10, max_samples=1.0,
                          max_features=0.8, max_depth=3)
        weights = np.array([10, 1, 10, 1, 10, 1])  # High weight on class 0
        rf.fit(self.X, self.y, sample_weight=weights)
        predictions = rf.predict(self.X)
        self.assertTrue(np.array_equal(predictions.round(), self.y))

    def test_max_depth(self):
        rf = RandomForest(n_estimators=10, max_samples=1.0,
                          max_features=0.8, max_depth=1)
        rf.fit(self.X, self.y)
        predictions = rf.predict(self.X)
        # Predictions should be closer to the majority class since trees are shallow
        majority_class = np.bincount(self.y).argmax()
        self.assertTrue(np.all(predictions.round() == majority_class))

    def test_no_variability(self):
        y_same = np.array([1, 1, 1, 1, 1, 1])
        rf = RandomForest(n_estimators=10, max_samples=1.0,
                          max_features=0.8, max_depth=3)
        rf.fit(self.X, y_same)
        predictions = rf.predict(self.X)
        self.assertTrue(np.all(predictions.round() == 1))

    def test_all_same_inputs(self):
        X_same = np.array([[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]])
        rf = RandomForest(n_estimators=10, max_samples=1.0,
                          max_features=0.8, max_depth=3)
        rf.fit(X_same, self.y)
        predictions = rf.predict(X_same)
        self.assertTrue(np.all(predictions.round() == np.mean(self.y).round()))

    def test_evaluate_accuracy(self):
        rf = RandomForest(n_estimators=10, max_samples=1.0,
                          max_features=0.8, max_depth=3)
        rf.fit(self.X, self.y)
        accuracy = rf.evaluate(self.X, self.y)
        self.assertEqual(accuracy, 1.0)

    def test_with_extreme_weights(self):
        rf = RandomForest(n_estimators=10, max_samples=1.0,
                          max_features=0.8, max_depth=3)
        weights_extreme = np.array(
            [0.1, 1000, 0.1, 1000, 0.1, 1000])  # Avoid zero weights
        rf.fit(self.X, self.y, sample_weight=weights_extreme)
        predictions = rf.predict(self.X)
        self.assertEqual(predictions[1].round(), 1)
        self.assertEqual(predictions[3].round(), 1)
        self.assertEqual(predictions[5].round(), 1)


if __name__ == '__main__':
    unittest.main()
