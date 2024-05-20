import unittest
import numpy as np
from models.linear_regression import LinearRegression
from models.ensemble.bagging import Bagging
from data.dataset import load_dataset
from optimizers.loss import mse


class TestBagging(unittest.TestCase):
    def setUp(self):
        # Load dataset
        self.X, self.y = load_dataset()

    def test_bagging_fit_predict(self):
        # Initialize the bagging model with LinearRegression
        bagging_model = Bagging(
            model=LinearRegression, n_estimators=5, max_samples=0.8, max_features=0.8)
        bagging_model.fit(self.X, self.y)

        # Make predictions
        predictions = bagging_model.predict(self.X)

        # Check that the predictions have the same shape as the labels
        self.assertEqual(predictions.shape, self.y.shape)

    def test_bagging_evaluate(self):
        # Initialize the bagging model with LinearRegression
        bagging_model = Bagging(
            model=LinearRegression, n_estimators=5, max_samples=0.8, max_features=0.8)
        bagging_model.fit(self.X, self.y)

        # Evaluate the model
        score = bagging_model.evaluate(self.X, self.y, metric=mse)

        # Check that the score is a float and within a reasonable range
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0)
        self.assertLess(score, 1)


if __name__ == '__main__':
    unittest.main()
