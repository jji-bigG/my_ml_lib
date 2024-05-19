import unittest
import numpy as np
from models.linear_regression import LinearRegression
from data.dataset import load_dataset
from optimizers.loss import mse


class TestLinearRegression(unittest.TestCase):
    def setUp(self):
        # Load dataset
        self.X, self.y = load_dataset()

    def test_fit_and_predict(self):
        # Initialize and fit the model
        model = LinearRegression(loss=mse, verbose=False)
        model.fit(self.X, self.y, epochs=1000, lr=0.01)

        # Make predictions
        predictions = model.predict(self.X)

        # Check that the predictions have the same shape as the labels
        self.assertEqual(predictions.shape, self.y.shape)

        # Check that the loss is below a certain threshold after training
        final_loss = mse(self.y, predictions)
        self.assertLess(final_loss, 0.1)

    def test_fit_verbose(self):
        # Initialize and fit the model with verbosity
        model = LinearRegression(loss=mse, verbose=True)
        model.fit(self.X, self.y, epochs=10, lr=0.01)

        # This test primarily checks that no exceptions are raised during verbose output


if __name__ == '__main__':
    unittest.main()
