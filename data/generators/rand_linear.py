import numpy as np


def load_dataset():
    # Placeholder for loading dataset
    X = np.random.rand(100, 1)  # Features
    y = 2 * X + 1 + 0.1 * np.random.randn(100, 1)  # Labels with some noise
    return X, y
