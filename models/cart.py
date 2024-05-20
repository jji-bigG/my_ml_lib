# decision trees
import numpy as np


# decision tree node
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


# decision tree
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._predict(inputs) for inputs in X])

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or n_labels == 1 or n_samples < 2:
            return Node(value=np.mean(y))

        # greedily select the best split
        feature, threshold = self._best_split(X, y, n_samples, n_features)
        left_indices, right_indices = self._split(X[:, feature], threshold)

        # grow the children that result from the split
        left = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right = self._grow_tree(X[right_indices], y[right_indices], depth + 1)
        return Node(feature, threshold, left, right)

    def _best_split(self, X, y, n_samples, n_features):
        m = np.mean(y)
        best_gini = np.inf
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices, right_indices = self._split(
                    X[:, feature], threshold)
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                gini = self._gini_impurity(y, left_indices, right_indices, m)
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold

    def _split(self, X, threshold):
        left_indices = np.argwhere(X <= threshold).flatten()
        right_indices = np.argwhere(X > threshold).flatten()
        return left_indices, right_indices

    def _gini_impurity(self, y, left_indices, right_indices, m):
        n = len(y)
        n_left = len(left_indices)
        n_right = len(right_indices)
        gini = 0
        for indices in (left_indices, right_indices):
            if len(indices) == 0:
                continue
            score = 0
            for label in np.unique(y):
                p = np.sum(y[indices] == label) / len(indices)
                score += p * p
            gini += (1 - score) * len(indices) / n
        return gini

    def _predict(self, inputs):
        node = self.root
        while node.left:
            if inputs[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y == y_pred)
