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
        self.weights = None

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            self.weights = sample_weight
        else:
            self.weights = np.ones(X.shape[0]) / X.shape[0]
        self.root = self._grow_tree(X, y, self.weights)

    def predict(self, X):
        return np.array([self._predict(inputs) for inputs in X])

    def _grow_tree(self, X, y, weights, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # stopping criteria
        # if all features are the same, return the majority class
        if np.all(X == X[0]):
            return Node(value=np.argmax(np.bincount(y, weights=weights)))

        if (self.max_depth is not None and depth >= self.max_depth) or n_labels == 1 or n_samples < 2:
            return Node(value=np.argmax(np.bincount(y, weights=weights)))

        # greedily select the best split
        feature, threshold = self._best_split(
            X, y, weights, n_samples, n_features)
        if feature is None:
            return Node(value=np.argmax(np.bincount(y, weights=weights)))

        left_indices, right_indices = self._split(X[:, feature], threshold)

        # grow the children that result from the split
        left = self._grow_tree(
            X[left_indices], y[left_indices], weights[left_indices], depth + 1)
        right = self._grow_tree(
            X[right_indices], y[right_indices], weights[right_indices], depth + 1)
        return Node(feature, threshold, left, right)

    def _best_split(self, X, y, weights, n_samples, n_features):
        best_gini = np.inf
        best_feature = None
        best_threshold = None
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices, right_indices = self._split(
                    X[:, feature], threshold)
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                gini = self._gini_impurity(
                    y, weights, left_indices, right_indices)
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold

    def _split(self, X, threshold):
        left_indices = np.argwhere(X <= threshold).flatten()
        right_indices = np.argwhere(X > threshold).flatten()
        return left_indices, right_indices

    def _gini_impurity(self, y, weights, left_indices, right_indices):
        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0

        weight_left = np.sum(weights[left_indices])
        weight_right = np.sum(weights[right_indices])
        weight_total = weight_left + weight_right

        if weight_total == 0:
            return 0

        def weighted_bincount(y, weights):
            unique, counts = np.unique(y, return_inverse=True)
            bincount = np.bincount(counts, weights=weights)
            return dict(zip(unique, bincount))

        left_bincount = weighted_bincount(
            y[left_indices], weights[left_indices])
        right_bincount = weighted_bincount(
            y[right_indices], weights[right_indices])

        left_gini = 1 - sum((count / weight_left) **
                            2 for count in left_bincount.values())
        right_gini = 1 - sum((count / weight_right) **
                             2 for count in right_bincount.values())

        return (weight_left * left_gini + weight_right * right_gini) / weight_total

    def _predict(self, inputs):
        node = self.root
        while node.left or node.right:
            if inputs[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y == y_pred)
