import numpy as np


class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


class TreeEditDistance:
    def __init__(self):
        self.dp = None

    def string_to_tree(self, s):
        if not s:
            return None
        root = TreeNode(s[0])
        current = root
        for char in s[1:]:
            current.right = TreeNode(char)
            current = current.right
        return root

    def compute_dp_table(self, t1, t2):
        size1 = self.tree_size(t1)
        size2 = self.tree_size(t2)

        # Initialize the DP table with dimensions (size1+1) x (size2+1)
        self.dp = np.zeros((size1 + 1, size2 + 1), dtype=int)

        # Fill the first row and first column
        for i in range(size1 + 1):
            self.dp[i, 0] = i
        for j in range(size2 + 1):
            self.dp[0, j] = j

        return size1, size2

    def ted(self, t1, t2, i, j):
        if t1 is None and t2 is None:
            return 0
        if t1 is None:
            return self.dp[i, j] if self.dp is not None else j
        if t2 is None:
            return self.dp[i, j] if self.dp is not None else i

        cost = 0 if t1.value == t2.value else 1

        if (i, j) not in self.dp:
            self.dp[i, j] = 0

        self.dp[i, j] = min(self.dp[i-1, j] + 1,   # Deletion
                            self.dp[i, j-1] + 1,   # Insertion
                            self.dp[i-1, j-1] + cost)  # Substitution

        if t1.right is not None and t2.right is not None:
            self.ted(t1.right, t2.right, i + 1, j + 1)
        if t1.right is not None:
            self.ted(t1.right, t2, i + 1, j)
        if t2.right is not None:
            self.ted(t1, t2.right, i, j + 1)

        return self.dp[i, j]

    def tree_size(self, root):
        if root is None:
            return 0
        return 1 + self.tree_size(root.left) + self.tree_size(root.right)

    def compute(self, str1, str2):
        t1 = self.string_to_tree(str1)
        t2 = self.string_to_tree(str2)

        size1, size2 = self.compute_dp_table(t1, t2)

        return self.ted(t1, t2, size1, size2)
