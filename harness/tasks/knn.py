"""Seed implementation: k-nearest-neighbours classifier in plain NumPy.

Contract (do not change): the file defines ``Model`` with
    Model()
    .fit(X, y) -> self
    .predict(X) -> 1-D int array of class labels

Only the code between the EVOLVE-BLOCK markers is evolved.
"""

import numpy as np

# EVOLVE-BLOCK-START


class Model:
    """Brute force with a Python loop per query. Correct, but slow."""

    def __init__(self, n_neighbors=5):
        self.k = int(n_neighbors)
        self.X_ = None
        self.y_ = None
        self.classes_ = None

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=np.float64)
        self.y_ = np.asarray(y)
        self.classes_ = np.unique(self.y_)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        out = np.empty(X.shape[0], dtype=self.y_.dtype)
        for i in range(X.shape[0]):
            diff = self.X_ - X[i]
            dist = np.sqrt(np.sum(diff * diff, axis=1))
            order = np.argsort(dist)
            nearest = self.y_[order[: self.k]]
            # Ties go to the smallest label.
            values, counts = np.unique(nearest, return_counts=True)
            out[i] = values[np.argmax(counts)]
        return out


# EVOLVE-BLOCK-END

assert "Model" in globals(), "the EVOLVE block must define Model"
