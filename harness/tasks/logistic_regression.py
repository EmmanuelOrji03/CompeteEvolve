"""Seed implementation: multiclass logistic regression in plain NumPy.

Contract (do not change): the file defines ``Model`` with
    Model()
    .fit(X, y) -> self          (y are integer class labels 0..C-1)
    .predict(X) -> 1-D int array of class labels

Only the code between the EVOLVE-BLOCK markers is evolved.
"""

import numpy as np

# EVOLVE-BLOCK-START


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class Model:
    """One-vs-rest, fixed-step full-batch gradient descent. Deliberately simple."""

    def __init__(self, lr=0.1, n_iter=500, l2=1e-4):
        self.lr = float(lr)
        self.n_iter = int(n_iter)
        self.l2 = float(l2)
        self.classes_ = None
        self.W_ = None  # shape (C, d)
        self.b_ = None  # shape (C,)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n, d = X.shape
        C = len(self.classes_)
        W = np.zeros((C, d))
        b = np.zeros(C)
        for c_idx, c in enumerate(self.classes_):
            t = (y == c).astype(np.float64)
            w = np.zeros(d)
            bias = 0.0
            for _ in range(self.n_iter):
                p = _sigmoid(X @ w + bias)
                err = p - t
                grad_w = X.T @ err / n + self.l2 * w
                grad_b = np.mean(err)
                w -= self.lr * grad_w
                bias -= self.lr * grad_b
            W[c_idx] = w
            b[c_idx] = bias
        self.W_ = W
        self.b_ = b
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float64)
        return X @ self.W_.T + self.b_

    def predict(self, X):
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]


# EVOLVE-BLOCK-END

assert "Model" in globals(), "the EVOLVE block must define Model"
