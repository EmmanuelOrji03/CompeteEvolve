"""Seed implementation: k-means clustering in plain NumPy.

Contract (do not change): the file defines ``Model`` with
    Model(n_clusters: int)
    .fit(X) -> self
    .predict(X) -> 1-D int array of cluster labels

Only the code between the EVOLVE-BLOCK markers is evolved. Everything outside
is fixed so the evaluation harness always finds the same interface.
"""

import numpy as np

# EVOLVE-BLOCK-START


def _pairwise_sq_dists(X, centers):
    # Materialises (n, k, d); correct but memory-hungry.
    diff = X[:, None, :] - centers[None, :, :]
    return np.sum(diff * diff, axis=2)


class Model:
    def __init__(self, n_clusters=8, max_iter=100, tol=1e-4):
        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        idx = np.random.choice(n, self.n_clusters, replace=False)
        centers = X[idx].copy()

        for _ in range(self.max_iter):
            d2 = _pairwise_sq_dists(X, centers)
            labels = np.argmin(d2, axis=1)
            new_centers = centers.copy()
            for k in range(self.n_clusters):
                members = X[labels == k]
                if len(members) > 0:
                    new_centers[k] = members.mean(axis=0)
            shift = np.sum((new_centers - centers) ** 2)
            centers = new_centers
            if shift <= self.tol:
                break

        d2 = _pairwise_sq_dists(X, centers)
        self.labels_ = np.argmin(d2, axis=1)
        self.inertia_ = float(np.sum(d2[np.arange(n), self.labels_]))
        self.centers_ = centers
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        d2 = _pairwise_sq_dists(X, self.centers_)
        return np.argmin(d2, axis=1)


# EVOLVE-BLOCK-END

assert "Model" in globals(), "the EVOLVE block must define Model"
