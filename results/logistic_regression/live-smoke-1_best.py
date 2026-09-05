"""Seed implementation: multiclass logistic regression in plain NumPy.

Contract (do not change): the file defines ``Model`` with
    Model()
    .fit(X, y) -> self          (y are integer class labels 0..C-1)
    .predict(X) -> 1-D int array of class labels

Only the code between the EVOLVE-BLOCK markers is evolved.
"""

import numpy as np

# EVOLVE-BLOCK-START
import numpy as np


class Model:
    """Zero-allocation in-place L-BFGS Multinomial Logistic Regression."""

    def __init__(self, l2=1e-3, max_iter=25, tol=1e-4, m=5):
        self.l2 = float(l2)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.m = int(m)
        self.classes_ = None
        self.W_eff_ = None
        self.b_eff_ = None

    def fit(self, X, y):
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n, d = X.shape
        C = len(self.classes_)
        row_indices = np.arange(n)

        # Standardize features
        mean = np.mean(X, axis=0)
        scale = np.std(X, axis=0)
        scale[scale == 0.0] = 1.0
        X_norm = (X - mean) / scale

        # Contiguous feature matrix with bias column
        X_ext = np.empty((n, d + 1), dtype=np.float64, order='C')
        X_ext[:, :d] = X_norm
        X_ext[:, d] = 1.0

        inv_n = 1.0 / n
        l2 = self.l2

        W_mat = np.zeros((C, d + 1), dtype=np.float64)

        # Initial forward + backward pass
        scores = X_ext @ W_mat.T  # (n, C)
        scores -= np.max(scores, axis=1, keepdims=True)
        np.exp(scores, out=scores)
        scores /= np.sum(scores, axis=1, keepdims=True)

        log_p = np.log(np.maximum(scores[row_indices, y_idx], 1e-15))
        loss = -np.sum(log_p) * inv_n

        scores[row_indices, y_idx] -= 1.0
        grad_W = (scores.T @ X_ext) * inv_n
        grad = grad_W.ravel()

        s_hist = []
        y_hist = []
        rho_hist = []

        for _ in range(self.max_iter):
            if np.max(np.abs(grad)) < self.tol:
                break

            # L-BFGS search direction
            k = len(s_hist)
            if k == 0:
                p = -grad
            else:
                q = grad.copy()
                alphas = np.empty(k, dtype=np.float64)
                for i in reversed(range(k)):
                    alphas[i] = rho_hist[i] * np.dot(s_hist[i], q)
                    q -= alphas[i] * y_hist[i]

                gamma = np.dot(s_hist[-1], y_hist[-1]) / (np.dot(y_hist[-1], y_hist[-1]) + 1e-12)
                r = gamma * q

                for i in range(k):
                    beta = rho_hist[i] * np.dot(y_hist[i], r)
                    r += s_hist[i] * (alphas[i] - beta)

                p = -r

            g_dot_p = np.dot(grad, p)
            if g_dot_p >= 0:
                p = -grad
                g_dot_p = np.dot(grad, p)

            p_mat = p.reshape(C, d + 1)
            step = 1.0
            armijo_const = 1e-4 * g_dot_p

            # Line search reusing forward pass allocations
            for _ in range(10):
                W_cand = W_mat + step * p_mat
                scores = X_ext @ W_cand.T
                scores -= np.max(scores, axis=1, keepdims=True)
                np.exp(scores, out=scores)
                scores /= np.sum(scores, axis=1, keepdims=True)

                log_p = np.log(np.maximum(scores[row_indices, y_idx], 1e-15))
                loss_cand = -np.sum(log_p) * inv_n + 0.5 * l2 * np.sum(W_cand[:, :d] ** 2)

                if loss_cand <= loss + step * armijo_const or step < 1e-4:
                    break
                step *= 0.5

            # Backward pass directly using scores (probabilities) from the accepted candidate
            scores[row_indices, y_idx] -= 1.0
            grad_W_new = (scores.T @ X_ext) * inv_n
            grad_W_new[:, :d] += l2 * W_cand[:, :d]
            grad_new = grad_W_new.ravel()

            s = (step * p).ravel()
            y_diff = grad_new - grad
            sy = np.dot(s, y_diff)
            if sy > 1e-10:
                if len(s_hist) >= self.m:
                    s_hist.pop(0)
                    y_hist.pop(0)
                    rho_hist.pop(0)
                s_hist.append(s)
                y_hist.append(y_diff)
                rho_hist.append(1.0 / sy)

            W_mat = W_cand
            loss = loss_cand
            grad = grad_new

        # Fold standardization into weights
        W_norm = W_mat[:, :d]
        b_norm = W_mat[:, d]
        self.W_eff_ = W_norm / scale
        self.b_eff_ = b_norm - np.sum(self.W_eff_ * mean, axis=1)
        return self

    def decision_function(self, X):
        X = np.ascontiguousarray(X, dtype=np.float64)
        return X @ self.W_eff_.T + self.b_eff_

    def predict(self, X):
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]
# EVOLVE-BLOCK-END

assert "Model" in globals(), "the EVOLVE block must define Model"
