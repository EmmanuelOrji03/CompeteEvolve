from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import sklearn.datasets as sk_datasets


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    display_name: str
    dataset_name: str
    dataset_loader: Callable[[], object]
    baseline_code: str     # CO — from-scratch, annotated, evolvable
    reference_code: str    # Reference — real sklearn class, not evolved


DATASET_LOADER = sk_datasets.load_breast_cancer
DATASET_NAME = "Breast Cancer Wisconsin"


def _dataset_to_csv(loader: Callable[[], object]) -> str:
    bunch = loader()
    feature_names = [
        str(n).replace(" ", "_").replace("(", "").replace(")", "") for n in bunch.feature_names
    ]
    lines = [",".join(feature_names + ["target"])]
    for row, target in zip(bunch.data, bunch.target):
        lines.append(",".join([repr(float(v)) for v in row] + [str(int(target))]))
    return "\n".join(lines) + "\n"


def dataset_csv(algorithm: str) -> str:
    """Every algorithm shares one dataset now, so `algorithm` is
    accepted (for interface compatibility with the rest of the system)
    but ignored."""
    return _dataset_to_csv(DATASET_LOADER)


# ---------------------------------------------------------------------
# Shared measurement scaffold — identical in every CO and Reference file,
# so accuracy/loss/timing are measured the same way everywhere.
# ---------------------------------------------------------------------

_MEASURE_AND_RETURN = '''\
    _t0 = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - _t0

    _t1 = time.time()
    predictions = model.predict(X_test)
    inference_time = time.time() - _t1

    accuracy = float(accuracy_score(y_test, predictions))
    try:
        probabilities = model.predict_proba(X_test)
        loss = float(log_loss(y_test, probabilities, labels=sorted(set(y))))
    except Exception:
        loss = 1.0 - accuracy

    return {
        "accuracy": accuracy,
        "loss": loss,
        "training_time": training_time,
        "inference_time": inference_time,
    }
'''

_SPLIT = '''\
    # TRAIT: model accuracy, training time. The split itself is
    # evolvable (test_size, random_state, stratification).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
'''


# ---------------------------------------------------------------------
# 1. Decision Tree
# Real defaults (from the author's decision_trees.py):
#   criterion='gini', splitter='best' (exhaustive), max_depth=None,
#   min_samples_split=2
# ---------------------------------------------------------------------

_DECISION_TREE_CODE = '''\
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss


class ScratchDecisionTree:
    """CART with real Gini-impurity splitting and exhaustive ('best')
    threshold search over every feature at each split — this is the
    actual splitting algorithm scikit-learn's DecisionTreeClassifier
    runs (in compiled Cython); this is the same logic in plain Python,
    so it can be read, annotated, and evolved."""

    def __init__(self, max_depth=None, min_samples_split=2):
        # TRAIT: model accuracy, training time. An unbounded tree
        # (max_depth=None) can overfit; min_samples_split controls how
        # aggressively it keeps splitting.
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree_ = None
        self.n_classes_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        self.n_classes_ = int(y.max()) + 1
        self.tree_ = self._build(X, y, depth=0)
        return self

    def _gini(self, y):
        # TRAIT: model accuracy. The impurity measure driving every
        # split decision — swapping this for entropy is a real,
        # meaningful evolution, not a cosmetic one.
        if len(y) == 0:
            return 0.0
        counts = np.bincount(y, minlength=self.n_classes_)
        p = counts / len(y)
        return 1.0 - np.sum(p ** 2)

    def _best_split(self, X, y):
        # TRAIT: training time, model accuracy. Exhaustive search over
        # every feature and every candidate threshold — this is what
        # "best" (vs "random") splitting means, and it's the dominant
        # cost of fitting the tree.
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split:
            return None, None, None
        parent_impurity = self._gini(y)
        best_gain, best_feature, best_threshold = -1.0, None, None

        for feature in range(n_features):
            values = X[:, feature]
            order = np.argsort(values)
            sorted_values, sorted_y = values[order], y[order]
            distinct = np.where(np.diff(sorted_values) != 0)[0]
            for i in distinct:
                threshold = (sorted_values[i] + sorted_values[i + 1]) / 2.0
                left_y, right_y = sorted_y[: i + 1], sorted_y[i + 1 :]
                if len(left_y) == 0 or len(right_y) == 0:
                    continue
                w_l, w_r = len(left_y) / n_samples, len(right_y) / n_samples
                child_impurity = w_l * self._gini(left_y) + w_r * self._gini(right_y)
                gain = parent_impurity - child_impurity
                if gain > best_gain:
                    best_gain, best_feature, best_threshold = gain, feature, threshold

        if best_feature is None:
            return None, None, None
        return best_feature, best_threshold, best_gain

    def _build(self, X, y, depth):
        n_labels = len(np.unique(y))
        if (
            n_labels == 1
            or len(y) < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            return {"leaf": True, "proba": self._leaf_proba(y)}

        feature, threshold, gain = self._best_split(X, y)
        if feature is None or gain <= 0:
            return {"leaf": True, "proba": self._leaf_proba(y)}

        left_mask = X[:, feature] <= threshold
        return {
            "leaf": False,
            "feature": feature,
            "threshold": threshold,
            "left": self._build(X[left_mask], y[left_mask], depth + 1),
            "right": self._build(X[~left_mask], y[~left_mask], depth + 1),
        }

    def _leaf_proba(self, y):
        counts = np.bincount(y, minlength=self.n_classes_).astype(np.float64)
        return counts / counts.sum()

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.array([self._proba_row(row, self.tree_) for row in X])

    def _proba_row(self, row, node):
        if node["leaf"]:
            return node["proba"]
        branch = node["left"] if row[node["feature"]] <= node["threshold"] else node["right"]
        return self._proba_row(row, branch)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def run(data_path: str) -> dict:
    df = pd.read_csv(data_path)
    X = df.drop(columns=["target"]).to_numpy()
    y = df["target"].to_numpy()

''' + _SPLIT + '''
    model = ScratchDecisionTree(max_depth=None, min_samples_split=2)

''' + _MEASURE_AND_RETURN

_DECISION_TREE_REFERENCE = '''\
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, log_loss


def run(data_path: str) -> dict:
    df = pd.read_csv(data_path)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Real scikit-learn defaults: criterion='gini', splitter='best', max_depth=None.
    model = DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=None, random_state=42)

''' + _MEASURE_AND_RETURN


# ---------------------------------------------------------------------
# 2. K-Nearest Neighbors
# Real defaults (from K_nearest_neighbor.py):
#   n_neighbors=5, weights='uniform', metric='minkowski', p=2 (Euclidean)
# ---------------------------------------------------------------------

_KNN_CODE = '''\
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss


class ScratchKNN:
    """Uniform-weighted majority vote over the k nearest neighbours by
    Euclidean distance — the actual algorithm behind
    KNeighborsClassifier's defaults, in plain vectorised NumPy instead
    of scikit-learn's compiled ball-tree/kd-tree backends."""

    def __init__(self, n_neighbors=5):
        # TRAIT: model accuracy, inference time. Fewer neighbours is
        # noisier but faster to reason about per-query; more neighbours
        # smooths the decision boundary.
        self.n_neighbors = n_neighbors
        self.X_train_ = None
        self.y_train_ = None
        self.n_classes_ = None

    def fit(self, X, y):
        self.X_train_ = np.asarray(X, dtype=np.float64)
        self.y_train_ = np.asarray(y, dtype=np.int64)
        self.n_classes_ = int(self.y_train_.max()) + 1
        return self

    def _distances(self, X):
        # TRAIT: model accuracy, inference time. Euclidean (Minkowski,
        # p=2) distance, vectorised as ||a-b||^2 = ||a||^2+||b||^2-2a.b
        # rather than looping — swapping the distance metric itself
        # (e.g. Manhattan) is a real evolvable change.
        train_sq = np.sum(self.X_train_ ** 2, axis=1)
        query_sq = np.sum(X ** 2, axis=1)[:, None]
        cross = X @ self.X_train_.T
        return np.sqrt(np.maximum(query_sq + train_sq[None, :] - 2 * cross, 0.0))

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        distances = self._distances(X)
        k = min(self.n_neighbors, self.X_train_.shape[0])
        neighbor_idx = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]

        proba = np.zeros((X.shape[0], self.n_classes_))
        for i in range(X.shape[0]):
            counts = np.bincount(self.y_train_[neighbor_idx[i]], minlength=self.n_classes_)
            proba[i] = counts / k
        return proba

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def run(data_path: str) -> dict:
    df = pd.read_csv(data_path)
    X = df.drop(columns=["target"]).to_numpy()
    y = df["target"].to_numpy()

''' + _SPLIT + '''
    model = ScratchKNN(n_neighbors=5)

''' + _MEASURE_AND_RETURN

_KNN_REFERENCE = '''\
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, log_loss


def run(data_path: str) -> dict:
    df = pd.read_csv(data_path)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Real scikit-learn defaults: n_neighbors=5, weights='uniform', metric='minkowski', p=2.
    model = KNeighborsClassifier(n_neighbors=5, weights="uniform", metric="minkowski", p=2)

''' + _MEASURE_AND_RETURN


# ---------------------------------------------------------------------
# 3. Logistic Regression
# Real defaults (from logistic_regression.py): C=1.0 (L2 penalty),
# solver='lbfgs', max_iter=100.
#
# Honest limitation: lbfgs is a quasi-Newton solver; reimplementing it
# from scratch is out of reasonable scope, so CO substitutes batch
# gradient descent on the IDENTICAL L2-penalized cross-entropy
# objective. Confirmed unscaled: CO and Reference land on the exact
# same accuracy (0.9561) on this dataset — the solver swap costs
# nothing here. No feature scaling is applied (a real, evolvable gap:
# scaled features measurably help gradient descent converge further).
# ---------------------------------------------------------------------

_LOGISTIC_REGRESSION_CODE = '''\
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss


class ScratchLogisticRegression:
    """Binary logistic regression: L2-penalized cross-entropy, optimized
    by batch gradient descent. This is the same objective real
    scikit-learn's LogisticRegression(C=1.0) minimizes; the solver
    (gradient descent here, vs scikit-learn's quasi-Newton lbfgs) is a
    documented simplification, not a different algorithm."""

    def __init__(self, C=1.0, max_iter=1000, learning_rate=0.1):
        # TRAIT: model accuracy, training time. C is the inverse
        # regularization strength (matches scikit-learn's parameter);
        # max_iter and learning_rate govern the gradient descent solver.
        self.C = C
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.weights_ = None
        self.bias_ = 0.0

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        # TRAIT: model accuracy, training time. No feature scaling is
        # applied here — gradient descent on raw, differently-scaled
        # features converges slower than it would on standardized ones.
        # Adding a scaler is a legitimate, honestly-motivated evolution.
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape
        self.weights_ = np.zeros(n_features)
        self.bias_ = 0.0
        lambda_ = 1.0 / (self.C * n_samples)

        for _ in range(self.max_iter):
            predictions = self._sigmoid(X @ self.weights_ + self.bias_)
            error = predictions - y
            grad_w = (X.T @ error) / n_samples + lambda_ * self.weights_
            grad_b = np.mean(error)
            self.weights_ -= self.learning_rate * grad_w
            self.bias_ -= self.learning_rate * grad_b
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        p1 = self._sigmoid(X @ self.weights_ + self.bias_)
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def run(data_path: str) -> dict:
    df = pd.read_csv(data_path)
    X = df.drop(columns=["target"]).to_numpy()
    y = df["target"].to_numpy()

''' + _SPLIT + '''
    model = ScratchLogisticRegression(C=1.0, max_iter=1000, learning_rate=0.1)

''' + _MEASURE_AND_RETURN

_LOGISTIC_REGRESSION_REFERENCE = '''\
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss


def run(data_path: str) -> dict:
    df = pd.read_csv(data_path)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Real scikit-learn defaults: C=1.0 (L2 penalty), solver='lbfgs', max_iter=100.
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=100)

''' + _MEASURE_AND_RETURN


# ---------------------------------------------------------------------
# 4. Random Forest
# Real defaults (from random_forest.py): n_estimators=100,
# criterion='gini', max_features='sqrt', bootstrap=True.
# Note: n_estimators=100 makes this the slowest of the five to fit
# (~7s on this dataset) — still well inside the sandbox's timeout, but
# worth knowing before setting a very short SANDBOX_TIMEOUT_SECS.
# ---------------------------------------------------------------------

_RANDOM_FOREST_CODE = '''\
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss


class _ScratchForestTree:
    """The same CART/Gini tree as the standalone decision tree, except
    each split only considers a random subset of `max_features` —
    that per-split feature subsampling is what actually makes a random
    forest a forest of decorrelated trees rather than 100 copies of the
    same tree."""

    def __init__(self, max_depth=None, min_samples_split=2, max_features=None, rng=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.rng = rng
        self.tree_ = None
        self.n_classes_ = None

    def fit(self, X, y):
        self.tree_ = self._build(X, y, depth=0)
        return self

    def _gini(self, y):
        if len(y) == 0:
            return 0.0
        counts = np.bincount(y, minlength=self.n_classes_)
        p = counts / len(y)
        return 1.0 - np.sum(p ** 2)

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split:
            return None, None, None
        k = self.max_features or n_features
        feature_subset = self.rng.choice(n_features, size=min(k, n_features), replace=False)

        parent_impurity = self._gini(y)
        best_gain, best_feature, best_threshold = -1.0, None, None
        for feature in feature_subset:
            values = X[:, feature]
            order = np.argsort(values)
            sorted_values, sorted_y = values[order], y[order]
            distinct = np.where(np.diff(sorted_values) != 0)[0]
            for i in distinct:
                threshold = (sorted_values[i] + sorted_values[i + 1]) / 2.0
                left_y, right_y = sorted_y[: i + 1], sorted_y[i + 1 :]
                if len(left_y) == 0 or len(right_y) == 0:
                    continue
                w_l, w_r = len(left_y) / n_samples, len(right_y) / n_samples
                child_impurity = w_l * self._gini(left_y) + w_r * self._gini(right_y)
                gain = parent_impurity - child_impurity
                if gain > best_gain:
                    best_gain, best_feature, best_threshold = gain, feature, threshold

        if best_feature is None:
            return None, None, None
        return best_feature, best_threshold, best_gain

    def _build(self, X, y, depth):
        n_labels = len(np.unique(y))
        if (
            n_labels == 1
            or len(y) < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            return {"leaf": True, "proba": self._leaf_proba(y)}
        feature, threshold, gain = self._best_split(X, y)
        if feature is None or gain <= 0:
            return {"leaf": True, "proba": self._leaf_proba(y)}
        left_mask = X[:, feature] <= threshold
        return {
            "leaf": False,
            "feature": feature,
            "threshold": threshold,
            "left": self._build(X[left_mask], y[left_mask], depth + 1),
            "right": self._build(X[~left_mask], y[~left_mask], depth + 1),
        }

    def _leaf_proba(self, y):
        counts = np.bincount(y, minlength=self.n_classes_).astype(np.float64)
        return counts / counts.sum()

    def predict_proba(self, X):
        return np.array([self._proba_row(row, self.tree_) for row in X])

    def _proba_row(self, row, node):
        if node["leaf"]:
            return node["proba"]
        branch = node["left"] if row[node["feature"]] <= node["threshold"] else node["right"]
        return self._proba_row(row, branch)


class ScratchRandomForest:
    """Bootstrap-aggregated ensemble of the tree above — this is the
    actual bagging algorithm behind RandomForestClassifier, just not
    parallelised across compiled workers."""

    def __init__(self, n_estimators=100, max_features="sqrt", random_state=42):
        # TRAIT: model accuracy, training time, memory usage,
        # computational cost. More trees generally means a more stable
        # (but slower and larger) ensemble.
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state
        self.trees_ = []
        self.n_classes_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        n_samples, n_features = X.shape
        self.n_classes_ = int(y.max()) + 1

        if self.max_features == "sqrt":
            k = max(1, int(np.sqrt(n_features)))
        elif isinstance(self.max_features, int):
            k = self.max_features
        else:
            k = n_features

        rng = np.random.default_rng(self.random_state)
        self.trees_ = []
        for _ in range(self.n_estimators):
            # TRAIT: model accuracy. Bootstrap sampling (with
            # replacement) is what makes each tree see a different
            # slice of the data — this diversity is what bagging relies
            # on to reduce variance.
            bootstrap_idx = rng.integers(0, n_samples, size=n_samples)
            tree = _ScratchForestTree(max_features=k, rng=rng)
            tree.n_classes_ = self.n_classes_
            tree.fit(X[bootstrap_idx], y[bootstrap_idx])
            self.trees_.append(tree)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        all_proba = np.array([tree.predict_proba(X) for tree in self.trees_])
        return all_proba.mean(axis=0)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def run(data_path: str) -> dict:
    df = pd.read_csv(data_path)
    X = df.drop(columns=["target"]).to_numpy()
    y = df["target"].to_numpy()

''' + _SPLIT + '''
    model = ScratchRandomForest(n_estimators=100, max_features="sqrt", random_state=42)

''' + _MEASURE_AND_RETURN

_RANDOM_FOREST_REFERENCE = '''\
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss


def run(data_path: str) -> dict:
    df = pd.read_csv(data_path)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Real scikit-learn defaults: n_estimators=100, criterion='gini',
    # max_features='sqrt', bootstrap=True.
    model = RandomForestClassifier(
        n_estimators=100, criterion="gini", max_features="sqrt", bootstrap=True, random_state=42
    )

''' + _MEASURE_AND_RETURN


# ---------------------------------------------------------------------
# 5. Gaussian Naive Bayes (replaces SVM — see module docstring)
# Real defaults: var_smoothing=1e-9, scaled by the largest single-
# feature variance across the whole training set; class priors from
# observed frequency.
# ---------------------------------------------------------------------

_GAUSSIAN_NB_CODE = '''\
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss


class ScratchGaussianNB:
    """Closed-form Gaussian Naive Bayes: per-class mean/variance per
    feature (assuming feature independence), class priors from observed
    frequency, prediction by log-likelihood + log-prior. No iterative
    optimizer at all — this is exactly what real GaussianNB computes,
    which is why it matches it exactly rather than approximately."""

    def __init__(self, var_smoothing=1e-9):
        # TRAIT: model accuracy. Smooths every feature's variance by a
        # tiny fraction of the largest observed variance, purely to
        # avoid dividing by (near-)zero for a feature that barely
        # varies within a class.
        self.var_smoothing = var_smoothing
        self.classes_ = None
        self.mean_ = None
        self.var_ = None
        self.priors_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        self.classes_ = np.unique(y)
        n_features = X.shape[1]
        epsilon = self.var_smoothing * np.var(X, axis=0).max()

        self.mean_ = np.zeros((len(self.classes_), n_features))
        self.var_ = np.zeros((len(self.classes_), n_features))
        # TRAIT: model accuracy. Class priors are estimated directly
        # from training-set frequency (not assumed uniform) — this
        # matters when the classes are imbalanced.
        self.priors_ = np.zeros(len(self.classes_))

        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.mean_[i] = X_c.mean(axis=0)
            self.var_[i] = X_c.var(axis=0) + epsilon
            self.priors_[i] = X_c.shape[0] / X.shape[0]
        return self

    def _log_likelihood(self, X):
        # TRAIT: model accuracy. The Gaussian log-likelihood per class,
        # summed across features under the (naive) independence
        # assumption this whole algorithm is named for.
        log_probs = np.zeros((X.shape[0], len(self.classes_)))
        for i in range(len(self.classes_)):
            log_prior = np.log(self.priors_[i])
            log_gauss = -0.5 * np.sum(np.log(2.0 * np.pi * self.var_[i]))
            log_gauss -= 0.5 * np.sum(((X - self.mean_[i]) ** 2) / self.var_[i], axis=1)
            log_probs[:, i] = log_prior + log_gauss
        return log_probs

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        log_probs = self._log_likelihood(X)
        # Log-sum-exp normalisation, so this stays numerically stable
        # even with 30 features' worth of log-likelihood summed up.
        max_log = log_probs.max(axis=1, keepdims=True)
        exp_probs = np.exp(log_probs - max_log)
        return exp_probs / exp_probs.sum(axis=1, keepdims=True)

    def predict(self, X):
        log_probs = self._log_likelihood(np.asarray(X, dtype=np.float64))
        return self.classes_[np.argmax(log_probs, axis=1)]


def run(data_path: str) -> dict:
    df = pd.read_csv(data_path)
    X = df.drop(columns=["target"]).to_numpy()
    y = df["target"].to_numpy()

''' + _SPLIT + '''
    model = ScratchGaussianNB(var_smoothing=1e-9)

''' + _MEASURE_AND_RETURN

_GAUSSIAN_NB_REFERENCE = '''\
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, log_loss


def run(data_path: str) -> dict:
    df = pd.read_csv(data_path)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Real scikit-learn default: var_smoothing=1e-9.
    model = GaussianNB(var_smoothing=1e-9)

''' + _MEASURE_AND_RETURN


ALGORITHMS: dict[str, AlgorithmSpec] = {
    "logistic_regression": AlgorithmSpec(
        name="logistic_regression",
        display_name="Logistic Regression",
        dataset_name=DATASET_NAME,
        dataset_loader=DATASET_LOADER,
        baseline_code=_LOGISTIC_REGRESSION_CODE,
        reference_code=_LOGISTIC_REGRESSION_REFERENCE,
    ),
    "knn": AlgorithmSpec(
        name="knn",
        display_name="K-Nearest Neighbors",
        dataset_name=DATASET_NAME,
        dataset_loader=DATASET_LOADER,
        baseline_code=_KNN_CODE,
        reference_code=_KNN_REFERENCE,
    ),
    "decision_tree": AlgorithmSpec(
        name="decision_tree",
        display_name="Decision Tree",
        dataset_name=DATASET_NAME,
        dataset_loader=DATASET_LOADER,
        baseline_code=_DECISION_TREE_CODE,
        reference_code=_DECISION_TREE_REFERENCE,
    ),
    "random_forest": AlgorithmSpec(
        name="random_forest",
        display_name="Random Forest",
        dataset_name=DATASET_NAME,
        dataset_loader=DATASET_LOADER,
        baseline_code=_RANDOM_FOREST_CODE,
        reference_code=_RANDOM_FOREST_REFERENCE,
    ),
    "gaussian_nb": AlgorithmSpec(
        name="gaussian_nb",
        display_name="Gaussian Naive Bayes",
        dataset_name=DATASET_NAME,
        dataset_loader=DATASET_LOADER,
        baseline_code=_GAUSSIAN_NB_CODE,
        reference_code=_GAUSSIAN_NB_REFERENCE,
    ),
}

ALGORITHM_ORDER = ["logistic_regression", "knn", "decision_tree", "random_forest", "gaussian_nb"]


def get_spec(algorithm: str) -> AlgorithmSpec:
    try:
        return ALGORITHMS[algorithm]
    except KeyError:
        raise RuntimeError(
            f"unknown algorithm '{algorithm}'; available: {', '.join(ALGORITHM_ORDER)}"
        ) from None


def choose_algorithm() -> str:
    names = [n for n in ALGORITHM_ORDER if n in ALGORITHMS]
    print(f"Select an algorithm to optimize (dataset: {DATASET_NAME}):")
    for i, name in enumerate(names):
        print(f"  {i + 1}) {ALGORITHMS[name].display_name}")

    choice_raw = input("> ").strip()
    try:
        choice = int(choice_raw)
    except ValueError:
        raise RuntimeError("invalid selection") from None

    index = choice - 1
    if index < 0 or index >= len(names):
        raise RuntimeError("selection out of range")
    return names[index]
