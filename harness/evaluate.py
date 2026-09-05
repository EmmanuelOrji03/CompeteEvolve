"""CompeteEvolve evaluation harness.

Runs a candidate implementation against a fixed set of scikit-learn datasets and
prints a single JSON object describing what happened. This is the *only* source
of fitness in the system: the Rust side never estimates speed or accuracy on its
own.

Usage:
    python harness/evaluate.py --task kmeans --candidate path/to/candidate.py
    python harness/evaluate.py --task knn --sklearn-reference

Candidate contract (see harness/tasks/*.py):
    * The file must define a class named ``Model``.
    * ``Model().fit(X, y)`` for classification tasks, ``Model().fit(X)`` for
      clustering tasks. ``fit`` may return anything.
    * ``Model().predict(X)`` returns a 1-D integer array of labels.
    * Clustering models are constructed as ``Model(n_clusters=k)``.

Output JSON (always printed, exit code 0 unless the harness itself is broken):
    {
      "status": "ok" | "syntax_error" | "import_error" | "runtime_error",
      "task": str,
      "syntax_error": str | null,
      "runtime_errors": [str, ...],       # one entry per failed fold
      "n_folds_total": int,
      "n_folds_ok": int,
      "fit_time": float,                  # seconds per fold: median within a dataset, mean across datasets
      "predict_time": float,
      "quality": float,                   # mean task quality in [0, 1], higher is better
      "quality_std": float,
      "per_dataset": [ {name, quality, fit_time, predict_time, n_ok, n_total}, ... ],
      "code_length": int                  # characters of the candidate file
    }
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
import time
import traceback
import types
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent


CLASSIFICATION_DATASETS = ["iris", "wine", "breast_cancer", "digits"]
CLUSTERING_DATASETS = ["iris", "wine", "digits"]

TASKS = {
    "kmeans": {
        "kind": "clustering",
        "datasets": CLUSTERING_DATASETS,
        "seed_file": HERE / "tasks" / "kmeans.py",
        "description": (
            "K-means clustering. Quality is the adjusted Rand index between the "
            "predicted cluster labels and the true class labels (clipped to [0, 1])."
        ),
    },
    "logistic_regression": {
        "kind": "classification",
        "datasets": CLASSIFICATION_DATASETS,
        "seed_file": HERE / "tasks" / "logistic_regression.py",
        "description": "Multiclass logistic regression. Quality is held-out accuracy.",
    },
    "knn": {
        "kind": "classification",
        "datasets": CLASSIFICATION_DATASETS,
        "seed_file": HERE / "tasks" / "knn.py",
        "description": "k-nearest-neighbours classifier (k=5). Quality is held-out accuracy.",
    },
}


def load_dataset(name: str):
    from sklearn import datasets as skd

    loader = {
        "iris": skd.load_iris,
        "wine": skd.load_wine,
        "breast_cancer": skd.load_breast_cancer,
        "digits": skd.load_digits,
    }[name]
    bunch = loader()
    X = np.asarray(bunch.data, dtype=np.float64)
    y = np.asarray(bunch.target, dtype=np.int64)
    return X, y




def check_syntax(source: str) -> str | None:
    try:
        ast.parse(source)
        return None
    except SyntaxError as exc:  # pragma: no cover - exercised by tests via CLI
        return f"{exc.msg} (line {exc.lineno}, col {exc.offset})"


# Only NumPy and the standard library: wrapping scikit-learn would be a hollow win.
ALLOWED_IMPORT_ROOTS = {
    "numpy", "math", "itertools", "functools", "collections", "typing",
    "heapq", "bisect", "random", "operator", "dataclasses", "abc", "numbers",
    "warnings", "time",
}


def check_imports(source: str) -> list[str]:
    tree = ast.parse(source)
    bad = []
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for name in names:
            root = name.split(".")[0]
            if root not in ALLOWED_IMPORT_ROOTS and root not in bad:
                bad.append(root)
    return bad


def load_candidate_module(path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("competeevolve_candidate", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "Model"):
        raise ImportError("candidate file does not define a class named `Model`")
    return module


def sklearn_reference_factory(task: str):
    if task == "kmeans":
        from sklearn.cluster import KMeans

        return lambda n_clusters: KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    if task == "logistic_regression":
        from sklearn.linear_model import LogisticRegression

        return lambda: LogisticRegression(max_iter=1000)
    if task == "knn":
        from sklearn.neighbors import KNeighborsClassifier

        return lambda: KNeighborsClassifier(n_neighbors=5)
    raise ValueError(task)




def evaluate_classification(make_model, X, y, folds: int, repeats: int, seed: int):
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.preprocessing import StandardScaler

    splitter = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    results = []
    errors = []
    for train_idx, test_idx in splitter.split(X, y):
        scaler = StandardScaler().fit(X[train_idx])
        Xtr = scaler.transform(X[train_idx])
        Xte = scaler.transform(X[test_idx])
        ytr, yte = y[train_idx], y[test_idx]
        try:
            np.random.seed(seed)
            model = make_model()
            t0 = time.perf_counter()
            model.fit(Xtr, ytr)
            fit_time = time.perf_counter() - t0
            t0 = time.perf_counter()
            pred = np.asarray(model.predict(Xte)).ravel()
            predict_time = time.perf_counter() - t0
            if pred.shape[0] != yte.shape[0]:
                raise ValueError(
                    f"predict returned {pred.shape[0]} labels for {yte.shape[0]} rows"
                )
            if not np.all(np.isfinite(pred.astype(np.float64))):
                raise ValueError("predict returned non-finite values")
            acc = float(np.mean(pred.astype(np.int64) == yte))
            results.append((acc, fit_time, predict_time))
        except Exception as exc:  # noqa: BLE001 - candidate code may raise anything
            errors.append(_short_traceback(exc))
    return results, errors


def evaluate_clustering(make_model, X, y, folds: int, repeats: int, seed: int):
    """No train/test split; each 'fold' is a different seed."""
    from sklearn.metrics import adjusted_rand_score
    from sklearn.preprocessing import StandardScaler

    Xs = StandardScaler().fit_transform(X)
    k = int(len(np.unique(y)))
    results = []
    errors = []
    n_runs = folds * repeats
    for run in range(n_runs):
        try:
            np.random.seed(seed + run)
            model = make_model(k)
            t0 = time.perf_counter()
            model.fit(Xs)
            fit_time = time.perf_counter() - t0
            t0 = time.perf_counter()
            pred = np.asarray(model.predict(Xs)).ravel()
            predict_time = time.perf_counter() - t0
            if pred.shape[0] != y.shape[0]:
                raise ValueError(
                    f"predict returned {pred.shape[0]} labels for {y.shape[0]} rows"
                )
            ari = float(adjusted_rand_score(y, pred.astype(np.int64)))
            results.append((min(max(ari, 0.0), 1.0), fit_time, predict_time))
        except Exception as exc:  # noqa: BLE001
            errors.append(_short_traceback(exc))
    return results, errors


def _short_traceback(exc: BaseException) -> str:
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    tail = "".join(tb[-4:]).strip()
    return tail[-800:]


def run(task_name: str, candidate: Path | None, use_sklearn: bool, folds: int, repeats: int, seed: int):
    task = TASKS[task_name]
    out = {
        "status": "ok",
        "task": task_name,
        "kind": task["kind"],
        "syntax_error": None,
        "runtime_errors": [],
        "n_folds_total": 0,
        "n_folds_ok": 0,
        "fit_time": None,
        "predict_time": None,
        "quality": None,
        "quality_std": None,
        "per_dataset": [],
        "code_length": 0,
    }

    if use_sklearn:
        factory = sklearn_reference_factory(task_name)
        make_model = factory
    else:
        assert candidate is not None
        source = candidate.read_text(encoding="utf-8")
        out["code_length"] = len(source)
        syntax_error = check_syntax(source)
        if syntax_error:
            out["status"] = "syntax_error"
            out["syntax_error"] = syntax_error
            return out
        forbidden = check_imports(source)
        if forbidden:
            out["status"] = "forbidden_import"
            out["runtime_errors"].append(
                "forbidden imports: " + ", ".join(forbidden)
                + ". Only numpy and the standard library are allowed; implement the algorithm yourself."
            )
            return out
        try:
            module = load_candidate_module(candidate)
        except Exception as exc:  # noqa: BLE001
            out["status"] = "import_error"
            out["runtime_errors"].append(_short_traceback(exc))
            return out
        Model = module.Model
        if task["kind"] == "clustering":
            make_model = lambda k: Model(n_clusters=k)  # noqa: E731
        else:
            make_model = lambda: Model()  # noqa: E731

    all_quality = []
    all_fit = []
    all_pred = []
    for ds_name in task["datasets"]:
        X, y = load_dataset(ds_name)
        if task["kind"] == "clustering":
            results, errors = evaluate_clustering(make_model, X, y, folds, repeats, seed)
        else:
            results, errors = evaluate_classification(make_model, X, y, folds, repeats, seed)
        n_total = len(results) + len(errors)
        out["n_folds_total"] += n_total
        out["n_folds_ok"] += len(results)
        out["runtime_errors"].extend(f"[{ds_name}] {e}" for e in errors)
        entry = {
            "name": ds_name,
            "n_ok": len(results),
            "n_total": n_total,
            "quality": None,
            "fit_time": None,
            "predict_time": None,
        }
        if results:
            q = [r[0] for r in results]
            f = [r[1] for r in results]
            p = [r[2] for r in results]
            # Median time per dataset: one disturbed fold cannot skew it.
            entry.update(
                quality=float(np.mean(q)), fit_time=float(np.median(f)), predict_time=float(np.median(p))
            )
            all_quality.extend(q)
            all_fit.extend(f)
            all_pred.extend(p)
        out["per_dataset"].append(entry)

    if all_quality:
        out["quality"] = float(np.mean(all_quality))
        out["quality_std"] = float(np.std(all_quality))
        out["fit_time"] = float(np.mean([d["fit_time"] for d in out["per_dataset"] if d["fit_time"] is not None]))
        out["predict_time"] = float(np.mean([d["predict_time"] for d in out["per_dataset"] if d["predict_time"] is not None]))
    if out["n_folds_ok"] == 0:
        out["status"] = "runtime_error"
    seen = set()
    unique_errors = []
    for e in out["runtime_errors"]:
        if e not in seen:
            seen.add(e)
            unique_errors.append(e)
    out["runtime_errors"] = unique_errors[:10]
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list-tasks", action="store_true", help="print the task registry as JSON and exit")
    parser.add_argument("--task", choices=sorted(TASKS))
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--candidate", type=Path, help="path to a candidate .py file")
    group.add_argument("--seed-file", action="store_true", help="evaluate the task's seed implementation")
    group.add_argument("--sklearn-reference", action="store_true", help="evaluate the scikit-learn reference model")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--describe", action="store_true", help="print the task description and seed path, then exit")
    args = parser.parse_args(argv)

    if args.list_tasks:
        print(json.dumps([
            {"task": name, "kind": t["kind"], "datasets": t["datasets"],
             "description": t["description"], "seed_file": str(t["seed_file"])}
            for name, t in TASKS.items()
        ]))
        return 0
    if not args.task:
        parser.error("--task is required")
    if not (args.candidate or args.seed_file or args.sklearn_reference or args.describe):
        parser.error("one of --candidate, --seed-file, --sklearn-reference is required")

    task = TASKS[args.task]
    if args.describe:
        print(json.dumps({"task": args.task, "kind": task["kind"], "datasets": task["datasets"],
                          "description": task["description"], "seed_file": str(task["seed_file"])}))
        return 0

    candidate = task["seed_file"] if args.seed_file else args.candidate
    try:
        result = run(args.task, candidate, args.sklearn_reference, args.folds, args.repeats, args.seed)
    except Exception as exc:  # harness bug, not candidate bug
        result = {"status": "harness_error", "task": args.task, "runtime_errors": [_short_traceback(exc)]}
        print(json.dumps(result))
        return 2
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
