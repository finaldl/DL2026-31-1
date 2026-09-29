"""Evaluate cached frozen features with kNN and a linear probe."""

from __future__ import annotations

import json
import os
import pickle
import time
import warnings

import numpy as np

from .metrics import macro_f1, top1_accuracy


_ALIASES = {
    "train_features": (
        "train_features", "x_train", "X_train", "train_x",
        "features_train", "train_feats",
    ),
    "train_labels": (
        "train_labels", "y_train", "train_y", "labels_train", "train_targets",
    ),
    "test_features": (
        "test_features", "x_test", "X_test", "test_x", "features_test",
        "test_feats", "val_features", "x_val",
    ),
    "test_labels": (
        "test_labels", "y_test", "test_y", "labels_test", "test_targets",
        "val_labels", "y_val",
    ),
}

# Aliases for single-split cache files (e.g. Thu's "*_trainval.pt" / "*_test.pt",
# each holding one split with generic "features"/"labels" keys rather than a
# combined train+test dictionary).
_SINGLE_SPLIT_ALIASES = {
    "features": ("features", "x", "X", "embeddings", "feats"),
    "labels": ("labels", "y", "targets"),
}


def l2_normalize(values, eps=1e-12):
    """L2-normalize each row of a two-dimensional feature matrix."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2-D feature matrix, got shape {values.shape}")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, eps)


def _to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _validate_features(features, source="feature cache"):
    train_x = np.asarray(features["train_features"])
    test_x = np.asarray(features["test_features"])
    train_y = np.asarray(features["train_labels"]).reshape(-1)
    test_y = np.asarray(features["test_labels"]).reshape(-1)

    if train_x.ndim != 2 or test_x.ndim != 2:
        raise ValueError(f"{source}: train/test features must both be 2-D")
    if train_x.shape[0] != train_y.size or test_x.shape[0] != test_y.size:
        raise ValueError(
            f"{source}: feature/label row counts do not match "
            f"(train={train_x.shape[0]}/{train_y.size}, "
            f"test={test_x.shape[0]}/{test_y.size})"
        )
    if train_x.shape[1] != test_x.shape[1]:
        raise ValueError(
            f"{source}: train/test feature dimensions differ "
            f"({train_x.shape[1]} != {test_x.shape[1]})"
        )
    if train_x.shape[0] == 0 or test_x.shape[0] == 0:
        raise ValueError(f"{source}: train/test features must not be empty")
    if not np.isfinite(train_x).all() or not np.isfinite(test_x).all():
        raise ValueError(f"{source}: features contain NaN or infinite values")

    train_classes = set(np.unique(train_y).tolist())
    test_classes = set(np.unique(test_y).tolist())
    missing_classes = test_classes - train_classes
    if missing_classes:
        raise ValueError(
            f"{source}: test labels contain classes absent from training: "
            f"{sorted(missing_classes)!r}"
        )

    return {
        "train_features": train_x.astype(np.float32, copy=False),
        "train_labels": train_y,
        "test_features": test_x.astype(np.float32, copy=False),
        "test_labels": test_y,
        "dim": int(train_x.shape[1]),
        "n_classes": len(train_classes),
    }


def _load_raw_cache(path):
    """Load a ``.pt``, ``.npz``, ``.pkl`` file into a plain dict of arrays."""
    extension = os.path.splitext(path)[1].lower()
    if extension == ".pt":
        try:
            import torch
        except ImportError as exc:
            raise ImportError("Loading .pt feature caches requires torch") from exc
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw_object = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(raw_object, dict):
            raise ValueError(f"{path}: expected a dictionary in the .pt cache")
        return {key: _to_numpy(value) for key, value in raw_object.items()}
    elif extension == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    elif extension in (".pkl", ".pickle"):
        with open(path, "rb") as handle:
            raw_object = pickle.load(handle)
        if not isinstance(raw_object, dict):
            raise ValueError(f"{path}: expected a dictionary in the pickle cache")
        return {key: _to_numpy(value) for key, value in raw_object.items()}
    else:
        raise ValueError(f"Unsupported feature-cache extension: {extension or '<none>'}")


def load_features(path):
    """Load and validate a combined train+test feature cache."""
    raw = _load_raw_cache(path)
    standardized = {}
    missing = []
    for standard_name, aliases in _ALIASES.items():
        alias = next((candidate for candidate in aliases if candidate in raw), None)
        if alias is None:
            missing.append(standard_name)
        else:
            standardized[standard_name] = np.asarray(raw[alias])
    if missing:
        raise KeyError(f"{path}: missing required arrays {missing}; found {sorted(raw)}")
    return _validate_features(standardized, source=path)


def _load_single_split(path):
    """Load one ``{features, labels}`` cache (one side of a train/test pair)."""
    raw = _load_raw_cache(path)
    values = {}
    missing = []
    for standard_name, aliases in _SINGLE_SPLIT_ALIASES.items():
        alias = next((candidate for candidate in aliases if candidate in raw), None)
        if alias is None:
            missing.append(standard_name)
        else:
            values[standard_name] = np.asarray(raw[alias])
    if missing:
        raise KeyError(f"{path}: missing required arrays {missing}; found {sorted(raw)}")
    return values["features"], values["labels"]


def load_split_features(train_path, test_path):
    """Load and validate a feature cache stored as two separate files.

    Thu's caches ship as ``{backbone}_{dataset}_trainval.pt`` /
    ``{backbone}_{dataset}_test.pt`` (or ``_train.pt`` for EuroSAT), each
    holding only a generic ``features``/``labels`` pair rather than a single
    combined train+test dictionary. This loads both halves and assembles
    them into the same standardized shape ``load_features`` returns.
    """
    train_x, train_y = _load_single_split(train_path)
    test_x, test_y = _load_single_split(test_path)
    return _validate_features(
        {
            "train_features": train_x,
            "train_labels": train_y,
            "test_features": test_x,
            "test_labels": test_y,
        },
        source=f"{train_path} + {test_path}",
    )


def validate_split_indices(indices, n_train, source="split"):
    """Validate uniqueness and bounds, returning an integer NumPy array."""
    if n_train is None or n_train <= 0:
        raise ValueError("n_train must be supplied as a positive integer")
    try:
        index_array = np.asarray(indices, dtype=np.int64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: split indices must be integers") from exc
    if index_array.ndim != 1 or index_array.size == 0:
        raise ValueError(f"{source}: split must be a non-empty one-dimensional list")
    if np.unique(index_array).size != index_array.size:
        raise ValueError(f"{source}: split contains duplicate indices")
    if index_array.min() < 0 or index_array.max() >= n_train:
        raise ValueError(
            f"{source}: index range [{index_array.min()}, {index_array.max()}] "
            f"is outside [0, {n_train})"
        )
    return index_array


def load_split_indices(splits_dir, dataset, seed, k, n_train=None):
    """Load ``{dataset}_seed{seed}_k{k}.json`` and validate its indices."""
    budget = "all" if str(k).lower() == "all" else str(int(k))
    path = os.path.join(splits_dir, f"{dataset}_seed{seed}_k{budget}.json")
    if not os.path.exists(path):
        if budget == "all":
            return None
        raise FileNotFoundError(f"Split file not found: {path}")
    with open(path, encoding="utf-8") as handle:
        indices = json.load(handle)
    return validate_split_indices(indices, n_train=n_train, source=path)


def subset_train(features, split_indices):
    """Select training rows from a validated feature-cache dictionary."""
    if split_indices is None:
        return features["train_features"], features["train_labels"]
    indices = validate_split_indices(
        split_indices, len(features["train_labels"]), source="split indices"
    )
    return features["train_features"][indices], features["train_labels"][indices]


def _prepare_eval_inputs(train_x, train_y, test_x, test_y, preprocess):
    train_x = np.asarray(train_x, dtype=np.float32)
    test_x = np.asarray(test_x, dtype=np.float32)
    train_y = np.asarray(train_y).reshape(-1)
    test_y = np.asarray(test_y).reshape(-1)
    _validate_features(
        {
            "train_features": train_x,
            "train_labels": train_y,
            "test_features": test_x,
            "test_labels": test_y,
        },
        source="evaluation input",
    )
    if preprocess == "l2":
        train_x, test_x = l2_normalize(train_x), l2_normalize(test_x)
    elif preprocess == "standard":
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        train_x, test_x = scaler.fit_transform(train_x), scaler.transform(test_x)
    elif preprocess != "none":
        raise ValueError(f"Unknown preprocessing mode: {preprocess}")
    return train_x, train_y, test_x, test_y


def knn_eval(
    train_x, train_y, test_x, test_y, k=20, preprocess="l2", weights="uniform"
):
    """Run cosine kNN, using majority vote by default."""
    from sklearn.neighbors import KNeighborsClassifier

    if not isinstance(k, (int, np.integer)) or k <= 0:
        raise ValueError("k must be a positive integer")
    if weights not in ("uniform", "distance"):
        raise ValueError("weights must be 'uniform' or 'distance'")
    started = time.perf_counter()
    train_x, train_y, test_x, test_y = _prepare_eval_inputs(
        train_x, train_y, test_x, test_y, preprocess
    )
    actual_k = min(int(k), len(train_y))
    if actual_k < k:
        warnings.warn(
            f"kNN k={k} exceeds n_train={len(train_y)}; using k={actual_k}",
            stacklevel=2,
        )
    classifier = KNeighborsClassifier(
        n_neighbors=actual_k, metric="cosine", algorithm="brute", weights=weights
    )
    classifier.fit(train_x, train_y)
    predictions = classifier.predict(test_x)
    return {
        "accuracy": top1_accuracy(test_y, predictions),
        "f1_macro": macro_f1(test_y, predictions),
        "k_actual": actual_k,
        "eval_time_sec": time.perf_counter() - started,
    }


def linear_probe_eval(
    train_x, train_y, test_x, test_y, C=1.0, max_iter=1000, preprocess="l2"
):
    """Fit multinomial logistic regression on frozen features."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    if C <= 0 or max_iter <= 0:
        raise ValueError("C and max_iter must be positive")
    started = time.perf_counter()
    train_x, train_y, test_x, test_y = _prepare_eval_inputs(
        train_x, train_y, test_x, test_y, preprocess
    )
    if np.unique(train_y).size < 2:
        raise ValueError("Linear probe requires at least two training classes")

    # scikit-learn >=1.8 removed multi_class; lbfgs selects multinomial for
    # multiclass data automatically.
    classifier = LogisticRegression(C=C, solver="lbfgs", max_iter=max_iter)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        classifier.fit(train_x, train_y)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        warnings.warn(
            f"Linear probe did not converge within {max_iter} iterations",
            stacklevel=2,
        )
    predictions = classifier.predict(test_x)
    return {
        "accuracy": top1_accuracy(test_y, predictions),
        "f1_macro": macro_f1(test_y, predictions),
        "n_iter": int(np.max(classifier.n_iter_)),
        "eval_time_sec": time.perf_counter() - started,
    }


def evaluate_one(
    features,
    split_indices,
    method,
    knn_k=20,
    knn_weights="uniform",
    lp_C=1.0,
    lp_max_iter=1000,
    preprocess="l2",
):
    """Evaluate one method and return metrics plus sample counts."""
    train_x, train_y = subset_train(features, split_indices)
    test_x, test_y = features["test_features"], features["test_labels"]
    if method == "knn":
        result = knn_eval(
            train_x, train_y, test_x, test_y,
            k=knn_k, preprocess=preprocess, weights=knn_weights,
        )
    elif method == "linear_probe":
        result = linear_probe_eval(
            train_x, train_y, test_x, test_y,
            C=lp_C, max_iter=lp_max_iter, preprocess=preprocess,
        )
    else:
        raise ValueError(f"Unknown evaluation method: {method}")
    result.update(num_train_samples=len(train_y), num_test_samples=len(test_y))
    return result
