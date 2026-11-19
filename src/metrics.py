"""Single metrics implementation used by EVERY module so numbers are computed identically."""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def compute_metrics(y_true, y_pred, labels) -> dict:
    """y_true / y_pred: label ids (ints). labels: list of label names, index = id.

    Returns {"accuracy", "macro_f1", "per_class_f1": {name: f1}, "confusion_matrix": [[int]]}
    confusion_matrix rows = true, cols = predicted.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    ids = list(range(len(labels)))
    per_class = f1_score(y_true, y_pred, labels=ids, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=ids, average="macro", zero_division=0)),
        "per_class_f1": {name: float(f) for name, f in zip(labels, per_class)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=ids).tolist(),
    }


def save_metrics(path, method: str, y_true, y_pred, labels, **extra) -> dict:
    """Compute metrics and write the standard result JSON (adds method and n_test)."""
    out = compute_metrics(y_true, y_pred, labels)
    out["method"] = method
    out["n_test"] = int(len(y_true))
    out.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return out
