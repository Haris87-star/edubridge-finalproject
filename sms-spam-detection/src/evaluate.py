"""Metric computation, reporting and persistence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .config import LABELS, METRICS_FILE, SPAM


def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Return a JSON-serialisable metrics dictionary."""
    y_true = list(y_true)
    y_pred = list(y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(LABELS), zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(LABELS))

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "spam": {
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
            "support": int(support[1]),
        },
        "ham": {
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
            "support": int(support[0]),
        },
        # rows = truth (ham, spam), columns = prediction (ham, spam)
        "confusion_matrix": {
            "labels": list(LABELS),
            "matrix": matrix.tolist(),
        },
        "classification_report": classification_report(
            y_true, y_pred, labels=list(LABELS), output_dict=True, zero_division=0
        ),
    }

    if y_proba is not None:
        metrics["roc_auc"] = float(
            roc_auc_score([1 if label == SPAM else 0 for label in y_true], y_proba)
        )

    return _to_python(metrics)


def confusion_text(metrics: dict) -> str:
    matrix = metrics["confusion_matrix"]["matrix"]
    truth = metrics["confusion_matrix"]["labels"]
    return (
        "                predicted ham   predicted spam\n"
        f"actual ham      {matrix[0][0]:>13,}   {matrix[0][1]:>14,}\n"
        f"actual spam     {matrix[1][0]:>13,}   {matrix[1][1]:>14,}\n"
        f"(labels: {truth[0]}, {truth[1]})"
    )


def format_report(metrics: dict) -> str:
    spam, ham = metrics["spam"], metrics["ham"]
    lines = [
        "",
        "=" * 58,
        " Evaluation on the held-out test set",
        "=" * 58,
        f" Accuracy ......... {metrics['accuracy']:.4f}",
        f" Macro F1 ......... {metrics['macro_f1']:.4f}",
        f" Weighted F1 ...... {metrics['weighted_f1']:.4f}",
    ]
    if "roc_auc" in metrics:
        lines.append(f" ROC AUC .......... {metrics['roc_auc']:.4f}")
    lines += [
        "",
        " Spam (positive class)",
        f"   precision ...... {spam['precision']:.4f}",
        f"   recall ......... {spam['recall']:.4f}",
        f"   f1 ............. {spam['f1']:.4f}",
        f"   support ........ {spam['support']:,}",
        "",
        " Ham",
        f"   precision ...... {ham['precision']:.4f}",
        f"   recall ......... {ham['recall']:.4f}",
        f"   f1 ............. {ham['f1']:.4f}",
        "",
        confusion_text(metrics),
        "=" * 58,
    ]
    return "\n".join(lines)


def save_metrics(metrics: dict, path: Path = METRICS_FILE) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path


def load_metrics(path: Path = METRICS_FILE) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _to_python(obj):
    """Convert numpy scalars/arrays so json.dump works."""
    if isinstance(obj, dict):
        return {str(k): _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
