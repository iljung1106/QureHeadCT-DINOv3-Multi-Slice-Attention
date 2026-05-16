from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score


def multilabel_metrics(y_true: np.ndarray, y_score: np.ndarray, labels: list[str]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    aucs = []
    aps = []
    for idx, label in enumerate(labels):
        truth = y_true[:, idx]
        score = y_score[:, idx]
        if len(np.unique(truth)) < 2:
            continue
        auc = float(roc_auc_score(truth, score))
        ap = float(average_precision_score(truth, score))
        metrics[f"{label}/auc"] = auc
        metrics[f"{label}/ap"] = ap
        aucs.append(auc)
        aps.append(ap)
    metrics["macro_auc"] = float(np.mean(aucs)) if aucs else float("nan")
    metrics["macro_ap"] = float(np.mean(aps)) if aps else float("nan")
    return metrics


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)
    metrics: dict[str, float] = {}
    if len(np.unique(y_true)) >= 2:
        metrics["auc"] = float(roc_auc_score(y_true, y_score))
        metrics["ap"] = float(average_precision_score(y_true, y_score))
    else:
        metrics["auc"] = float("nan")
        metrics["ap"] = float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["accuracy"] = float((tp + tn) / max(tp + tn + fp + fn, 1))
    metrics["sensitivity"] = float(tp / max(tp + fn, 1))
    metrics["specificity"] = float(tn / max(tn + fp, 1))
    metrics["precision"] = float(tp / max(tp + fp, 1))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics["positive_rate"] = float(y_true.mean()) if len(y_true) else float("nan")
    metrics["threshold"] = float(threshold)
    return metrics
