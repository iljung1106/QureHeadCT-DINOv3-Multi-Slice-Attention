from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


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

