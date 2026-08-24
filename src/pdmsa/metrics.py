from __future__ import annotations

from typing import Any

import numpy as np


def classification_metrics(
    y_true,
    y_score,
    threshold: float = 0.5,
    positive_label: int = 1,
    threshold_inclusive: bool = True,
) -> dict[str, Any]:
    """Compute patient-level metrics with an explicit positive class."""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        roc_auc_score,
    )

    truth = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    if positive_label not in {0, 1}:
        raise ValueError("positive_label must be 0 or 1")
    positive_truth = (truth == positive_label).astype(int)
    positive_prediction = (score >= threshold if threshold_inclusive else score > threshold).astype(
        int
    )
    tn, fp, fn, tp = confusion_matrix(positive_truth, positive_prediction, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    return {
        "n": int(len(truth)),
        "positive_label": int(positive_label),
        "threshold": float(threshold),
        "threshold_inclusive": bool(threshold_inclusive),
        "accuracy": float(accuracy_score(positive_truth, positive_prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(positive_truth, positive_prediction)),
        "macro_f1": float(f1_score(positive_truth, positive_prediction, average="macro")),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "roc_auc": float(roc_auc_score(positive_truth, score)),
        "pr_auc": float(average_precision_score(positive_truth, score)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def bootstrap_confidence_intervals(
    y_true,
    y_score,
    threshold: float = 0.5,
    positive_label: int = 1,
    repetitions: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
    threshold_inclusive: bool = True,
) -> dict[str, dict[str, float]]:
    """Patient-level nonparametric bootstrap intervals."""
    truth = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    rng = np.random.default_rng(seed)
    collected: dict[str, list[float]] = {}
    for _ in range(repetitions):
        indices = rng.integers(0, len(truth), len(truth))
        sampled_truth = truth[indices]
        if np.unique(sampled_truth).size < 2:
            continue
        metrics = classification_metrics(
            sampled_truth,
            score[indices],
            threshold=threshold,
            positive_label=positive_label,
            threshold_inclusive=threshold_inclusive,
        )
        for key, value in metrics.items():
            if key in {
                "n",
                "positive_label",
                "threshold",
                "threshold_inclusive",
                "tn",
                "fp",
                "fn",
                "tp",
            }:
                continue
            collected.setdefault(key, []).append(float(value))

    intervals: dict[str, dict[str, float]] = {}
    for key, values in collected.items():
        intervals[key] = {
            "lower": float(np.quantile(values, alpha / 2)),
            "upper": float(np.quantile(values, 1 - alpha / 2)),
            "valid_repetitions": int(len(values)),
        }
    return intervals
