from __future__ import annotations

import json

import numpy as np
import pandas as pd


REQUIRED_SLICE_COLUMNS = {
    "subject_id",
    "slice_index",
    "y_true",
    "p_positive",
    "is_positive_prediction",
}


def aggregate_slice_predictions(
    slice_predictions: pd.DataFrame,
    expected_slices: int = 5,
    patient_threshold: float = 0.5,
    method: str = "soft",
    positive_label: int = 1,
    weights: list[float] | tuple[float, ...] | np.ndarray | None = None,
    threshold_inclusive: bool = False,
) -> pd.DataFrame:
    """Aggregate five independently classified slices into one subject score.

    ``weighted_soft`` implements the fixed-weight MSV used by the retained
    source (weights are ordered by ascending ``slice_index``). Weights are
    supplied by configuration and are never estimated from labels here.
    """
    missing = REQUIRED_SLICE_COLUMNS.difference(slice_predictions.columns)
    if missing:
        raise ValueError(f"Slice predictions are missing columns: {sorted(missing)}")
    if method not in {"soft", "hard", "weighted_soft"}:
        raise ValueError("method must be 'soft', 'hard', or 'weighted_soft'")
    if positive_label not in {0, 1}:
        raise ValueError("positive_label must be 0 or 1")

    normalized_weights: np.ndarray | None = None
    if method == "weighted_soft":
        if weights is None:
            raise ValueError("weighted_soft requires explicit fixed weights")
        normalized_weights = np.asarray(weights, dtype=float)
        if normalized_weights.ndim != 1 or len(normalized_weights) != expected_slices:
            raise ValueError(
                f"weights must contain exactly {expected_slices} values in slice order"
            )
        if not np.isfinite(normalized_weights).all() or (normalized_weights < 0).any():
            raise ValueError("weights must be finite and non-negative")
        weight_sum = float(normalized_weights.sum())
        if weight_sum <= 0:
            raise ValueError("At least one weight must be positive")
        normalized_weights = normalized_weights / weight_sum

    rows: list[dict[str, object]] = []
    for subject_id, group in slice_predictions.groupby("subject_id", sort=True):
        ordered = group.sort_values("slice_index")
        if ordered["y_true"].nunique() != 1:
            raise ValueError(f"Subject {subject_id} has inconsistent y_true values")
        if ordered["slice_index"].nunique() != expected_slices or len(ordered) != expected_slices:
            raise ValueError(
                f"Subject {subject_id} must have exactly {expected_slices} unique slices"
            )

        scores = ordered["p_positive"].astype(float).to_numpy()
        slice_votes = ordered["is_positive_prediction"].astype(int).to_numpy()
        if not set(np.unique(slice_votes)).issubset({0, 1}):
            raise ValueError("is_positive_prediction must contain only 0 and 1")
        soft_score = float(scores.mean())
        vote_fraction = float(slice_votes.mean())
        if method == "soft":
            patient_score = soft_score
        elif method == "hard":
            patient_score = vote_fraction
        else:
            assert normalized_weights is not None
            patient_score = float(np.dot(scores, normalized_weights))
        is_positive_prediction = int(
            patient_score >= patient_threshold
            if threshold_inclusive
            else patient_score > patient_threshold
        )
        patient_prediction = positive_label if is_positive_prediction else 1 - positive_label
        if "predicted_class_label" in ordered:
            slice_classes = ordered["predicted_class_label"].astype(int).tolist()
        else:
            slice_classes = [positive_label if vote else 1 - positive_label for vote in slice_votes]

        rows.append(
            {
                "subject_id": str(subject_id),
                "y_true": int(ordered["y_true"].iloc[0]),
                "patient_score": patient_score,
                "patient_prediction": patient_prediction,
                "is_positive_prediction": is_positive_prediction,
                "positive_label": positive_label,
                "soft_score": soft_score,
                "positive_vote_fraction": vote_fraction,
                "n_positive_votes": int(slice_votes.sum()),
                "n_slices": expected_slices,
                "slice_indices": json.dumps(ordered["slice_index"].astype(int).tolist()),
                "slice_scores": json.dumps(np.round(scores, 8).tolist()),
                "slice_predicted_classes": json.dumps(slice_classes),
                "slice_positive_indicators": json.dumps(slice_votes.tolist()),
                "slice_weights": (
                    json.dumps(np.round(normalized_weights, 8).tolist())
                    if normalized_weights is not None
                    else None
                ),
                "aggregation_method": method,
                "patient_threshold": patient_threshold,
                "threshold_inclusive": bool(threshold_inclusive),
            }
        )
    return pd.DataFrame(rows)
