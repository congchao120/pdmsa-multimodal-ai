from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .manifest import DEFAULT_SLICE_OFFSETS

DEFAULT_MSV_WEIGHTS = (0.10, 0.20, 0.40, 0.20, 0.10)

REQUIRED_SLICE_COLUMNS = {
    "subject_id",
    "center_slice_index",
    "slice_offset",
    "slice_index",
    "y_true",
    "p_positive",
    "is_positive_prediction",
}


def _strict_integer_series(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="raise")
    array = values.to_numpy(dtype=float)
    if not np.isfinite(array).all() or not (array == np.floor(array)).all():
        raise ValueError(f"{column} must contain finite integer values")
    return values.astype(int)


def aggregate_slice_predictions(
    slice_predictions: pd.DataFrame,
    expected_slices: int = 5,
    patient_threshold: float = 0.5,
    method: str = "soft",
    positive_label: int = 1,
    weights: list[float] | tuple[float, ...] | np.ndarray | None = None,
    expected_slice_offsets: tuple[int, ...] | list[int] = DEFAULT_SLICE_OFFSETS,
    threshold_inclusive: bool = False,
) -> pd.DataFrame:
    """Aggregate five shared-ViT slice predictions into one subject score.

    ``weighted_soft`` applies weights in ascending relative ``slice_offset``
    order (center-2 through center+2), so patients may have different absolute
    center indices. Weights are supplied by configuration and are never
    estimated from labels here.
    """
    missing = REQUIRED_SLICE_COLUMNS.difference(slice_predictions.columns)
    if missing:
        raise ValueError(f"Slice predictions are missing columns: {sorted(missing)}")
    if method not in {"soft", "hard", "weighted_soft"}:
        raise ValueError("method must be 'soft', 'hard', or 'weighted_soft'")
    if positive_label not in {0, 1}:
        raise ValueError("positive_label must be 0 or 1")
    if not np.isfinite(patient_threshold) or not 0 <= patient_threshold <= 1:
        raise ValueError("patient_threshold must be finite and between 0 and 1")
    offset_array = np.asarray(expected_slice_offsets, dtype=float)
    if not np.isfinite(offset_array).all() or not (offset_array == np.floor(offset_array)).all():
        raise ValueError("expected_slice_offsets must contain finite integers")
    expected_offsets = tuple(offset_array.astype(int).tolist())
    if len(expected_offsets) != expected_slices or len(set(expected_offsets)) != expected_slices:
        raise ValueError("expected_slice_offsets must contain one unique value per slice")
    if expected_offsets != tuple(sorted(expected_offsets)):
        raise ValueError("expected_slice_offsets must be in ascending order")

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

    normalized = slice_predictions.copy()
    for column in (
        "center_slice_index",
        "slice_offset",
        "slice_index",
        "y_true",
        "is_positive_prediction",
    ):
        normalized[column] = _strict_integer_series(normalized, column)
    if not set(normalized["y_true"].unique()).issubset({0, 1}):
        raise ValueError("y_true must contain only class labels 0 and 1")
    if not set(normalized["is_positive_prediction"].unique()).issubset({0, 1}):
        raise ValueError("is_positive_prediction must contain only 0 and 1")
    normalized["p_positive"] = pd.to_numeric(normalized["p_positive"], errors="raise")
    score_array = normalized["p_positive"].to_numpy(dtype=float)
    if not np.isfinite(score_array).all() or ((score_array < 0) | (score_array > 1)).any():
        raise ValueError("p_positive must contain finite probabilities between 0 and 1")
    if "positive_label" in normalized.columns:
        recorded_positive = _strict_integer_series(normalized, "positive_label")
        if not (recorded_positive == positive_label).all():
            raise ValueError("Recorded positive_label values do not match the aggregation setting")
        normalized["positive_label"] = recorded_positive
    if "predicted_class_label" in normalized.columns:
        normalized["predicted_class_label"] = _strict_integer_series(
            normalized, "predicted_class_label"
        )
        if not set(normalized["predicted_class_label"].unique()).issubset({0, 1}):
            raise ValueError("predicted_class_label must contain only 0 and 1")

    rows: list[dict[str, object]] = []
    for subject_id, group in normalized.groupby("subject_id", sort=True):
        ordered = group.sort_values("slice_offset")
        if ordered["y_true"].nunique() != 1:
            raise ValueError("A subject has inconsistent y_true values")
        if ordered["center_slice_index"].nunique() != 1:
            raise ValueError("A subject has inconsistent center_slice_index values")
        observed_offsets = tuple(ordered["slice_offset"].astype(int).tolist())
        if observed_offsets != expected_offsets:
            raise ValueError(
                f"A subject must have offsets {sorted(expected_offsets)}; "
                f"found {list(observed_offsets)}"
            )
        center_slice_index = int(ordered["center_slice_index"].iloc[0])
        expected_indices = center_slice_index + ordered["slice_offset"].astype(int)
        if not (ordered["slice_index"].astype(int).to_numpy() == expected_indices.to_numpy()).all():
            raise ValueError("A subject has slice_index values inconsistent with its center")

        scores = ordered["p_positive"].astype(float).to_numpy()
        slice_votes = ordered["is_positive_prediction"].to_numpy(dtype=int)
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
                "center_slice_index": center_slice_index,
                "patient_score": patient_score,
                "patient_prediction": patient_prediction,
                "is_positive_prediction": is_positive_prediction,
                "positive_label": positive_label,
                "soft_score": soft_score,
                "positive_vote_fraction": vote_fraction,
                "n_positive_votes": int(slice_votes.sum()),
                "n_slices": expected_slices,
                "slice_indices": json.dumps(ordered["slice_index"].astype(int).tolist()),
                "slice_offsets": json.dumps(ordered["slice_offset"].astype(int).tolist()),
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
