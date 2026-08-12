from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def class_weight_vector(
    subject_labels: Iterable[int], classes: tuple[int, ...] = (0, 1)
) -> np.ndarray:
    """Inverse subject-frequency weights normalized to a mean of one."""
    labels = np.asarray(list(subject_labels), dtype=int)
    counts = np.asarray([(labels == value).sum() for value in classes], dtype=float)
    if (counts <= 0).any():
        raise ValueError(f"Every class must be represented; counts={counts.tolist()}")
    weights = labels.size / (len(classes) * counts)
    return weights / weights.mean()


def row_sampling_weights(frame: pd.DataFrame) -> np.ndarray:
    """Balance classes at subject level without favoring subjects with more slices."""
    required = {"subject_id", "label"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Frame must contain {sorted(required)}")

    normalized = frame[["subject_id", "label"]].copy()
    normalized["subject_id"] = normalized["subject_id"].astype(str)
    normalized["label"] = pd.to_numeric(normalized["label"], errors="raise").astype(int)
    per_subject_labels = normalized.drop_duplicates()
    if per_subject_labels.groupby("subject_id")["label"].nunique().max() != 1:
        raise ValueError("A subject has more than one label")

    class_subject_counts = per_subject_labels["label"].value_counts().to_dict()
    subject_row_counts = normalized["subject_id"].value_counts().to_dict()
    weights = []
    for record in normalized.itertuples(index=False):
        class_count = class_subject_counts[int(record.label)]
        slice_count = subject_row_counts[str(record.subject_id)]
        weights.append(1.0 / (class_count * slice_count))
    return np.asarray(weights, dtype=float)


def build_weighted_sampler(frame: pd.DataFrame):
    """Create a PyTorch sampler lazily so data-audit utilities remain lightweight."""
    import torch

    weights = torch.as_tensor(row_sampling_weights(frame), dtype=torch.double)
    return torch.utils.data.WeightedRandomSampler(
        weights=weights, num_samples=len(weights), replacement=True
    )
