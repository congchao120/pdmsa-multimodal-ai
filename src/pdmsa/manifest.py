from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_BASE_COLUMNS = ("subject_id", "label", "slice_index")


@dataclass(frozen=True)
class ManifestAudit:
    rows: int
    subjects: int
    class_counts: dict[int, int]
    slices_per_subject: dict[int, int]
    channels: tuple[str, ...]


def load_manifest(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"subject_id": "string"})
    if "subject_id" in frame:
        frame["subject_id"] = frame["subject_id"].str.strip()
    return frame


def validate_manifest(
    frame: pd.DataFrame,
    channels: Iterable[str],
    expected_slices: int | None = 5,
    expected_slice_indices: Iterable[int] | None = None,
    data_root: str | Path | None = None,
    check_files: bool = False,
) -> ManifestAudit:
    """Validate identity, label, slice, and channel invariants before splitting."""
    channel_tuple = tuple(channels)
    required = set(REQUIRED_BASE_COLUMNS).union(channel_tuple)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")

    if frame[list(required)].isna().any().any():
        null_columns = frame[list(required)].columns[frame[list(required)].isna().any()].tolist()
        raise ValueError(f"Required manifest fields contain missing values: {null_columns}")

    if (frame["subject_id"].astype(str).str.strip() == "").any():
        raise ValueError("subject_id contains blank values")

    numeric_labels = pd.to_numeric(frame["label"], errors="raise")
    if not np.isfinite(numeric_labels.to_numpy(dtype=float)).all():
        raise ValueError("label contains a non-finite value")
    if not (numeric_labels.to_numpy(dtype=float) == np.floor(numeric_labels)).all():
        raise ValueError("label values must be integers; fractional values are not allowed")
    labels = numeric_labels.astype(int)
    if not set(labels.unique()).issubset({0, 1}):
        raise ValueError("This binary pipeline requires labels encoded as 0 and 1")

    numeric_slices = pd.to_numeric(frame["slice_index"], errors="raise")
    if not np.isfinite(numeric_slices.to_numpy(dtype=float)).all():
        raise ValueError("slice_index contains a non-finite value")
    if not (numeric_slices.to_numpy(dtype=float) == np.floor(numeric_slices)).all():
        raise ValueError("slice_index values must be integers; fractional values are not allowed")
    normalized = frame.copy()
    normalized["label"] = labels
    normalized["slice_index"] = numeric_slices.astype(int)

    label_counts_per_subject = normalized.groupby("subject_id")["label"].nunique()
    inconsistent = label_counts_per_subject[label_counts_per_subject != 1].index.tolist()
    if inconsistent:
        raise ValueError(f"Subjects with inconsistent labels: {inconsistent[:10]}")

    duplicate_key = normalized.duplicated(["subject_id", "slice_index"], keep=False)
    if duplicate_key.any():
        examples = normalized.loc[duplicate_key, ["subject_id", "slice_index"]].head(10)
        raise ValueError(
            f"Duplicate subject/slice rows detected:\n{examples.to_string(index=False)}"
        )

    for channel in channel_tuple:
        if (frame[channel].astype(str).str.strip() == "").any():
            raise ValueError(f"Path column {channel} contains blank values")
        path_subject_counts = (
            frame.assign(_path=frame[channel].astype(str).str.strip())
            .groupby("_path")["subject_id"]
            .nunique()
        )
        reused = path_subject_counts[path_subject_counts > 1]
        if not reused.empty:
            raise ValueError(
                f"Channel {channel} reuses a file path across subjects; examples: "
                f"{reused.head(10).index.tolist()}"
            )

    slices = normalized.groupby("subject_id")["slice_index"].nunique()
    if expected_slices is not None:
        bad = slices[slices != expected_slices]
        if not bad.empty:
            raise ValueError(
                f"Expected {expected_slices} unique slices per subject; mismatches: "
                f"{bad.head(10).to_dict()}"
            )

    if expected_slice_indices is not None:
        expected = tuple(int(value) for value in expected_slice_indices)
        if len(expected) != len(set(expected)):
            raise ValueError("expected_slice_indices contains duplicates")
        if expected_slices is not None and len(expected) != expected_slices:
            raise ValueError(
                "expected_slice_indices length does not match expected_slices: "
                f"{len(expected)} != {expected_slices}"
            )
        expected_set = set(expected)
        bad_indices: dict[str, list[int]] = {}
        for subject_id, group in normalized.groupby("subject_id", sort=False):
            observed = set(group["slice_index"].astype(int))
            if observed != expected_set:
                bad_indices[str(subject_id)] = sorted(observed)
                if len(bad_indices) >= 10:
                    break
        if bad_indices:
            raise ValueError(
                f"Expected slice indices {sorted(expected_set)} for every subject; "
                f"mismatches: {bad_indices}"
            )

    if check_files:
        root = Path(data_root or ".").expanduser().resolve()
        missing_paths: list[str] = []
        for channel in channel_tuple:
            for value in frame[channel].astype(str):
                path = Path(value).expanduser()
                if not path.is_absolute():
                    path = root / path
                if not path.is_file():
                    missing_paths.append(str(path))
                    if len(missing_paths) >= 20:
                        break
            if len(missing_paths) >= 20:
                break
        if missing_paths:
            raise FileNotFoundError(
                "Missing modality files (first 20):\n" + "\n".join(missing_paths)
            )

    subject_labels = normalized.drop_duplicates("subject_id")
    class_counts = {
        int(key): int(value) for key, value in subject_labels["label"].value_counts().items()
    }
    slice_distribution = {int(k): int(v) for k, v in slices.value_counts().items()}
    return ManifestAudit(
        rows=len(frame),
        subjects=frame["subject_id"].nunique(),
        class_counts=class_counts,
        slices_per_subject=slice_distribution,
        channels=channel_tuple,
    )


def subject_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one label-consistent row per subject."""
    validate_manifest(frame, channels=(), expected_slices=None)
    return (
        frame[["subject_id", "label"]]
        .assign(label=lambda x: pd.to_numeric(x["label"]).astype(int))
        .drop_duplicates("subject_id")
        .sort_values("subject_id")
        .reset_index(drop=True)
    )
