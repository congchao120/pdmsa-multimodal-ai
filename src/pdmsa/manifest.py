from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SLICE_OFFSETS = (-2, -1, 0, 1, 2)
REQUIRED_BASE_COLUMNS = (
    "subject_id",
    "label",
    "center_slice_index",
    "slice_offset",
    "slice_index",
)


@dataclass(frozen=True)
class ManifestAudit:
    rows: int
    subjects: int
    class_counts: dict[int, int]
    slices_per_subject: dict[int, int]
    slice_offsets: tuple[int, ...]
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
    expected_slice_offsets: Iterable[int] | None = DEFAULT_SLICE_OFFSETS,
    data_root: str | Path | None = None,
    check_files: bool = False,
) -> ManifestAudit:
    """Validate identity, label, centered-slice, and channel invariants.

    Slice indices are zero-based array indices. Every subject records an independently
    selected ``center_slice_index`` and five offsets around that center; file names are
    opaque and are never used to infer an index.
    """
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
    numeric_centers = pd.to_numeric(frame["center_slice_index"], errors="raise")
    numeric_offsets = pd.to_numeric(frame["slice_offset"], errors="raise")
    for name, values in (
        ("slice_index", numeric_slices),
        ("center_slice_index", numeric_centers),
        ("slice_offset", numeric_offsets),
    ):
        array = values.to_numpy(dtype=float)
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains a non-finite value")
        if not (array == np.floor(array)).all():
            raise ValueError(f"{name} values must be integers; fractional values are not allowed")
    normalized = frame.copy()
    normalized["label"] = labels
    normalized["slice_index"] = numeric_slices.astype(int)
    normalized["center_slice_index"] = numeric_centers.astype(int)
    normalized["slice_offset"] = numeric_offsets.astype(int)
    if (normalized[["slice_index", "center_slice_index"]] < 0).any().any():
        raise ValueError(
            "slice_index and center_slice_index must be non-negative zero-based indices"
        )

    label_counts_per_subject = normalized.groupby("subject_id")["label"].nunique()
    inconsistent = label_counts_per_subject[label_counts_per_subject != 1].index.tolist()
    if inconsistent:
        raise ValueError(f"Found {len(inconsistent)} subjects with inconsistent labels")

    center_counts_per_subject = normalized.groupby("subject_id")["center_slice_index"].nunique()
    inconsistent_centers = center_counts_per_subject[center_counts_per_subject != 1].index.tolist()
    if inconsistent_centers:
        raise ValueError(
            "Found "
            f"{len(inconsistent_centers)} subjects with inconsistent center_slice_index values"
        )

    expected_absolute = normalized["center_slice_index"] + normalized["slice_offset"]
    inconsistent_mapping = normalized["slice_index"] != expected_absolute
    if inconsistent_mapping.any():
        raise ValueError(
            "slice_index must equal center_slice_index + slice_offset; "
            f"found {int(inconsistent_mapping.sum())} mismatched rows"
        )

    duplicate_key = normalized.duplicated(["subject_id", "slice_offset"], keep=False)
    if duplicate_key.any():
        raise ValueError(
            f"Duplicate subject/slice-offset rows detected: {int(duplicate_key.sum())} rows"
        )

    for channel in channel_tuple:
        if (frame[channel].astype(str).str.strip() == "").any():
            raise ValueError(f"Path column {channel} contains blank values")
        normalized_paths = frame[channel].astype(str).str.strip()
        reused = normalized_paths[normalized_paths.duplicated(keep=False)]
        if not reused.empty:
            raise ValueError(
                f"Channel {channel} reuses a file path across {len(reused)} manifest rows"
            )

    slices = normalized.groupby("subject_id")["slice_offset"].nunique()
    if expected_slices is not None:
        bad = slices[slices != expected_slices]
        if not bad.empty:
            raise ValueError(
                f"Expected {expected_slices} unique slices per subject; "
                f"found {len(bad)} subjects with a different count"
            )

    expected: tuple[int, ...] = tuple()
    if expected_slice_offsets is not None:
        expected = tuple(int(value) for value in expected_slice_offsets)
        if len(expected) != len(set(expected)):
            raise ValueError("expected_slice_offsets contains duplicates")
        if expected_slices is not None and len(expected) != expected_slices:
            raise ValueError(
                "expected_slice_offsets length does not match expected_slices: "
                f"{len(expected)} != {expected_slices}"
            )
        expected_set = set(expected)
        bad_offset_count = 0
        for _, group in normalized.groupby("subject_id", sort=False):
            observed = set(group["slice_offset"].astype(int))
            if observed != expected_set:
                bad_offset_count += 1
        if bad_offset_count:
            raise ValueError(
                f"Expected slice offsets {sorted(expected_set)} for every subject; "
                f"found {bad_offset_count} subjects with a different offset set"
            )

    if check_files:
        root = Path(data_root or ".").expanduser().resolve()
        missing_path_count = 0
        for channel in channel_tuple:
            for value in frame[channel].astype(str):
                path = Path(value).expanduser()
                if not path.is_absolute():
                    path = root / path
                if not path.is_file():
                    missing_path_count += 1
        if missing_path_count:
            raise FileNotFoundError(f"Missing {missing_path_count} configured image files")

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
        slice_offsets=expected,
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
