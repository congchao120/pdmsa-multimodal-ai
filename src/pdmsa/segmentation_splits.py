"""Deterministic nnU-Net cross-validation split generation.

The functions in this module operate on case identifiers only.  They never
open images or labels, which keeps split preparation lightweight and avoids
copying subject data into logs or repository artifacts.
"""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable, Mapping, Sequence

import numpy as np
from sklearn.model_selection import KFold


Split = dict[str, list[str]]


def _normalise_case_ids(case_ids: Iterable[str]) -> list[str]:
    if isinstance(case_ids, (str, bytes)):
        raise TypeError("case_ids must be an iterable of identifiers, not one string")

    normalised: list[str] = []
    for raw_case_id in case_ids:
        case_id = str(raw_case_id).strip()
        if not case_id:
            raise ValueError("Case identifiers must not be empty")
        if case_id in {".", ".."} or "/" in case_id or "\\" in case_id:
            raise ValueError("A case identifier is not a safe filename stem")
        normalised.append(case_id)

    if not normalised:
        raise ValueError("No case identifiers were found")
    duplicates = sorted(case_id for case_id, count in Counter(normalised).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate case identifiers found: {len(duplicates)}")
    return sorted(normalised)


def discover_case_ids(labels_dir: str | Path, file_ending: str = ".nii.gz") -> list[str]:
    """Discover case IDs from label filenames without opening label contents.

    ``file_ending`` must match the ``file_ending`` field in nnU-Net's
    ``dataset.json``.  Compound endings such as ``.nii.gz`` are stripped as a
    single unit.
    """

    directory = Path(labels_dir)
    if not directory.is_dir():
        raise NotADirectoryError(f"labelsTr directory not found: {directory}")
    if not file_ending or not file_ending.startswith("."):
        raise ValueError("file_ending must start with '.', for example '.nii.gz'")

    case_ids = [
        path.name[: -len(file_ending)]
        for path in directory.iterdir()
        if path.is_file() and path.name.endswith(file_ending)
    ]
    return _normalise_case_ids(case_ids)


def read_case_ids(case_list: str | Path) -> list[str]:
    """Read one case ID per line from a UTF-8 text file.

    Empty lines and lines whose first non-whitespace character is ``#`` are
    ignored.  Case IDs themselves are never printed by this module.
    """

    path = Path(case_list)
    if not path.is_file():
        raise FileNotFoundError(f"Case-list file not found: {path}")
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    case_ids = [
        line.strip()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return _normalise_case_ids(case_ids)


def make_kfold_splits(
    case_ids: Iterable[str],
    n_splits: int = 4,
    seed: int = 12345,
) -> list[Split]:
    """Create balanced, deterministic nnU-Net ``train``/``val`` splits.

    This mirrors nnU-Net v2's retained ``do_split`` implementation: case keys
    are sorted with NumPy and passed to scikit-learn ``KFold`` with shuffling
    and a fixed random state.  The only study-specific change is four folds
    instead of nnU-Net's usual five.
    """

    cases = _normalise_case_ids(case_ids)
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if len(cases) < n_splits:
        raise ValueError(
            f"At least n_splits={n_splits} cases are required; found {len(cases)}"
        )

    sorted_keys = np.sort(np.asarray(cases, dtype=str))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits: list[Split] = []
    for train_indices, validation_indices in splitter.split(sorted_keys):
        splits.append(
            {
                "train": sorted_keys[train_indices].tolist(),
                "val": sorted_keys[validation_indices].tolist(),
            }
        )

    validate_splits(splits, cases, n_splits=n_splits)
    return splits


def validate_splits(
    splits: Sequence[Mapping[str, Sequence[str]]],
    case_ids: Iterable[str],
    n_splits: int | None = None,
) -> list[int]:
    """Validate exact coverage and return validation-set sizes by fold."""

    cases = _normalise_case_ids(case_ids)
    expected = set(cases)
    expected_folds = len(splits) if n_splits is None else n_splits
    if expected_folds < 2:
        raise ValueError("n_splits must be at least 2")
    if len(splits) != expected_folds:
        raise ValueError(f"Expected {expected_folds} folds, found {len(splits)}")

    validation_counts: Counter[str] = Counter()
    validation_sizes: list[int] = []
    for fold, split in enumerate(splits):
        if set(split) != {"train", "val"}:
            raise ValueError(f"Fold {fold} must contain exactly 'train' and 'val' keys")
        train = _normalise_case_ids(split["train"])
        validation = _normalise_case_ids(split["val"])
        train_ids = set(train)
        validation_ids = set(validation)

        overlap = train_ids.intersection(validation_ids)
        if overlap:
            raise ValueError(f"Fold {fold} contains train/validation overlap")
        if train_ids.union(validation_ids) != expected:
            raise ValueError(f"Fold {fold} does not contain the exact expected case set")
        if train_ids != expected.difference(validation_ids):
            raise ValueError(f"Fold {fold} training set is not the complement of validation")

        validation_counts.update(validation)
        validation_sizes.append(len(validation))

    if set(validation_counts) != expected or any(
        validation_counts[case_id] != 1 for case_id in cases
    ):
        raise ValueError("Every case must appear in validation exactly once")
    if max(validation_sizes) - min(validation_sizes) > 1:
        raise ValueError("Validation fold sizes differ by more than one")
    return validation_sizes


def write_splits_final(
    output_path: str | Path,
    splits: Sequence[Mapping[str, Sequence[str]]],
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write nnU-Net's ``splits_final.json`` representation.

    The completed JSON is first flushed to a temporary file in the destination
    directory.  Without ``overwrite``, a same-filesystem hard-link installs it
    only if the destination does not exist, avoiding a check/write race.
    """

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing split file: {destination}. "
            "Pass overwrite=True only after reviewing it."
        )

    serialisable = [
        {"train": list(split["train"]), "val": list(split["val"])} for split in splits
    ]
    payload = json.dumps(serialisable, ensure_ascii=False, indent=2) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

        if overwrite:
            os.replace(temporary, destination)
        else:
            try:
                os.link(temporary, destination)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"Refusing to overwrite existing split file: {destination}"
                ) from exc
            finally:
                temporary.unlink(missing_ok=True)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
