from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .manifest import subject_table


@dataclass(frozen=True)
class SplitAudit:
    subjects: int
    folds: int
    validation_appearances_min: int
    validation_appearances_max: int
    role_counts: dict[str, int]


def _assign_stratified_folds(subjects: pd.DataFrame, n_folds: int, seed: int) -> pd.Series:
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    assignments = pd.Series(index=subjects.index, dtype="int64")
    rng = np.random.default_rng(seed)
    for label, group in subjects.groupby("label", sort=True):
        indices = group.index.to_numpy(copy=True)
        if len(indices) < n_folds:
            raise ValueError(
                f"Class {label} has {len(indices)} subjects, fewer than n_folds={n_folds}"
            )
        rng.shuffle(indices)
        for fold, fold_indices in enumerate(np.array_split(indices, n_folds)):
            assignments.loc[fold_indices] = fold
    return assignments.astype(int)


def make_four_fold_assignments(
    manifest: pd.DataFrame,
    n_folds: int = 4,
    seed: int = 42,
    stratified: bool = True,
) -> pd.DataFrame:
    """Create frozen subject-level train/validation assignments.

    Each subject occurs in validation exactly once. Stratification is enabled by
    default for the imbalanced cohort. Setting it to false retains subject-level
    separation but creates an unstratified random partition.
    """
    subjects = subject_table(manifest)
    if stratified:
        subjects["validation_fold"] = _assign_stratified_folds(subjects, n_folds, seed)
    else:
        if len(subjects) < n_folds:
            raise ValueError("The number of subjects must be at least n_folds")
        rng = np.random.default_rng(seed)
        shuffled = subjects.index.to_numpy(copy=True)
        rng.shuffle(shuffled)
        fold_ids = pd.Series(index=subjects.index, dtype="int64")
        for fold, indices in enumerate(np.array_split(shuffled, n_folds)):
            fold_ids.loc[indices] = fold
        subjects["validation_fold"] = fold_ids.astype(int)

    rows: list[dict[str, object]] = []
    for fold in range(n_folds):
        for record in subjects.itertuples(index=False):
            rows.append(
                {
                    "fold": fold,
                    "subject_id": str(record.subject_id),
                    "label": int(record.label),
                    "role": "validation" if int(record.validation_fold) == fold else "train",
                    "seed": int(seed),
                    "stratified": bool(stratified),
                }
            )
    result = pd.DataFrame(rows).sort_values(["fold", "role", "subject_id"])
    audit_four_fold_assignments(
        result,
        expected_subjects=len(subjects),
        expected_folds=n_folds,
    )
    return result.reset_index(drop=True)


def audit_four_fold_assignments(
    assignments: pd.DataFrame,
    expected_subjects: int | None = None,
    expected_folds: int | None = None,
) -> SplitAudit:
    """Check complete coverage, label consistency, and train/validation separation."""
    required = {"fold", "subject_id", "label", "role"}
    missing = required.difference(assignments.columns)
    if missing:
        raise ValueError(f"Assignments are missing columns: {sorted(missing)}")
    if assignments.duplicated(["fold", "subject_id"]).any():
        raise ValueError("A subject appears more than once within a fold")
    if set(assignments["role"]) != {"train", "validation"}:
        raise ValueError("Assignments must contain train and validation roles only")
    if (assignments.groupby("subject_id")["label"].nunique() != 1).any():
        raise ValueError("A subject has inconsistent labels across folds")

    folds = int(assignments["fold"].nunique())
    subjects = int(assignments["subject_id"].nunique())
    if expected_subjects is not None and subjects != expected_subjects:
        raise ValueError(f"Expected {expected_subjects} subjects, found {subjects}")
    if expected_folds is not None and folds != expected_folds:
        raise ValueError(f"Expected {expected_folds} folds, found {folds}")
    observed_folds = set(pd.to_numeric(assignments["fold"], errors="raise").astype(int))
    if observed_folds != set(range(folds)):
        raise ValueError(
            f"Fold identifiers must be contiguous from zero; found {sorted(observed_folds)}"
        )

    appearances = assignments.loc[assignments["role"] == "validation"].groupby("subject_id").size()
    if len(appearances) != subjects or not (appearances == 1).all():
        raise ValueError("Every subject must appear in validation exactly once")
    for fold, group in assignments.groupby("fold"):
        for role in ("train", "validation"):
            subset = group.loc[group["role"] == role]
            if subset.empty:
                raise ValueError(f"Fold {fold} has no subjects assigned to {role}")
            if subset["label"].nunique() != 2:
                raise ValueError(f"Fold {fold}, role {role} does not contain both classes")
        train_ids = set(group.loc[group["role"] == "train", "subject_id"].astype(str))
        validation_ids = set(group.loc[group["role"] == "validation", "subject_id"].astype(str))
        if train_ids.intersection(validation_ids):
            raise ValueError(f"Fold {fold} contains train/validation subject leakage")

    return SplitAudit(
        subjects=subjects,
        folds=folds,
        validation_appearances_min=int(appearances.min()),
        validation_appearances_max=int(appearances.max()),
        role_counts={
            str(key): int(value) for key, value in assignments["role"].value_counts().items()
        },
    )


def audit_manifest_assignment_alignment(
    manifest: pd.DataFrame,
    assignments: pd.DataFrame,
    expected_folds: int | None = None,
) -> SplitAudit:
    """Require exact subject and label agreement with frozen fold assignments."""
    subjects = subject_table(manifest)
    audit = audit_four_fold_assignments(
        assignments,
        expected_subjects=len(subjects),
        expected_folds=expected_folds,
    )
    manifest_ids = set(subjects["subject_id"].astype(str))
    assignment_ids = set(assignments["subject_id"].astype(str))
    if manifest_ids != assignment_ids:
        missing = sorted(manifest_ids.difference(assignment_ids))
        extra = sorted(assignment_ids.difference(manifest_ids))
        raise ValueError(
            "Manifest/assignment subject mismatch: "
            f"missing_from_assignments={len(missing)}, extra_in_assignments={len(extra)}"
        )

    manifest_labels = (
        subjects.assign(subject_id=subjects["subject_id"].astype(str))
        .set_index("subject_id")["label"]
        .astype(int)
    )
    assignment_labels = (
        assignments.assign(subject_id=assignments["subject_id"].astype(str))
        .drop_duplicates("subject_id")
        .set_index("subject_id")["label"]
        .astype(int)
    )
    mismatched = manifest_labels[
        manifest_labels != assignment_labels.reindex(manifest_labels.index)
    ]
    if not mismatched.empty:
        raise ValueError(f"Manifest/assignment label mismatch for {len(mismatched)} subjects")
    return audit
