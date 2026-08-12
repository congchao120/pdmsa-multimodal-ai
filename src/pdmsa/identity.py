from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd


_ID_CLEANER = re.compile(r"[^A-Z0-9]+")


def normalize_observed_id(value: object) -> str | None:
    """Conservatively normalize an observed ID without inventing an identity."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    cleaned = _ID_CLEANER.sub("", text)
    return cleaned or None


def stable_record_key(source_file: str, source_row: int, path: str, label: object) -> str:
    payload = f"{Path(source_file).as_posix()}|{source_row}|{path}|{label}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def reconcile_records(
    records: pd.DataFrame,
    observed_id_column: str = "observed_id",
    label_column: str = "label",
    fingerprint_column: str = "record_fingerprint",
) -> pd.DataFrame:
    """
    Classify deterministic duplicate records, ID/label candidates, conflicts, and unresolved rows.

    `candidate_exact_id_label` is deliberately not called a verified identity. Manual or source
    image confirmation is still required before cross-experiment merging.
    """
    required = {observed_id_column, label_column, fingerprint_column}
    missing = required.difference(records.columns)
    if missing:
        raise ValueError(f"Records are missing columns: {sorted(missing)}")

    result = records.copy()
    result["normalized_observed_id"] = result[observed_id_column].map(normalize_observed_id)
    result["provisional_research_id"] = ""
    result["match_status"] = "unresolved_no_id"
    result["identity_note"] = "No deterministic identity evidence; do not merge by diagnosis."

    fingerprint_counts = result[fingerprint_column].value_counts(dropna=False)
    duplicate_fingerprints = set(fingerprint_counts[fingerprint_counts > 1].index)

    label_sets = (
        result.dropna(subset=["normalized_observed_id"])
        .groupby("normalized_observed_id")[label_column]
        .agg(lambda values: set(str(v) for v in values))
    )
    conflicting_ids = set(label_sets[label_sets.map(len) > 1].index)

    for index, record in result.iterrows():
        fingerprint = record[fingerprint_column]
        normalized = record["normalized_observed_id"]
        if fingerprint in duplicate_fingerprints:
            result.at[index, "provisional_research_id"] = f"REC-{str(fingerprint)[:12]}"
            result.at[index, "match_status"] = "deterministic_same_record"
            result.at[index, "identity_note"] = "Exact record fingerprint recurs across sources."
        elif normalized in conflicting_ids:
            result.at[index, "provisional_research_id"] = f"CONFLICT-{normalized}"
            result.at[index, "match_status"] = "conflicting_label_for_id"
            result.at[index, "identity_note"] = (
                "Same observed ID is associated with different labels."
            )
        elif normalized:
            label = str(record[label_column]).strip()
            result.at[index, "provisional_research_id"] = f"CAND-{label}-{normalized}"
            result.at[index, "match_status"] = "candidate_exact_id_label"
            result.at[index, "identity_note"] = (
                "Exact normalized ID and label agree; verify against original image/export "
                "before merging."
            )
        else:
            digest = hashlib.sha256(str(fingerprint).encode("utf-8")).hexdigest()[:12]
            result.at[index, "provisional_research_id"] = f"UNRESOLVED-{digest}"
    return result
