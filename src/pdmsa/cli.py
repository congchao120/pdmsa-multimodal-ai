from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import load_config, resolve_config_path
from .manifest import load_manifest, validate_manifest
from .metrics import bootstrap_confidence_intervals, classification_metrics
from .splits import make_four_fold_assignments
from .training import train_fold
from .voting import aggregate_slice_predictions


def _config_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path to a TOML configuration")
    return parser


def _gradcam_parser() -> argparse.ArgumentParser:
    parser = _config_parser("Generate Grad-CAM artifacts for one pre-fused RGB slice")
    parser.add_argument("--checkpoint", required=True, help="ViT state-dict checkpoint (.pth)")
    parser.add_argument("--input", required=True, help="Pre-fused RGB input image")
    parser.add_argument("--output-dir", required=True, help="Directory for CAM artifacts")
    parser.add_argument(
        "--target-class",
        type=int,
        choices=[0, 1],
        help="Class to explain; defaults to [gradcam].target_class in the configuration",
    )
    parser.add_argument(
        "--target-layer",
        type=int,
        help="ViT encoder-layer index; defaults to [gradcam].target_layer_index",
    )
    return parser


def gradcam_main() -> None:
    args = _gradcam_parser().parse_args()
    from .gradcam import create_gradcam_artifacts

    output = create_gradcam_artifacts(
        config=load_config(args.config),
        checkpoint=args.checkpoint,
        input_image=args.input,
        output_dir=args.output_dir,
        target_class=args.target_class,
        target_layer_index=args.target_layer,
    )
    print(f"Grad-CAM artifacts saved to {output}")


def audit_manifest_main() -> None:
    parser = _config_parser("Validate a PD/MSA-P classification manifest")
    parser.add_argument("--check-files", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    data = config["data"]
    manifest = load_manifest(resolve_config_path(config, data["manifest"]))
    audit = validate_manifest(
        manifest,
        [str(data.get("image_column", "fused_path"))],
        expected_slices=int(data.get("expected_slices", 5)),
        expected_slice_indices=data.get("slice_indices", [6, 7, 8, 9, 10]),
        data_root=resolve_config_path(config, data.get("root", ".")),
        check_files=args.check_files,
    )
    print(json.dumps(audit.__dict__, indent=2))


def make_splits_main() -> None:
    parser = _config_parser(
        "Create frozen patient-level four-fold train/validation assignments"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    data = config["data"]
    split_config = config["splits"]
    manifest = load_manifest(resolve_config_path(config, data["manifest"]))
    validate_manifest(
        manifest,
        [str(data.get("image_column", "fused_path"))],
        expected_slices=int(data.get("expected_slices", 5)),
        expected_slice_indices=data.get("slice_indices", [6, 7, 8, 9, 10]),
        check_files=False,
    )
    assignments = make_four_fold_assignments(
        manifest,
        n_folds=int(split_config.get("n_folds", 4)),
        seed=int(config.get("seed", 42)),
        stratified=bool(split_config.get("stratified", True)),
    )
    output = resolve_config_path(config, split_config["assignments_file"])
    output.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(output, index=False)
    print(f"Saved {len(assignments)} assignment rows to {output}")


def train_fold_main() -> None:
    parser = _config_parser("Train five slice-specific ViTs and MSV for one fold")
    parser.add_argument("--fold", required=True, type=int)
    args = parser.parse_args()
    output = train_fold(load_config(args.config), args.fold)
    print(f"Fold outputs saved to {output}")


def _prepare_slice_frame(frame: pd.DataFrame, positive_label: int) -> pd.DataFrame:
    frame = frame.copy()
    positive_score_column = f"p_class_{positive_label}"
    if positive_score_column in frame.columns:
        frame["p_positive"] = frame[positive_score_column]
    elif "p_positive" not in frame.columns:
        raise ValueError(
            f"Input requires either {positive_score_column} or an explicitly defined p_positive"
        )
    if "predicted_class_label" not in frame.columns:
        if "slice_prediction" in frame.columns:
            frame["predicted_class_label"] = frame["slice_prediction"]
        elif {"p_class_0", "p_class_1"}.issubset(frame.columns):
            frame["predicted_class_label"] = (
                frame[["p_class_0", "p_class_1"]].to_numpy().argmax(axis=1)
            )
        else:
            raise ValueError("Input requires predicted_class_label or both class-score columns")
    frame["is_positive_prediction"] = (
        frame["predicted_class_label"].astype(int) == positive_label
    ).astype(int)
    return frame


def aggregate_msv_main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate five slice scores per subject")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-slices", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5, help="Patient-level threshold")
    parser.add_argument(
        "--method", choices=["soft", "hard", "weighted_soft"], default="weighted_soft"
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=[0.05, 0.05, 0.8, 0.05, 0.05],
        help="Fixed weights ordered by ascending slice_index",
    )
    parser.add_argument(
        "--threshold-inclusive",
        action="store_true",
        help="Use >= threshold (source scripts used strict > threshold)",
    )
    parser.add_argument("--positive-label", type=int, choices=[0, 1], default=1)
    args = parser.parse_args()
    frame = _prepare_slice_frame(
        pd.read_csv(args.input, dtype={"subject_id": "string"}), args.positive_label
    )
    output_frame = aggregate_slice_predictions(
        frame,
        expected_slices=args.expected_slices,
        patient_threshold=args.threshold,
        method=args.method,
        positive_label=args.positive_label,
        weights=args.weights if args.method == "weighted_soft" else None,
        threshold_inclusive=args.threshold_inclusive,
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(output, index=False)
    print(f"Saved {len(output_frame)} subject predictions to {output}")


def evaluate_oof_main() -> None:
    """Pool held-out fold predictions and compute patient-level OOF metrics once."""
    parser = argparse.ArgumentParser(description="Evaluate pooled out-of-fold slice predictions")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-slices", type=int, default=5)
    parser.add_argument("--expected-subjects", type=int)
    parser.add_argument("--patient-threshold", type=float, default=0.5)
    parser.add_argument(
        "--method", choices=["soft", "hard", "weighted_soft"], default="weighted_soft"
    )
    parser.add_argument(
        "--weights", type=float, nargs="+", default=[0.05, 0.05, 0.8, 0.05, 0.05]
    )
    parser.add_argument("--threshold-inclusive", action="store_true")
    parser.add_argument("--positive-label", type=int, choices=[0, 1], default=1)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = _prepare_slice_frame(
        pd.read_csv(args.input, dtype={"subject_id": "string"}), args.positive_label
    )
    fold_column = "fold" if "fold" in frame.columns else "outer_fold"
    if fold_column not in frame.columns:
        raise ValueError("Pooled OOF input must contain fold")
    folds_per_subject = frame.groupby("subject_id")[fold_column].nunique()
    if not (folds_per_subject == 1).all():
        raise ValueError("A subject has predictions from more than one held-out fold")

    subject_predictions = aggregate_slice_predictions(
        frame,
        expected_slices=args.expected_slices,
        patient_threshold=args.patient_threshold,
        method=args.method,
        positive_label=args.positive_label,
        weights=args.weights if args.method == "weighted_soft" else None,
        threshold_inclusive=args.threshold_inclusive,
    )
    fold_map = frame.groupby("subject_id")[fold_column].first().astype(int)
    subject_predictions.insert(
        1,
        "fold",
        subject_predictions["subject_id"].map(fold_map).astype(int),
    )
    if args.expected_subjects is not None and len(subject_predictions) != args.expected_subjects:
        raise ValueError(
            f"Expected {args.expected_subjects} OOF subjects, found {len(subject_predictions)}"
        )

    metrics = classification_metrics(
        subject_predictions["y_true"],
        subject_predictions["patient_score"],
        threshold=args.patient_threshold,
        positive_label=args.positive_label,
        threshold_inclusive=args.threshold_inclusive,
    )
    intervals = bootstrap_confidence_intervals(
        subject_predictions["y_true"],
        subject_predictions["patient_score"],
        threshold=args.patient_threshold,
        positive_label=args.positive_label,
        repetitions=args.bootstrap_repetitions,
        seed=args.seed,
        threshold_inclusive=args.threshold_inclusive,
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    subject_predictions.to_csv(output_dir / "oof_subject_predictions.csv", index=False)
    with (output_dir / "oof_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({"point_estimates": metrics, "bootstrap_ci": intervals}, handle, indent=2)
    print(f"Saved pooled OOF results for {len(subject_predictions)} subjects to {output_dir}")
