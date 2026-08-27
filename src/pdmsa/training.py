from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .config import public_config, resolve_config_path
from .data import FusedRGBSliceDataset, build_transforms
from .imbalance import build_weighted_sampler, class_weight_vector
from .manifest import DEFAULT_SLICE_OFFSETS, load_manifest, validate_manifest
from .metrics import bootstrap_confidence_intervals, classification_metrics
from .models import create_vit
from .reproducibility import file_sha256, seed_everything, seed_worker
from .splits import audit_manifest_assignment_alignment
from .voting import DEFAULT_MSV_WEIGHTS, aggregate_slice_predictions

SHARED_CHECKPOINT_NAME = "best_model_weights.pth"


def _subset_rows(manifest: pd.DataFrame, assignments: pd.DataFrame, fold: int, role: str):
    ids = set(
        assignments.loc[
            (assignments["fold"].astype(int) == fold) & (assignments["role"] == role),
            "subject_id",
        ].astype(str)
    )
    return manifest.loc[manifest["subject_id"].astype(str).isin(ids)].copy()


def _make_loader(dataset, batch_size, workers, seed, sampler=None, shuffle=False):
    import torch

    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
    )


def _run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    import torch

    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            use_amp = scaler is not None and scaler.is_enabled()
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(pixel_values=images).logits
                loss = criterion(logits, labels)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += float(loss.item()) * labels.size(0)
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_examples += labels.size(0)
    denominator = max(total_examples, 1)
    return total_loss / denominator, total_correct / denominator


def _predict_slices(model, loader, device, fold: int, positive_label: int) -> pd.DataFrame:
    import torch

    model.eval()
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for images, labels, metadata in loader:
            images = images.to(device)
            probabilities = torch.softmax(model(pixel_values=images).logits, dim=1).cpu().numpy()
            predictions = probabilities.argmax(axis=1)
            for index in range(len(labels)):
                rows.append(
                    {
                        "subject_id": str(metadata["subject_id"][index]),
                        "fold": int(fold),
                        "center_slice_index": int(metadata["center_slice_index"][index]),
                        "slice_offset": int(metadata["slice_offset"][index]),
                        "slice_index": int(metadata["slice_index"][index]),
                        "y_true": int(labels[index]),
                        "p_class_0": float(probabilities[index, 0]),
                        "p_class_1": float(probabilities[index, 1]),
                        "p_positive": float(probabilities[index, positive_label]),
                        "predicted_class_label": int(predictions[index]),
                        "is_positive_prediction": int(predictions[index] == positive_label),
                        "positive_label": int(positive_label),
                    }
                )
    return pd.DataFrame(rows).sort_values(["subject_id", "slice_offset"])


def _make_optimizer(model, training_config: dict[str, Any]):
    import torch

    name = str(training_config.get("optimizer", "adam")).lower()
    learning_rate = float(training_config.get("learning_rate", 1e-3))
    weight_decay = float(training_config.get("weight_decay", 0.0))
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    raise ValueError("training.optimizer must be 'adam' or 'adamw'")


def _audit_shared_frames(
    frames: dict[str, pd.DataFrame], expected_offsets: tuple[int, ...]
) -> dict[str, dict[str, int]]:
    """Verify that both roles contain every centered slice for every patient."""
    summary: dict[str, dict[str, int]] = {}
    for role in ("train", "validation"):
        frame = frames[role]
        if frame.empty:
            raise ValueError(f"The {role} frame is empty")
        expected_set = set(expected_offsets)
        bad_count = 0
        for _, group in frame.groupby("subject_id", sort=False):
            observed = set(group["slice_offset"].astype(int))
            if observed != expected_set or len(group) != len(expected_offsets):
                bad_count += 1
        if bad_count:
            raise ValueError(
                f"The {role} frame does not contain one complete centered window "
                f"for {bad_count} subjects"
            )
        summary[role] = {
            "subjects": int(frame["subject_id"].nunique()),
            "slice_rows": int(len(frame)),
        }
    return summary


def _train_shared_model(
    config: dict[str, Any],
    fold: int,
    frames: dict[str, pd.DataFrame],
    data_root: Path,
    fold_output: Path,
    device,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Train one fold-shared ViT on all five slices from every training patient."""
    import torch

    seed = int(config.get("seed", 42))
    training_config = config["training"]
    data_config = config["data"]
    positive_label = int(config.get("positive_label", 1))
    image_column = str(data_config.get("image_column", "fused_path"))
    expected_offsets = tuple(
        int(value) for value in data_config.get("slice_offsets", DEFAULT_SLICE_OFFSETS)
    )
    frame_summary = _audit_shared_frames(frames, expected_offsets)

    image_size = int(data_config.get("image_size", 384))
    augmentation = config.get("augmentation", {})
    train_transform = build_transforms(image_size, augmentation, training=True)
    eval_transform = build_transforms(image_size, augmentation, training=False)
    datasets = {
        "train": FusedRGBSliceDataset(
            frames["train"], data_root, image_column=image_column, transform=train_transform
        ),
        "validation": FusedRGBSliceDataset(
            frames["validation"],
            data_root,
            image_column=image_column,
            transform=eval_transform,
        ),
    }

    imbalance_strategy = str(config.get("imbalance", {}).get("strategy", "none"))
    allowed_strategies = {"none", "weighted_sampler", "class_weighted_loss", "both"}
    if imbalance_strategy not in allowed_strategies:
        raise ValueError(f"Unknown imbalance strategy: {imbalance_strategy}")
    sampler = (
        build_weighted_sampler(frames["train"])
        if imbalance_strategy in {"weighted_sampler", "both"}
        else None
    )
    batch_size = int(training_config.get("batch_size", 2))
    workers = int(training_config.get("num_workers", 2))
    run_seed = seed + fold
    seed_everything(run_seed)
    loaders = {
        "train": _make_loader(
            datasets["train"],
            batch_size,
            workers,
            run_seed,
            sampler,
            True,
        ),
        "validation": _make_loader(
            datasets["validation"],
            batch_size,
            workers,
            run_seed,
        ),
    }

    model_config = dict(config["model"])
    if model_config.get("checkpoint"):
        model_config["checkpoint"] = str(resolve_config_path(config, model_config["checkpoint"]))
    model = create_vit(model_config).to(device)
    ce_weight = None
    if imbalance_strategy in {"class_weighted_loss", "both"}:
        subject_labels = frames["train"][["subject_id", "label"]].drop_duplicates()["label"]
        ce_weight = torch.as_tensor(
            class_weight_vector(subject_labels), dtype=torch.float32, device=device
        )
    criterion = torch.nn.CrossEntropyLoss(weight=ce_weight)
    optimizer = _make_optimizer(model, training_config)
    amp_enabled = bool(training_config.get("amp", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    selection_metric = str(training_config.get("selection_metric", "val_accuracy"))
    if selection_metric not in {"val_accuracy", "val_loss"}:
        raise ValueError("selection_metric must be 'val_accuracy' or 'val_loss'")
    best_value = -float("inf") if selection_metric == "val_accuracy" else float("inf")
    best_path = fold_output / SHARED_CHECKPOINT_NAME
    history: list[dict[str, float | int]] = []
    epochs = int(training_config.get("epochs", 150))
    if epochs <= 0:
        raise ValueError("training.epochs must be a positive integer")
    saved_checkpoint = False
    for epoch in range(epochs):
        train_loss, train_accuracy = _run_epoch(
            model, loaders["train"], criterion, device, optimizer=optimizer, scaler=scaler
        )
        val_loss, val_accuracy = _run_epoch(
            model, loaders["validation"], criterion, device, scaler=None
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )
        value = val_accuracy if selection_metric == "val_accuracy" else val_loss
        improved = value > best_value if selection_metric == "val_accuracy" else value < best_value
        if improved:
            best_value = value
            torch.save(model.state_dict(), best_path)
            saved_checkpoint = True

    pd.DataFrame(history).to_csv(fold_output / "training_history.csv", index=False)
    if not saved_checkpoint or not best_path.is_file():
        raise RuntimeError("Training did not produce a finite best shared-model checkpoint")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    predictions = _predict_slices(
        model, loaders["validation"], device, fold=fold, positive_label=positive_label
    )
    predictions.to_csv(fold_output / "validation_slice_predictions.csv", index=False)
    return predictions, {
        "scope": "shared_across_all_slice_positions",
        "checkpoint": best_path.name,
        "selection_metric": selection_metric,
        "best_value": float(best_value),
        "slice_offsets": list(expected_offsets),
        "train_subjects": frame_summary["train"]["subjects"],
        "train_slice_rows": frame_summary["train"]["slice_rows"],
        "validation_subjects": frame_summary["validation"]["subjects"],
        "validation_slice_rows": frame_summary["validation"]["slice_rows"],
    }


def train_fold(config: dict[str, Any], fold: int) -> Path:
    """Train one shared ViT on all fold slices, then apply patient-level MSV."""
    import torch

    seed = int(config.get("seed", 42))
    seed_everything(seed + fold)
    data_config = config["data"]
    positive_label = int(config.get("positive_label", 1))
    if positive_label not in {0, 1}:
        raise ValueError("positive_label must be 0 or 1")

    manifest_path = resolve_config_path(config, data_config["manifest"])
    data_root = resolve_config_path(config, data_config.get("root", "."))
    assignments_path = resolve_config_path(config, config["splits"]["assignments_file"])
    output_root = resolve_config_path(config, config["output_dir"])
    fold_output = output_root / f"fold_{fold}"
    fold_output.mkdir(parents=True, exist_ok=True)

    image_column = str(data_config.get("image_column", "fused_path"))
    slice_offsets = tuple(
        int(value) for value in data_config.get("slice_offsets", DEFAULT_SLICE_OFFSETS)
    )
    manifest = load_manifest(manifest_path)
    validate_manifest(
        manifest,
        [image_column],
        expected_slices=int(data_config.get("expected_slices", 5)),
        expected_slice_offsets=slice_offsets,
        data_root=data_root,
        check_files=True,
    )
    assignments = pd.read_csv(assignments_path, dtype={"subject_id": "string"})
    expected_folds = int(config["splits"].get("n_folds", 4))
    audit_manifest_assignment_alignment(
        manifest,
        assignments,
        expected_folds=expected_folds,
    )
    if fold not in set(assignments["fold"].astype(int)):
        raise ValueError(f"Requested fold {fold} is not present in assignments")
    frames = {
        role: _subset_rows(manifest, assignments, fold, role) for role in ("train", "validation")
    }
    if any(frame.empty for frame in frames.values()):
        raise ValueError(f"Fold {fold} has an empty role")

    conflicting_directories = sorted(
        path.name for path in fold_output.glob("slice_*") if path.is_dir()
    )
    if conflicting_directories:
        raise RuntimeError(
            "The fold output contains incompatible slice-specific model directories "
            f"{conflicting_directories}. Use a new output directory for the shared-ViT run."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    slice_predictions, shared_model_metadata = _train_shared_model(
        config,
        fold,
        frames,
        data_root,
        fold_output,
        device,
    )

    voting_config = config.get("voting", {})
    patient_threshold = float(voting_config.get("patient_threshold", 0.5))
    voting_method = str(voting_config.get("method", "weighted_soft"))
    voting_weights = voting_config.get("weights", DEFAULT_MSV_WEIGHTS)
    threshold_inclusive = bool(voting_config.get("threshold_inclusive", False))
    subject_predictions = aggregate_slice_predictions(
        slice_predictions,
        expected_slices=len(slice_offsets),
        patient_threshold=patient_threshold,
        method=voting_method,
        positive_label=positive_label,
        weights=voting_weights if voting_method == "weighted_soft" else None,
        expected_slice_offsets=slice_offsets,
        threshold_inclusive=threshold_inclusive,
    )
    subject_predictions.insert(1, "fold", fold)
    subject_predictions.to_csv(fold_output / "validation_subject_predictions.csv", index=False)

    metrics = classification_metrics(
        subject_predictions["y_true"],
        subject_predictions["patient_score"],
        threshold=patient_threshold,
        positive_label=positive_label,
        threshold_inclusive=threshold_inclusive,
    )
    intervals = bootstrap_confidence_intervals(
        subject_predictions["y_true"],
        subject_predictions["patient_score"],
        threshold=patient_threshold,
        positive_label=positive_label,
        repetitions=int(config.get("bootstrap_repetitions", 2000)),
        seed=seed + fold,
        threshold_inclusive=threshold_inclusive,
    )
    with (fold_output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({"point_estimates": metrics, "bootstrap_ci": intervals}, handle, indent=2)

    try:
        import torchvision
        import transformers

        transformers_version = transformers.__version__
        torchvision_version = torchvision.__version__
    except ImportError:
        transformers_version = "unavailable"
        torchvision_version = "unavailable"
    metadata = {
        "config": public_config(config),
        "metadata_schema_version": 2,
        "fold": fold,
        "label_encoding": {"0": "MSA-P", "1": "PD"},
        "positive_label": positive_label,
        "model_scope": "one_shared_vit_per_fold",
        "shared_model": shared_model_metadata,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision_version,
        "transformers": transformers_version,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "manifest_sha256": file_sha256(manifest_path),
        "assignments_sha256": file_sha256(assignments_path),
        "train_subjects": int(frames["train"]["subject_id"].nunique()),
        "validation_subjects": int(frames["validation"]["subject_id"].nunique()),
    }
    with (fold_output / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    return fold_output
