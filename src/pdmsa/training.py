from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import public_config, resolve_config_path
from .data import FusedRGBSliceDataset, build_transforms
from .imbalance import build_weighted_sampler, class_weight_vector
from .manifest import load_manifest, validate_manifest
from .metrics import bootstrap_confidence_intervals, classification_metrics
from .models import create_vit
from .reproducibility import file_sha256, seed_everything, seed_worker
from .splits import audit_manifest_assignment_alignment
from .voting import aggregate_slice_predictions


def _subset_rows(manifest: pd.DataFrame, assignments: pd.DataFrame, fold: int, role: str):
    ids = set(
        assignments.loc[
            (assignments["fold"].astype(int) == fold) & (assignments["role"] == role),
            "subject_id",
        ].astype(str)
    )
    return manifest.loc[manifest["subject_id"].astype(str).isin(ids)].copy()


def _make_loader(dataset, frame, batch_size, workers, seed, sampler=None, shuffle=False):
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
                        "slice_index": int(metadata["slice_index"][index]),
                        "input_file": str(metadata["input_file"][index]),
                        "y_true": int(labels[index]),
                        "p_class_0": float(probabilities[index, 0]),
                        "p_class_1": float(probabilities[index, 1]),
                        "p_positive": float(probabilities[index, positive_label]),
                        "predicted_class_label": int(predictions[index]),
                        "is_positive_prediction": int(predictions[index] == positive_label),
                        "positive_label": int(positive_label),
                    }
                )
    return pd.DataFrame(rows).sort_values(["subject_id", "slice_index"])


def _make_optimizer(model, training_config: dict[str, Any]):
    import torch

    name = str(training_config.get("optimizer", "adam")).lower()
    learning_rate = float(training_config.get("learning_rate", 1e-3))
    weight_decay = float(training_config.get("weight_decay", 0.0))
    if name == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    raise ValueError("training.optimizer must be 'adam' or 'adamw'")


def _train_one_slice(
    config: dict[str, Any],
    fold: int,
    slice_index: int,
    frames: dict[str, pd.DataFrame],
    data_root: Path,
    fold_output: Path,
    device,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Train one independent ViT, mirroring one retained ``vit*.py`` run."""
    import torch

    seed = int(config.get("seed", 42))
    training_config = config["training"]
    data_config = config["data"]
    positive_label = int(config.get("positive_label", 1))
    image_column = str(data_config.get("image_column", "fused_path"))
    slice_frames = {
        role: frame.loc[frame["slice_index"].astype(int) == slice_index].copy()
        for role, frame in frames.items()
    }
    if any(frame.empty for frame in slice_frames.values()):
        raise ValueError(f"Fold {fold}, slice {slice_index} has an empty train/validation role")

    image_size = int(data_config.get("image_size", 224))
    augmentation = config.get("augmentation", {})
    train_transform = build_transforms(image_size, augmentation, training=True)
    eval_transform = build_transforms(image_size, augmentation, training=False)
    datasets = {
        "train": FusedRGBSliceDataset(
            slice_frames["train"], data_root, image_column=image_column, transform=train_transform
        ),
        "validation": FusedRGBSliceDataset(
            slice_frames["validation"],
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
        build_weighted_sampler(slice_frames["train"])
        if imbalance_strategy in {"weighted_sampler", "both"}
        else None
    )
    batch_size = int(training_config.get("batch_size", 32))
    workers = int(training_config.get("num_workers", 4))
    run_seed = seed + fold * 100 + slice_index
    seed_everything(run_seed)
    loaders = {
        "train": _make_loader(
            datasets["train"],
            slice_frames["train"],
            batch_size,
            workers,
            run_seed,
            sampler,
            True,
        ),
        "validation": _make_loader(
            datasets["validation"],
            slice_frames["validation"],
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
        subject_labels = (
            slice_frames["train"][["subject_id", "label"]].drop_duplicates()["label"]
        )
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
    layer_output = fold_output / f"slice_{slice_index}"
    layer_output.mkdir(parents=True, exist_ok=True)
    best_path = layer_output / "best_model_weights.pth"
    history: list[dict[str, float | int]] = []
    for epoch in range(int(training_config.get("epochs", 150))):
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

    pd.DataFrame(history).to_csv(layer_output / "training_history.csv", index=False)
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    predictions = _predict_slices(
        model, loaders["validation"], device, fold=fold, positive_label=positive_label
    )
    predictions.to_csv(layer_output / "validation_predictions.csv", index=False)
    return predictions, {
        "slice_index": slice_index,
        "checkpoint": str(best_path.relative_to(fold_output)),
        "selection_metric": selection_metric,
        "best_value": float(best_value),
        "train_subjects": int(slice_frames["train"]["subject_id"].nunique()),
        "validation_subjects": int(slice_frames["validation"]["subject_id"].nunique()),
    }


def train_fold(config: dict[str, Any], fold: int) -> Path:
    """Train five independent slice models, then apply fixed-weight MSV."""
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
    slice_indices = [int(value) for value in data_config.get("slice_indices", [6, 7, 8, 9, 10])]
    manifest = load_manifest(manifest_path)
    validate_manifest(
        manifest,
        [image_column],
        expected_slices=int(data_config.get("expected_slices", 5)),
        expected_slice_indices=slice_indices,
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
        role: _subset_rows(manifest, assignments, fold, role)
        for role in ("train", "validation")
    }
    if any(frame.empty for frame in frames.values()):
        raise ValueError(f"Fold {fold} has an empty role")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions: list[pd.DataFrame] = []
    layer_runs: list[dict[str, object]] = []
    for slice_index in slice_indices:
        slice_predictions, layer_metadata = _train_one_slice(
            config,
            fold,
            slice_index,
            frames,
            data_root,
            fold_output,
            device,
        )
        predictions.append(slice_predictions)
        layer_runs.append(layer_metadata)

    slice_predictions = pd.concat(predictions, ignore_index=True).sort_values(
        ["subject_id", "slice_index"]
    )
    slice_predictions.to_csv(fold_output / "validation_slice_predictions.csv", index=False)

    voting_config = config.get("voting", {})
    patient_threshold = float(voting_config.get("patient_threshold", 0.5))
    voting_method = str(voting_config.get("method", "weighted_soft"))
    voting_weights = voting_config.get("weights", [0.1, 0.2, 0.4, 0.2, 0.1])
    threshold_inclusive = bool(voting_config.get("threshold_inclusive", False))
    subject_predictions = aggregate_slice_predictions(
        slice_predictions,
        expected_slices=len(slice_indices),
        patient_threshold=patient_threshold,
        method=voting_method,
        positive_label=positive_label,
        weights=voting_weights if voting_method == "weighted_soft" else None,
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
        import transformers
        import torchvision

        transformers_version = transformers.__version__
        torchvision_version = torchvision.__version__
    except ImportError:
        transformers_version = "unavailable"
        torchvision_version = "unavailable"
    metadata = {
        "config": public_config(config),
        "fold": fold,
        "label_encoding": {"0": "MSA-P", "1": "PD"},
        "positive_label": positive_label,
        "slice_models": layer_runs,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision_version,
        "transformers": transformers_version,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device": str(device),
        "manifest_sha256": file_sha256(manifest_path),
        "assignments_sha256": file_sha256(assignments_path),
        "train_subjects": int(frames["train"]["subject_id"].nunique()),
        "validation_subjects": int(frames["validation"]["subject_id"].nunique()),
    }
    with (fold_output / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    return fold_output
