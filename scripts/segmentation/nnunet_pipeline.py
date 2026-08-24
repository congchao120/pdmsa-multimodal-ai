#!/usr/bin/env python3
"""Thin wrapper around the official nnU-Net v2 command-line tools.

This script does not implement or modify nnU-Net.  It makes the three nnU-Net
storage locations explicit and prints the exact command that is executed. Run
with ``--dry-run`` before starting a long training job.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

NNUNET_COMMANDS = {
    "plan": "nnUNetv2_plan_and_preprocess",
    "train": "nnUNetv2_train",
    "predict": "nnUNetv2_predict",
}
FOUR_FOLDS = ("0", "1", "2", "3")
ENVIRONMENT_DIRECTORIES = (
    "nnUNet_raw",
    "nnUNet_preprocessed",
    "nnUNet_results",
)
DATASET_PATTERN = re.compile(r"^(?:\d+|Dataset\d{3}_.+)$")
SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


def _absolute_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _dataset(value: str) -> str:
    candidate = value.strip()
    if not DATASET_PATTERN.fullmatch(candidate):
        raise argparse.ArgumentTypeError(
            "dataset must be a numeric ID or an nnU-Net name such as Dataset800_PD"
        )
    return candidate


def _dataset_id(value: str) -> str:
    candidate = value.strip()
    try:
        number = int(candidate)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "planning requires the numeric nnU-Net dataset ID"
        ) from exc
    if number < 0:
        raise argparse.ArgumentTypeError("dataset ID must be non-negative")
    return str(number)


def _fold(value: str) -> str:
    candidate = value.strip()
    if candidate == "all":
        return candidate
    try:
        number = int(candidate)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("fold must be a non-negative integer or 'all'") from exc
    if number < 0:
        raise argparse.ArgumentTypeError("fold must be a non-negative integer or 'all'")
    return str(number)


def _sha256(value: str) -> str:
    candidate = value.strip().lower()
    if not SHA256_PATTERN.fullmatch(candidate):
        raise argparse.ArgumentTypeError("SHA-256 must contain exactly 64 hexadecimal characters")
    return candidate


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--raw-dir",
        type=_absolute_path,
        required=True,
        help="Absolute or resolvable path assigned to nnUNet_raw.",
    )
    parser.add_argument(
        "--preprocessed-dir",
        type=_absolute_path,
        required=True,
        help="Absolute or resolvable path assigned to nnUNet_preprocessed.",
    )
    parser.add_argument(
        "--results-dir",
        type=_absolute_path,
        required=True,
        help="Absolute or resolvable path assigned to nnUNet_results.",
    )
    parser.add_argument(
        "--create-dirs",
        action="store_true",
        help="Create the three environment directories if they do not yet exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print checks and the exact command without executing nnU-Net.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an official nnU-Net v2 pipeline with explicit, reproducible inputs."
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    check = subparsers.add_parser(
        "check",
        help=(
            "Inspect the installed nnU-Net tools, environment paths, and optionally "
            "an artifact hash."
        ),
    )
    _add_runtime_arguments(check)
    check.add_argument(
        "--artifact",
        type=_absolute_path,
        help="Optional release asset or checkpoint whose SHA-256 should be calculated.",
    )
    check.add_argument(
        "--expected-sha256",
        type=_sha256,
        help="Expected SHA-256; requires --artifact and fails if it does not match.",
    )

    plan = subparsers.add_parser(
        "plan",
        help="Call nnUNetv2_plan_and_preprocess for one dataset.",
    )
    _add_runtime_arguments(plan)
    plan.add_argument(
        "--dataset", type=_dataset_id, required=True, help="Numeric nnU-Net dataset ID."
    )
    plan.add_argument(
        "--configurations",
        nargs="+",
        required=True,
        help="Configurations to preprocess, for example: 2d 3d_fullres.",
    )
    plan.add_argument(
        "--plans",
        help="Plans identifier passed as -overwrite_plans_name.",
    )
    plan.add_argument(
        "--planner",
        help="Experiment planner class passed as -pl (omit to use the verified upstream default).",
    )
    plan.add_argument(
        "--verify-dataset-integrity",
        action="store_true",
        help="Pass --verify_dataset_integrity to nnU-Net.",
    )

    train = subparsers.add_parser(
        "train",
        help="Call nnUNetv2_train for one dataset/configuration/fold.",
    )
    _add_runtime_arguments(train)
    train.add_argument(
        "--dataset", type=_dataset, required=True, help="Dataset ID or DatasetXXX_Name."
    )
    train.add_argument("--configuration", required=True, help="For example: 2d or 3d_fullres.")
    train.add_argument("--fold", type=_fold, required=True, help="Fold number, or 'all'.")
    train.add_argument("--trainer", help="Trainer class passed as -tr.")
    train.add_argument("--plans", help="Plans identifier passed as -p.")
    train.add_argument(
        "--pretrained-weights",
        type=_absolute_path,
        help="Optional initialization checkpoint passed as -pretrained_weights.",
    )
    train.add_argument(
        "--save-probabilities",
        action="store_true",
        help="Pass --npz to save validation probabilities.",
    )
    train.add_argument(
        "--continue-training",
        "--continue",
        dest="continue_training",
        action="store_true",
        help="Resume this fold from its latest checkpoint (passed to nnU-Net as --c).",
    )

    train_fourfold = subparsers.add_parser(
        "train-fourfold",
        help="Train folds 0, 1, 2, and 3 sequentially for a four-fold experiment.",
    )
    _add_runtime_arguments(train_fourfold)
    train_fourfold.add_argument(
        "--study-config",
        type=_absolute_path,
        help=(
            "TOML file supplying [study], [split], and [model] settings. "
            "Explicit command-line values take precedence."
        ),
    )
    train_fourfold.add_argument(
        "--dataset",
        type=_dataset,
        help="Dataset ID or DatasetXXX_Name (required unless supplied by --study-config).",
    )
    train_fourfold.add_argument(
        "--configuration",
        help="For example: 2d or 3d_fullres (required unless supplied by --study-config).",
    )
    train_fourfold.add_argument("--trainer", help="Trainer class passed as -tr.")
    train_fourfold.add_argument("--plans", help="Plans identifier passed as -p.")
    train_fourfold.add_argument(
        "--pretrained-weights",
        type=_absolute_path,
        help="Optional initialization checkpoint passed as -pretrained_weights.",
    )
    train_fourfold.add_argument(
        "--save-probabilities",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Pass --npz to save validation probabilities for every fold.",
    )
    train_fourfold.add_argument(
        "--continue-training",
        "--continue",
        dest="continue_training",
        action="store_true",
        help=(
            "Pass --c to every fold. Existing checkpoints are resumed; the current "
            "nnU-Net starts a fold from scratch when that fold has no checkpoint."
        ),
    )

    predict = subparsers.add_parser(
        "predict",
        help="Call nnUNetv2_predict using an official nnU-Net results directory.",
    )
    _add_runtime_arguments(predict)
    predict.add_argument(
        "--input-dir", type=_absolute_path, required=True, help="Folder of input cases."
    )
    predict.add_argument(
        "--output-dir", type=_absolute_path, required=True, help="Prediction output folder."
    )
    predict.add_argument(
        "--dataset", type=_dataset, required=True, help="Dataset ID or DatasetXXX_Name."
    )
    predict.add_argument("--configuration", required=True, help="For example: 2d or 3d_fullres.")
    predict.add_argument(
        "--folds",
        type=_fold,
        nargs="+",
        required=True,
        help="One or more trained folds, or 'all'.",
    )
    predict.add_argument("--trainer", help="Trainer class passed as -tr.")
    predict.add_argument("--plans", help="Plans identifier passed as -p.")
    predict.add_argument(
        "--checkpoint-name",
        default="checkpoint_final.pth",
        help="Checkpoint file name inside each fold directory (default: checkpoint_final.pth).",
    )
    predict.add_argument(
        "--save-probabilities",
        action="store_true",
        help="Pass --save_probabilities to nnU-Net.",
    )

    predict_fourfold = subparsers.add_parser(
        "predict-fourfold",
        help="Ensemble folds 0, 1, 2, and 3 in one nnUNetv2_predict call.",
    )
    _add_runtime_arguments(predict_fourfold)
    predict_fourfold.add_argument(
        "--study-config",
        type=_absolute_path,
        help=(
            "TOML file supplying [study], [split], and [model] settings. "
            "Explicit command-line values take precedence."
        ),
    )
    predict_fourfold.add_argument(
        "--input-dir", type=_absolute_path, required=True, help="Folder of input cases."
    )
    predict_fourfold.add_argument(
        "--output-dir", type=_absolute_path, required=True, help="Prediction output folder."
    )
    predict_fourfold.add_argument(
        "--dataset",
        type=_dataset,
        help="Dataset ID or DatasetXXX_Name (required unless supplied by --study-config).",
    )
    predict_fourfold.add_argument(
        "--configuration",
        help="For example: 2d or 3d_fullres (required unless supplied by --study-config).",
    )
    predict_fourfold.add_argument("--trainer", help="Trainer class passed as -tr.")
    predict_fourfold.add_argument("--plans", help="Plans identifier passed as -p.")
    predict_fourfold.add_argument(
        "--checkpoint-name",
        help=(
            "Checkpoint file name inside each fold directory (default: value from "
            "--study-config, then checkpoint_final.pth)."
        ),
    )
    predict_fourfold.add_argument(
        "--save-probabilities",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=("Control --save_probabilities (default: value from --study-config, then false)."),
    )
    predict_fourfold.add_argument(
        "--step-size",
        type=float,
        help="Sliding-window step size in (0, 1] (default: study config, then 0.5).",
    )
    predict_fourfold.add_argument(
        "--use-mirroring",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable test-time mirroring (default: study config, then enabled).",
    )

    return parser


def _prepare_environment(args: argparse.Namespace) -> dict[str, str]:
    directories = {
        "nnUNet_raw": args.raw_dir,
        "nnUNet_preprocessed": args.preprocessed_dir,
        "nnUNet_results": args.results_dir,
    }
    for name, path in directories.items():
        if args.create_dirs:
            path.mkdir(parents=True, exist_ok=True)
        if not path.is_dir() and not args.dry_run:
            raise FileNotFoundError(
                f"{name} does not exist or is not a directory: {path}. "
                "Create it first or pass --create-dirs."
            )

    environment = os.environ.copy()
    environment.update({name: str(path) for name, path in directories.items()})
    return environment


def _installed_version() -> str | None:
    try:
        return importlib.metadata.version("nnunetv2")
    except importlib.metadata.PackageNotFoundError:
        return None


def _hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _display_command(command: Sequence[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    import shlex

    return shlex.join(command)


def _print_environment(args: argparse.Namespace) -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"nnunetv2 distribution: {_installed_version() or 'NOT INSTALLED'}")
    print(f"nnUNet_raw={args.raw_dir}")
    print(f"nnUNet_preprocessed={args.preprocessed_dir}")
    print(f"nnUNet_results={args.results_dir}")
    for command in NNUNET_COMMANDS.values():
        print(f"{command}: {shutil.which(command) or 'NOT FOUND'}")


def _check_command_available(command: str, dry_run: bool) -> None:
    if shutil.which(command) is None and not dry_run:
        raise RuntimeError(
            f"Required executable '{command}' was not found on PATH. "
            "Install a compatible nnunetv2 package before running."
        )


def _run(command: Sequence[str], environment: Mapping[str, str], dry_run: bool) -> int:
    print(f"Command: {_display_command(command)}")
    if dry_run:
        print("Dry run: command was not executed.")
        return 0
    completed = subprocess.run(list(command), env=dict(environment), check=False)
    return completed.returncode


def _plan_command(args: argparse.Namespace) -> list[str]:
    command = [
        NNUNET_COMMANDS["plan"],
        "-d",
        args.dataset,
        "-c",
        *args.configurations,
    ]
    if args.plans:
        command.extend(["-overwrite_plans_name", args.plans])
    if args.planner:
        command.extend(["-pl", args.planner])
    if args.verify_dataset_integrity:
        command.append("--verify_dataset_integrity")
    return command


def _train_command(args: argparse.Namespace, fold: str | None = None) -> list[str]:
    command = [
        NNUNET_COMMANDS["train"],
        args.dataset,
        args.configuration,
        fold if fold is not None else args.fold,
    ]
    if args.trainer:
        command.extend(["-tr", args.trainer])
    if args.plans:
        command.extend(["-p", args.plans])
    if args.pretrained_weights:
        command.extend(["-pretrained_weights", str(args.pretrained_weights)])
    if args.save_probabilities:
        command.append("--npz")
    if getattr(args, "continue_training", False):
        command.append("--c")
    return command


def _predict_command(args: argparse.Namespace, folds: Sequence[str] | None = None) -> list[str]:
    command = [
        NNUNET_COMMANDS["predict"],
        "-i",
        str(args.input_dir),
        "-o",
        str(args.output_dir),
        "-d",
        args.dataset,
        "-c",
        args.configuration,
        "-f",
        *(folds if folds is not None else args.folds),
    ]
    if args.trainer:
        command.extend(["-tr", args.trainer])
    if args.plans:
        command.extend(["-p", args.plans])
    if getattr(args, "step_size", None) is not None:
        command.extend(["-step_size", str(args.step_size)])
    if getattr(args, "use_mirroring", True) is False:
        command.append("--disable_tta")
    command.extend(["-chk", args.checkpoint_name])
    if args.save_probabilities:
        command.append("--save_probabilities")
    return command


def _check_artifact(args: argparse.Namespace) -> int:
    if args.expected_sha256 and not args.artifact:
        raise ValueError("--expected-sha256 requires --artifact")
    if not args.artifact:
        return 0
    if not args.artifact.is_file():
        raise FileNotFoundError(f"Artifact not found: {args.artifact}")
    actual = _hash_file(args.artifact)
    print(f"SHA-256 ({args.artifact.name}): {actual}")
    if args.expected_sha256 and actual != args.expected_sha256:
        print(
            f"SHA-256 mismatch: expected {args.expected_sha256}, got {actual}",
            file=sys.stderr,
        )
        return 2
    if args.expected_sha256:
        print("SHA-256 verification: OK")
    return 0


def _read_toml(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Study configuration not found: {path}")
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
        import tomli as tomllib  # type: ignore[no-redef]

    with path.open("rb") as stream:
        document = tomllib.load(stream)
    if not isinstance(document, dict):  # pragma: no cover - TOML documents are mappings
        raise ValueError(f"Study configuration must be a TOML table: {path}")
    return document


def _config_table(document: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = document.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"[{name}] in the study configuration must be a TOML table")
    return value


def _config_string(table: Mapping[str, object], key: str) -> str | None:
    value = table.get(key)
    if value is None:
        return None
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        raise ValueError(f"Study configuration value '{key}' must be a string or integer")
    candidate = str(value).strip()
    if not candidate:
        raise ValueError(f"Study configuration value '{key}' cannot be empty")
    return candidate


def _config_bool(table: Mapping[str, object], key: str) -> bool | None:
    value = table.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"Study configuration value '{key}' must be true or false")
    return value


def _config_float(table: Mapping[str, object], key: str) -> float | None:
    value = table.get(key)
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"Study configuration value '{key}' must be numeric")
    return float(value)


def _config_int(table: Mapping[str, object], key: str) -> int | None:
    value = table.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"Study configuration value '{key}' must be a positive integer")
    return value


def _configure_fourfold_action(args: argparse.Namespace) -> None:
    """Resolve four-fold arguments, with explicit CLI values overriding TOML."""

    document: Mapping[str, object] = {}
    if args.study_config:
        document = _read_toml(args.study_config)

    study = _config_table(document, "study")
    split = _config_table(document, "split")
    model = _config_table(document, "model")
    training = _config_table(document, "training")
    inference = _config_table(document, "inference")

    configured_dataset = _config_string(study, "dataset_name")
    if configured_dataset is None:
        configured_dataset = _config_string(study, "dataset_id")
    args.dataset = args.dataset or configured_dataset
    args.configuration = args.configuration or _config_string(model, "configuration")
    args.trainer = args.trainer or _config_string(model, "trainer")
    args.plans = args.plans or _config_string(model, "plans")
    args.expected_training_cases = _config_int(study, "num_training_cases")
    args.expected_train_cases_per_fold = _config_int(split, "expected_train_cases_per_fold")
    args.expected_validation_cases_per_fold = _config_int(
        split, "expected_validation_cases_per_fold"
    )
    if hasattr(args, "checkpoint_name"):
        args.checkpoint_name = (
            args.checkpoint_name
            or _config_string(model, "checkpoint_name")
            or "checkpoint_final.pth"
        )
    if args.action == "train-fourfold":
        configured_save_probabilities = _config_bool(training, "save_probabilities")
        if args.save_probabilities is None:
            args.save_probabilities = configured_save_probabilities or False
    else:
        configured_save_probabilities = _config_bool(inference, "save_probabilities")
        if args.save_probabilities is None:
            args.save_probabilities = configured_save_probabilities or False
        if args.step_size is None:
            configured_step_size = _config_float(inference, "step_size")
            args.step_size = configured_step_size if configured_step_size is not None else 0.5
        if not 0 < args.step_size <= 1:
            raise ValueError("inference step size must be greater than 0 and at most 1")
        if args.use_mirroring is None:
            configured_mirroring = _config_bool(inference, "use_mirroring")
            args.use_mirroring = configured_mirroring if configured_mirroring is not None else True
        configured_gaussian = _config_bool(inference, "use_gaussian")
        if configured_gaussian is False:
            raise ValueError(
                "nnUNetv2_predict enables Gaussian weighting by default; "
                "[inference].use_gaussian=false is unsupported"
            )

    if args.dataset is None:
        raise ValueError(
            "dataset is required: pass --dataset or set [study].dataset_name in --study-config"
        )
    if not DATASET_PATTERN.fullmatch(args.dataset):
        raise ValueError("dataset must be a numeric ID or an nnU-Net name such as Dataset800_PD")
    if args.configuration is None:
        raise ValueError(
            "configuration is required: pass --configuration or set "
            "[model].configuration in --study-config"
        )

    configured_split_count = _config_int(split, "n_splits")
    if configured_split_count is not None and configured_split_count != len(FOUR_FOLDS):
        raise ValueError("The four-fold workflow requires [split].n_splits = 4")
    configured_folds = split.get("folds", list(FOUR_FOLDS))
    if not isinstance(configured_folds, list):
        raise ValueError("[split].folds must be a TOML array containing 0, 1, 2, and 3")
    folds = tuple(str(fold) for fold in configured_folds)
    if folds != FOUR_FOLDS:
        raise ValueError("The four-fold workflow requires [split].folds = [0, 1, 2, 3] exactly")
    args.fourfold_folds = folds

    if args.study_config:
        print(f"Study config: {args.study_config}")
    print(f"Four-fold folds: {' '.join(args.fourfold_folds)}")


def _validate_training_options(args: argparse.Namespace) -> None:
    if getattr(args, "continue_training", False) and args.pretrained_weights:
        raise ValueError(
            "--continue-training and --pretrained-weights cannot be used together; "
            "nnU-Net treats resume and initialization as mutually exclusive."
        )


def _preprocessed_dataset_directory(
    preprocessed_dir: Path, dataset: str, dry_run: bool
) -> Path | None:
    if dataset.startswith("Dataset"):
        return preprocessed_dir / dataset

    dataset_prefix = f"Dataset{int(dataset):03d}_"
    matches = sorted(path for path in preprocessed_dir.glob(f"{dataset_prefix}*") if path.is_dir())
    if len(matches) == 1:
        return matches[0]
    if not matches and dry_run:
        print(
            "Dry run split check: expected exactly one directory matching "
            f"{preprocessed_dir / (dataset_prefix + '*')}."
        )
        return None
    if not matches:
        raise FileNotFoundError(
            f"No preprocessed directory matches dataset ID {dataset} under {preprocessed_dir}"
        )
    raise ValueError(
        f"Dataset ID {dataset} is ambiguous: found {len(matches)} matching "
        f"preprocessed directories under {preprocessed_dir}"
    )


def _validate_fourfold_split_document(document: object, path: Path) -> list[tuple[int, int]]:
    if not isinstance(document, list) or len(document) != len(FOUR_FOLDS):
        raise ValueError(f"{path} must contain exactly four split objects for folds 0, 1, 2, and 3")

    fold_sets: list[tuple[set[str], set[str]]] = []
    universe: set[str] | None = None
    validation_appearances: Counter[str] = Counter()
    counts: list[tuple[int, int]] = []

    for fold_index, split in enumerate(document):
        if not isinstance(split, dict):
            raise ValueError(f"Fold {fold_index} in {path} must be a JSON object")
        train = split.get("train")
        validation = split.get("val")
        if not isinstance(train, list) or not isinstance(validation, list):
            raise ValueError(f"Fold {fold_index} in {path} must contain train and val arrays")
        if not train or not validation:
            raise ValueError(f"Fold {fold_index} in {path} has an empty train or val array")
        if not all(isinstance(case_id, str) and case_id for case_id in train + validation):
            raise ValueError(f"Fold {fold_index} in {path} contains an invalid case ID")

        train_set = set(train)
        validation_set = set(validation)
        if len(train_set) != len(train) or len(validation_set) != len(validation):
            raise ValueError(f"Fold {fold_index} in {path} contains duplicate case IDs")
        if train_set & validation_set:
            raise ValueError(f"Fold {fold_index} in {path} has train/val overlap")

        fold_universe = train_set | validation_set
        if universe is None:
            universe = fold_universe
        elif fold_universe != universe:
            raise ValueError(f"Fold {fold_index} in {path} does not cover the same cases")

        fold_sets.append((train_set, validation_set))
        validation_appearances.update(validation_set)
        counts.append((len(train_set), len(validation_set)))

    assert universe is not None  # four non-empty folds were required above
    invalid_appearances = [case_id for case_id in universe if validation_appearances[case_id] != 1]
    if invalid_appearances:
        raise ValueError(
            f"{path} is not a valid four-fold partition: each case must appear in val once"
        )
    for fold_index, (train_set, validation_set) in enumerate(fold_sets):
        if train_set != universe - validation_set:
            raise ValueError(
                f"Fold {fold_index} in {path} train cases are not the complement of val cases"
            )
    return counts


def _verify_fourfold_splits(args: argparse.Namespace) -> None:
    dataset_dir = _preprocessed_dataset_directory(args.preprocessed_dir, args.dataset, args.dry_run)
    if dataset_dir is None:
        print(
            "Dry run: splits_final.json was not validated because the dataset directory "
            "does not exist yet. Generate the four-fold split before real training."
        )
        return

    split_path = dataset_dir / "splits_final.json"
    print(f"Four-fold split file: {split_path}")
    if not split_path.is_file():
        if args.dry_run:
            print(
                "Dry run: splits_final.json does not exist yet. Real training will refuse "
                "to start until a validated four-fold file is present."
            )
            return
        raise FileNotFoundError(
            f"Required four-fold split file not found: {split_path}. Refusing to let "
            "nnU-Net silently generate its default split."
        )

    try:
        with split_path.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in four-fold split file {split_path}: {exc}") from exc
    counts = _validate_fourfold_split_document(document, split_path)
    total_cases = sum(counts[0])
    if args.expected_training_cases is not None and total_cases != args.expected_training_cases:
        raise ValueError(
            f"{split_path} covers {total_cases} cases, but the study config expects "
            f"{args.expected_training_cases}"
        )
    if args.expected_train_cases_per_fold is not None and any(
        train_count != args.expected_train_cases_per_fold for train_count, _ in counts
    ):
        raise ValueError(
            f"{split_path} does not have the configured train-case count in every fold"
        )
    if args.expected_validation_cases_per_fold is not None and any(
        validation_count != args.expected_validation_cases_per_fold
        for _, validation_count in counts
    ):
        raise ValueError(
            f"{split_path} does not have the configured validation-case count in every fold"
        )
    print(
        "Four-fold split validation: OK ("
        + ", ".join(
            f"fold {fold}: train={train_count}, val={validation_count}"
            for fold, (train_count, validation_count) in enumerate(counts)
        )
        + ")"
    )


def _run_fourfold_training(args: argparse.Namespace, environment: Mapping[str, str]) -> int:
    """Run the four independent training commands in a deterministic order."""

    _validate_training_options(args)
    _verify_fourfold_splits(args)
    _check_command_available(NNUNET_COMMANDS["train"], args.dry_run)
    for fold in args.fourfold_folds:
        print(f"Four-fold training: fold {fold} ({int(fold) + 1}/{len(args.fourfold_folds)})")
        return_code = _run(
            _train_command(args, fold=fold),
            environment,
            args.dry_run,
        )
        if return_code != 0:
            print(
                f"Four-fold training stopped because fold {fold} returned {return_code}.",
                file=sys.stderr,
            )
            return return_code
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.action in {"train-fourfold", "predict-fourfold"}:
            _configure_fourfold_action(args)
        environment = _prepare_environment(args)
        _print_environment(args)

        if args.action == "check":
            return _check_artifact(args)
        if args.action == "plan":
            command = _plan_command(args)
        elif args.action == "train":
            _validate_training_options(args)
            command = _train_command(args)
        elif args.action == "train-fourfold":
            return _run_fourfold_training(args, environment)
        elif args.action == "predict":
            if not args.input_dir.is_dir() and not args.dry_run:
                raise FileNotFoundError(f"Prediction input directory not found: {args.input_dir}")
            command = _predict_command(args)
        elif args.action == "predict-fourfold":
            if not args.input_dir.is_dir() and not args.dry_run:
                raise FileNotFoundError(f"Prediction input directory not found: {args.input_dir}")
            command = _predict_command(args, folds=args.fourfold_folds)
        else:  # pragma: no cover - argparse guarantees a known action
            parser.error(f"Unknown action: {args.action}")

        _check_command_available(command[0], args.dry_run)
        return _run(command, environment, args.dry_run)
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
