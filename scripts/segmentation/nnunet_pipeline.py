#!/usr/bin/env python3
"""Thin, auditable wrapper around the official nnU-Net v2 command-line tools.

This script does not implement or modify nnU-Net.  It makes the three nnU-Net
storage locations explicit and records/prints the exact upstream command that
is executed.  Run with ``--dry-run`` before starting a long server job.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Iterable, Mapping, Sequence


NNUNET_COMMANDS = {
    "plan": "nnUNetv2_plan_and_preprocess",
    "train": "nnUNetv2_train",
    "predict": "nnUNetv2_predict",
}
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
            "dataset must be a numeric ID or an nnU-Net name such as "
            "Dataset800_PD"
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
        help="Pass --npz so validation probabilities are retained.",
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
            "Install the exact, server-verified nnunetv2 revision before running."
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


def _train_command(args: argparse.Namespace) -> list[str]:
    command = [
        NNUNET_COMMANDS["train"],
        args.dataset,
        args.configuration,
        args.fold,
    ]
    if args.trainer:
        command.extend(["-tr", args.trainer])
    if args.plans:
        command.extend(["-p", args.plans])
    if args.pretrained_weights:
        command.extend(["-pretrained_weights", str(args.pretrained_weights)])
    if args.save_probabilities:
        command.append("--npz")
    return command


def _predict_command(args: argparse.Namespace) -> list[str]:
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
        *args.folds,
    ]
    if args.trainer:
        command.extend(["-tr", args.trainer])
    if args.plans:
        command.extend(["-p", args.plans])
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


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        environment = _prepare_environment(args)
        _print_environment(args)

        if args.action == "check":
            return _check_artifact(args)
        if args.action == "plan":
            command = _plan_command(args)
        elif args.action == "train":
            command = _train_command(args)
        elif args.action == "predict":
            if not args.input_dir.is_dir() and not args.dry_run:
                raise FileNotFoundError(f"Prediction input directory not found: {args.input_dir}")
            command = _predict_command(args)
        else:  # pragma: no cover - argparse guarantees a known action
            parser.error(f"Unknown action: {args.action}")

        _check_command_available(command[0], args.dry_run)
        return _run(command, environment, args.dry_run)
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
