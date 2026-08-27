import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.segmentation import nnunet_pipeline as pipeline


def _write_study_config(
    path: Path,
    folds: str = "[0, 1, 2, 3]",
    num_training_cases: int = 12,
) -> None:
    path.write_text(
        "\n".join(
            [
                "[study]",
                'dataset_name = "Dataset800_PD"',
                f"num_training_cases = {num_training_cases}",
                "",
                "[split]",
                "n_splits = 4",
                f"folds = {folds}",
                "expected_train_cases_per_fold = 9",
                "expected_validation_cases_per_fold = 3",
                "",
                "[model]",
                'configuration = "3d_fullres"',
                'trainer = "nnUNetTrainer_150epochs"',
                'plans = "nnUNetPlans"',
                'checkpoint_name = "checkpoint_final.pth"',
                "",
                "[inference]",
                "save_probabilities = true",
                "step_size = 0.5",
                "use_gaussian = true",
                "use_mirroring = true",
            ]
        ),
        encoding="utf-8",
    )


def _fourfold_document(case_count: int = 12) -> list[dict[str, list[str]]]:
    cases = [f"case_{index:03d}" for index in range(case_count)]
    validation_size = case_count // 4
    document = []
    for fold in range(4):
        validation = cases[fold * validation_size : (fold + 1) * validation_size]
        document.append(
            {
                "train": [case for case in cases if case not in validation],
                "val": validation,
            }
        )
    return document


def _runtime_arguments(root: Path) -> list[str]:
    return [
        "--raw-dir",
        str(root / "raw"),
        "--preprocessed-dir",
        str(root / "preprocessed"),
        "--results-dir",
        str(root / "results"),
    ]


class NnUNetPipelineTests(unittest.TestCase):
    def test_existing_train_command_remains_compatible(self):
        args = argparse.Namespace(
            dataset="Dataset800_PD",
            configuration="3d_fullres",
            fold="2",
            trainer=None,
            plans=None,
            pretrained_weights=None,
            save_probabilities=False,
            continue_training=False,
        )
        self.assertEqual(
            pipeline._train_command(args),
            ["nnUNetv2_train", "Dataset800_PD", "3d_fullres", "2"],
        )

    def test_fourfold_split_validation_accepts_one_validation_appearance(self):
        counts = pipeline._validate_fourfold_split_document(
            _fourfold_document(), Path("splits_final.json")
        )
        self.assertEqual(counts, [(9, 3), (9, 3), (9, 3), (9, 3)])

    def test_fourfold_split_validation_rejects_validation_reuse(self):
        document = _fourfold_document()
        document[1]["val"] = list(document[0]["val"])
        document[1]["train"] = [
            case
            for case in document[0]["train"] + document[0]["val"]
            if case not in document[1]["val"]
        ]
        with self.assertRaisesRegex(ValueError, "each case must appear in val once"):
            pipeline._validate_fourfold_split_document(document, Path("splits_final.json"))

    def test_train_fourfold_uses_config_and_runs_folds_in_order(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = root / "study.toml"
            _write_study_config(config)
            argv = [
                "train-fourfold",
                *_runtime_arguments(root),
                "--study-config",
                str(config),
                "--save-probabilities",
                "--continue-training",
                "--dry-run",
            ]
            with mock.patch.object(pipeline, "_run", return_value=0) as run:
                return_code = pipeline.main(argv)

        self.assertEqual(return_code, 0)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual([command[3] for command in commands], ["0", "1", "2", "3"])
        for command in commands:
            self.assertEqual(command[0:3], ["nnUNetv2_train", "Dataset800_PD", "3d_fullres"])
            self.assertIn("nnUNetTrainer_100epochs", command)
            self.assertIn("nnUNetPlans", command)
            self.assertIn("--npz", command)
            self.assertEqual(command[-1], "--c")

    def test_train_fourfold_stops_after_first_failed_fold(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            argv = [
                "train-fourfold",
                *_runtime_arguments(root),
                "--dataset",
                "Dataset800_PD",
                "--configuration",
                "3d_fullres",
                "--dry-run",
            ]
            with mock.patch.object(pipeline, "_run", side_effect=[0, 7]) as run:
                return_code = pipeline.main(argv)

        self.assertEqual(return_code, 7)
        self.assertEqual(run.call_count, 2)

    def test_real_train_fourfold_refuses_missing_split_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for directory in ("raw", "preprocessed", "results"):
                (root / directory).mkdir()
            (root / "preprocessed" / "Dataset800_PD").mkdir()
            return_code = pipeline.main(
                [
                    "train-fourfold",
                    *_runtime_arguments(root),
                    "--dataset",
                    "Dataset800_PD",
                    "--configuration",
                    "3d_fullres",
                ]
            )

        self.assertEqual(return_code, 2)

    def test_real_train_fourfold_accepts_valid_split_before_launch(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for directory in ("raw", "preprocessed", "results"):
                (root / directory).mkdir()
            dataset_dir = root / "preprocessed" / "Dataset800_PD"
            dataset_dir.mkdir()
            (dataset_dir / "splits_final.json").write_text(
                json.dumps(_fourfold_document()), encoding="utf-8"
            )
            argv = [
                "train-fourfold",
                *_runtime_arguments(root),
                "--dataset",
                "Dataset800_PD",
                "--configuration",
                "3d_fullres",
            ]
            with (
                mock.patch.object(pipeline, "_check_command_available"),
                mock.patch.object(pipeline, "_run", return_value=0) as run,
            ):
                return_code = pipeline.main(argv)

        self.assertEqual(return_code, 0)
        self.assertEqual(run.call_count, 4)

    def test_existing_split_is_checked_against_configured_case_count(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = root / "study.toml"
            _write_study_config(config, num_training_cases=156)
            dataset_dir = root / "preprocessed" / "Dataset800_PD"
            dataset_dir.mkdir(parents=True)
            (dataset_dir / "splits_final.json").write_text(
                json.dumps(_fourfold_document()), encoding="utf-8"
            )
            with mock.patch.object(pipeline, "_run") as run:
                return_code = pipeline.main(
                    [
                        "train-fourfold",
                        *_runtime_arguments(root),
                        "--study-config",
                        str(config),
                        "--dry-run",
                    ]
                )

        self.assertEqual(return_code, 2)
        run.assert_not_called()

    def test_predict_fourfold_builds_one_ensemble_command(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = root / "study.toml"
            _write_study_config(config)
            argv = [
                "predict-fourfold",
                *_runtime_arguments(root),
                "--study-config",
                str(config),
                "--input-dir",
                str(root / "input"),
                "--output-dir",
                str(root / "output"),
                "--dry-run",
            ]
            with mock.patch.object(pipeline, "_run", return_value=0) as run:
                return_code = pipeline.main(argv)

        self.assertEqual(return_code, 0)
        run.assert_called_once()
        command = run.call_args.args[0]
        fold_flag = command.index("-f")
        self.assertEqual(command[fold_flag + 1 : fold_flag + 5], ["0", "1", "2", "3"])
        step_flag = command.index("-step_size")
        self.assertEqual(command[step_flag + 1], "0.5")
        self.assertEqual(command[-3:], ["-chk", "checkpoint_final.pth", "--save_probabilities"])

    def test_study_config_rejects_non_fourfold_fold_list(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = root / "study.toml"
            _write_study_config(config, folds="[0, 1, 2, 3, 4]")
            with mock.patch.object(pipeline, "_run") as run:
                return_code = pipeline.main(
                    [
                        "train-fourfold",
                        *_runtime_arguments(root),
                        "--study-config",
                        str(config),
                        "--dry-run",
                    ]
                )

        self.assertEqual(return_code, 2)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
