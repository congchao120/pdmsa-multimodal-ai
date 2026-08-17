import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from pdmsa.segmentation_splits import (
    discover_case_ids,
    make_kfold_splits,
    read_case_ids,
    validate_splits,
    write_splits_final,
)


class SegmentationSplitTests(unittest.TestCase):
    def test_discovers_compound_file_ending_without_reading_contents(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            labels_dir = Path(temporary_directory)
            for name in ("SYN_003.nii.gz", "SYN_001.nii.gz", "SYN_002.nii.gz"):
                (labels_dir / name).touch()
            (labels_dir / "dataset.json").touch()
            (labels_dir / "SYN_004.nrrd").touch()

            self.assertEqual(
                discover_case_ids(labels_dir, file_ending=".nii.gz"),
                ["SYN_001", "SYN_002", "SYN_003"],
            )

    def test_reads_case_list_and_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            case_list = Path(temporary_directory) / "cases.txt"
            case_list.write_text("# synthetic IDs only\nSYN_002\n\nSYN_001\n", encoding="utf-8")
            self.assertEqual(read_case_ids(case_list), ["SYN_001", "SYN_002"])

            case_list.write_text("SYN_001\nSYN_001\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_case_ids(case_list)

    def test_fourfold_membership_is_deterministic_balanced_and_complete(self):
        case_ids = [f"SYN_{index:03d}" for index in range(10)]
        left = make_kfold_splits(case_ids, n_splits=4, seed=12345)
        right = make_kfold_splits(reversed(case_ids), n_splits=4, seed=12345)

        self.assertEqual(left, right)
        self.assertEqual(validate_splits(left, case_ids, n_splits=4), [3, 3, 2, 2])
        validation_ids = [case_id for split in left for case_id in split["val"]]
        self.assertCountEqual(validation_ids, case_ids)
        for split in left:
            self.assertFalse(set(split["train"]).intersection(split["val"]))
            self.assertEqual(set(split["train"]).union(split["val"]), set(case_ids))

        compatibility_cases = [f"SYN_{index:03d}" for index in range(8)]
        compatibility_splits = make_kfold_splits(compatibility_cases)
        self.assertEqual(
            [split["val"] for split in compatibility_splits],
            [
                ["SYN_003", "SYN_007"],
                ["SYN_000", "SYN_004"],
                ["SYN_001", "SYN_006"],
                ["SYN_002", "SYN_005"],
            ],
        )

    def test_rejects_invalid_split_inputs(self):
        with self.assertRaisesRegex(ValueError, "At least"):
            make_kfold_splits(["SYN_001", "SYN_002", "SYN_003"], n_splits=4)

        case_ids = [f"SYN_{index:03d}" for index in range(8)]
        splits = make_kfold_splits(case_ids)
        splits[0]["train"].append(splits[0]["val"][0])
        with self.assertRaisesRegex(ValueError, "overlap|Duplicate"):
            validate_splits(splits, case_ids, n_splits=4)

    def test_156_synthetic_cases_produce_117_train_and_39_validation(self):
        case_ids = [f"SYN_{index:03d}" for index in range(156)]
        splits = make_kfold_splits(case_ids)
        self.assertEqual(validate_splits(splits, case_ids, n_splits=4), [39, 39, 39, 39])
        self.assertEqual([len(split["train"]) for split in splits], [117, 117, 117, 117])

    def test_atomic_writer_refuses_overwrite_by_default(self):
        case_ids = [f"SYN_{index:03d}" for index in range(8)]
        splits = make_kfold_splits(case_ids)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "splits_final.json"
            write_splits_final(output, splits)
            original = output.read_bytes()
            self.assertEqual(json.loads(original), splits)

            with self.assertRaises(FileExistsError):
                write_splits_final(output, list(reversed(splits)))
            self.assertEqual(output.read_bytes(), original)
            self.assertEqual(list(output.parent.glob(f".{output.name}.*.tmp")), [])

            write_splits_final(output, list(reversed(splits)), overwrite=True)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), list(reversed(splits)))

    def test_cli_generates_only_synthetic_fourfold_json(self):
        repository = Path(__file__).resolve().parents[1]
        script = repository / "scripts" / "segmentation" / "make_fourfold_splits.py"
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            labels_dir = temporary / "labelsTr"
            labels_dir.mkdir()
            for index in range(8):
                (labels_dir / f"SYN_{index:03d}.nii.gz").touch()
            output = temporary / "preprocessed" / "Dataset999_SYN" / "splits_final.json"

            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--labels-dir",
                    str(labels_dir),
                    "--output",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            generated = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(generated), 4)
            self.assertEqual([len(split["val"]) for split in generated], [2, 2, 2, 2])
            self.assertNotIn("SYN_000", completed.stdout)


if __name__ == "__main__":
    unittest.main()
