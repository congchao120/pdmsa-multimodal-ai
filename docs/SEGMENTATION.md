# nnU-Net four-fold segmentation reproducibility

Segmentation is a separate upstream stage that produces ROI masks for the classification
pipeline. This repository provides an auditable four-fold nnU-Net v2 retraining recipe; it does
not distribute patient data, manual labels, real split assignments, predictions, logs, or model
weights.

The downloaded nnU-Net result directory contains five trained folds (0--4). Those checkpoints
cannot be relabeled as a four-fold model. The instructions below define a new run with folds 0--3.
Until that run has been completed, its outputs and metrics must not be described as the source of
previously reported results.

## Recorded configuration

The executable methods record is `configs/segmentation_fourfold.toml`. The main settings are:

| Item | Value |
| --- | --- |
| Dataset | `Dataset800_PD` (ID 800), 156 training cases |
| Channels | `0000`: FLAIR, `0001`: T1w, `0002`: T2w |
| Foreground labels | Generic `Region 1`--`Region 4` (values 1--4) |
| Configuration | `3d_fullres`, `nnUNetPlans`, `PlainConvUNet` |
| Trainer | `nnUNetTrainer_100epochs` |
| Cross-validation | folds 0--3; 117 training and 39 validation cases per fold |
| Split algorithm | sorted case keys + non-stratified `KFold(n_splits=4, shuffle=True, random_state=12345)` |
| Patch / batch | `[20, 320, 256]` / 2 |

The generic region labels are copied from the retained `dataset.json`. Their anatomical meanings
have not been independently confirmed and are deliberately not guessed in this repository. A
privacy-safe dataset metadata template is available at
`segmentation/templates/Dataset800_PD/dataset.json`.

## 1. Prepare the environment

The repository utilities require NumPy and scikit-learn for split generation:

```bash
python -m pip install -r requirements/segmentation.txt
```

Install the audited retained nnU-Net source tree separately:

```bash
python -m pip install -e /absolute/path/to/audited/retained/nnUNet/source
```

The retained source labels itself `2.1.1`, but the exact upstream commit is not yet verified and a
matching public PyPI release must not be invented. The retained logs record PyTorch 2.0.1+cu117,
cuDNN 8.5.0, and an NVIDIA GeForce RTX 3080; these are provenance evidence, not a portable lock.

Confirm that the existing 100-epoch trainer variant is importable:

```bash
python -c "from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import nnUNetTrainer_100epochs"
```

This explicit trainer matters: the retained run logs show 100 epochs, while the base
`nnUNetTrainer` in the retained source currently defaults to 200. The recipe does not add or modify
a network implementation.

## 2. Prepare the authorized dataset

Use nnU-Net v2 channel suffixes and keep all private files outside the Git checkout:

```text
/secure/nnUNet_raw/
└── Dataset800_PD/
    ├── dataset.json
    ├── imagesTr/
    │   ├── CASE_A_0000.nii.gz
    │   ├── CASE_A_0001.nii.gz
    │   └── CASE_A_0002.nii.gz
    └── labelsTr/
        └── CASE_A.nii.gz
```

Replace the synthetic `CASE_A` example with the local case key. The same key must identify all
three channels and the corresponding label. Do not copy actual case keys into the repository.

Inspect the installed tools and the three explicit nnU-Net storage locations:

```bash
python scripts/segmentation/nnunet_pipeline.py check \
  --raw-dir /secure/nnUNet_raw \
  --preprocessed-dir /secure/nnUNet_preprocessed \
  --results-dir /secure/nnUNet_results
```

## 3. Plan and preprocess

Run dataset integrity checks and create the retained `3d_fullres` configuration:

```bash
python scripts/segmentation/nnunet_pipeline.py plan \
  --raw-dir /secure/nnUNet_raw \
  --preprocessed-dir /secure/nnUNet_preprocessed \
  --results-dir /secure/nnUNet_results \
  --dataset 800 \
  --configurations 3d_fullres \
  --verify-dataset-integrity \
  --dry-run
```

Review the printed paths and upstream command, then remove `--dry-run` to execute it. Compare the
generated plan against `configs/segmentation_fourfold.toml`, especially the patch size, target
spacing, normalization, and batch size. Stop if these values differ; do not silently force a new
plan and describe it as the recorded one.

## 4. Generate the four-fold split

Generate `splits_final.json` locally after preprocessing and before training:

```bash
python scripts/segmentation/make_fourfold_splits.py \
  --labels-dir /secure/nnUNet_raw/Dataset800_PD/labelsTr \
  --output /secure/nnUNet_preprocessed/Dataset800_PD/splits_final.json \
  --file-ending .nii.gz \
  --n-splits 4 \
  --seed 12345
```

The generator matches nnU-Net v2's default construction except for changing `n_splits` from 5 to
4. It sorts the case keys and calls scikit-learn `KFold` with shuffling and seed 12345. For 156
unique cases, each fold must contain 117 training cases and 39 validation cases; every case must
appear in validation exactly once. The script validates those invariants and prints counts without
printing the case keys.

The output is protected against accidental replacement. If a five-fold `splits_final.json` already
exists, use a clean preprocessing directory or preserve it outside the run directory first. Use
`--overwrite` only after confirming the exact target. The real split file contains case keys and is
not part of this public repository.

## 5. Train folds 0--3

First preview all four upstream commands from the TOML configuration:

```bash
python scripts/segmentation/nnunet_pipeline.py train-fourfold \
  --raw-dir /secure/nnUNet_raw \
  --preprocessed-dir /secure/nnUNet_preprocessed \
  --results-dir /secure/nnUNet_results \
  --study-config configs/segmentation_fourfold.toml \
  --dry-run
```

The configuration resolves to Dataset800_PD, `3d_fullres`, `nnUNetTrainer_100epochs`,
`nnUNetPlans`, and folds 0, 1, 2, and 3. Command-line values, when provided, override TOML values.
Remove `--dry-run` only after checking all four commands. The wrapper runs the folds sequentially
and stops at the first failure, so a failed fold cannot be hidden by later successful commands.
Before a real run, it also requires and validates `splits_final.json`: exactly four folds, disjoint
training/validation sets, one validation appearance per case, 156 total cases, and 117/39 counts.
It refuses to train if the file is absent or still contains the default five-fold split.

`train-fourfold --continue-training` is intentionally strict: every fold must already have a
resume checkpoint. To resume only one interrupted fold, use the wrapper's single-fold `train`
command with the same dataset, configuration, trainer, plans, fold number, and
`--continue-training`. Confirm that the result hierarchy contains only the intended four-fold
trainer/configuration before using it for inference.

## 6. Predict with the four-fold ensemble

Prediction inputs follow the same three-channel suffix convention but do not require `labelsTr`:

```bash
python scripts/segmentation/nnunet_pipeline.py predict-fourfold \
  --raw-dir /secure/nnUNet_raw \
  --preprocessed-dir /secure/nnUNet_preprocessed \
  --results-dir /secure/nnUNet_results \
  --input-dir /secure/inference_images \
  --output-dir /secure/predicted_masks \
  --study-config configs/segmentation_fourfold.toml \
  --dry-run
```

This fixes inference to folds 0--3 and `checkpoint_final.pth`. The retained prediction metadata
records step size 0.5 with Gaussian weighting and mirroring enabled; these values are also recorded
in the TOML file. The wrapper emits `-step_size 0.5`, retains test-time mirroring by omitting
`--disable_tta`, validates that Gaussian weighting remains enabled, and adds
`--save_probabilities`. Verify the complete printed command before removing `--dry-run`.

## Reproducibility and publication boundary

For a completed rerun, retain the following inside the approved secure environment:

- the exact audited nnU-Net source revision and a full environment lock;
- the generated plans, fingerprint, four-fold `splits_final.json`, and its SHA-256 checksum;
- sanitized command logs and per-fold validation summaries;
- checksums for each checkpoint and derived prediction artifact.

Do not commit NIfTI/DICOM data, labels, case lists, real `splits_final.json`, `.pth` checkpoints,
probability arrays, predictions, or logs containing identifiers or private server paths. Update
`segmentation/model_manifest.json` from `configuration_only_not_yet_rerun` only after all four
folds have actually completed and the recorded checks have been performed.
