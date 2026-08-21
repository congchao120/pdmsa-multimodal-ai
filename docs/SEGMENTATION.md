# nnU-Net four-fold segmentation reproducibility

Segmentation is a separate upstream stage that produces ROI masks for the classification
pipeline. This repository provides an auditable four-fold nnU-Net v2 retraining recipe; it does
not distribute patient data, manual labels, real split assignments, predictions, logs, or model
weights.

## Recorded configuration

The executable methods record is `configs/segmentation_fourfold.toml`. The main settings are:

| Item | Value |
| --- | --- |
| Dataset | `Dataset800_PD` (ID 800), 155 training cases |
| Channels | `0000`: FLAIR, `0001`: T1w, `0002`: T2w |
| Foreground labels | Generic `Region 1`--`Region 4` (values 1--4) |
| Configuration | `3d_fullres`, `nnUNetPlans`, `PlainConvUNet` |
| Trainer | `nnUNetTrainer_150epochs` |
| Cross-validation | folds 0--3; 116 training and 39 validation cases per fold |
| Split algorithm | sorted case keys + stratified `KFold(n_splits=4, shuffle=True, random_state=12345)` |
| Patch / batch | `[20, 320, 256]` / 32 |

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

Confirm that the existing 150-epoch trainer variant is importable:

```bash
python -c "from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import nnUNetTrainer_150epochs"
```

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

