# nnU-Net four-fold segmentation

The segmentation workflow produces ROI masks for slice selection and classification. It uses
three MRI channels (FLAIR, T1w, and T2w), the nnU-Net v2 `3d_fullres` configuration, and four
cross-validation folds. The executable settings are stored in
[`configs/segmentation_fourfold.toml`](../configs/segmentation_fourfold.toml).

## Install

Create a dedicated environment, install the PyTorch build appropriate for the local hardware,
and install the segmentation dependencies:

```bash
python -m venv .venv-nnunet
source .venv-nnunet/bin/activate          # Windows: .venv-nnunet\Scripts\activate
python -m pip install --upgrade pip
# Install torch using https://pytorch.org/get-started/locally/
python -m pip install -r requirements/segmentation.txt
```

Confirm that the configured trainer is available:

```bash
python -c "from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import nnUNetTrainer_100epochs"
```

## Dataset layout

Keep the nnU-Net runtime directories outside the Git checkout:

```text
/secure/nnUNet_raw/
└── Dataset800_PD/
    ├── dataset.json
    ├── imagesTr/
    │   ├── CASE_A_0000.nii.gz    # FLAIR
    │   ├── CASE_A_0001.nii.gz    # T1w
    │   └── CASE_A_0002.nii.gz    # T2w
    └── labelsTr/
        └── CASE_A.nii.gz
```

The matching metadata template is
[`segmentation/templates/Dataset800_PD/dataset.json`](../segmentation/templates/Dataset800_PD/dataset.json).
Use the same pseudonymous case key for all channels and the corresponding label.

Inspect the nnU-Net installation and runtime directories:

```bash
python scripts/segmentation/nnunet_pipeline.py check \
  --raw-dir /secure/nnUNet_raw \
  --preprocessed-dir /secure/nnUNet_preprocessed \
  --results-dir /secure/nnUNet_results
```

## Plan and preprocess

Preview the command first:

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

Remove `--dry-run` after checking the displayed paths and command.

## Generate four folds

The split generator sorts the case keys and applies shuffled `KFold` with seed `12345`. It reads
filenames only and writes nnU-Net's `splits_final.json` format. For 155 cases, each fold contains
116/117 training cases and 38/39 validation cases.

```bash
python scripts/segmentation/make_fourfold_splits.py \
  --labels-dir /secure/nnUNet_raw/Dataset800_PD/labelsTr \
  --output /secure/nnUNet_preprocessed/Dataset800_PD/splits_final.json \
  --file-ending .nii.gz \
  --n-splits 4 \
  --seed 12345
```

The training wrapper validates fold coverage, train/validation separation, validation frequency,
and configured case counts before invoking nnU-Net.

## Train folds 0 through 3

Preview all four commands:

```bash
python scripts/segmentation/nnunet_pipeline.py train-fourfold \
  --raw-dir /secure/nnUNet_raw \
  --preprocessed-dir /secure/nnUNet_preprocessed \
  --results-dir /secure/nnUNet_results \
  --study-config configs/segmentation_fourfold.toml \
  --dry-run
```

Remove `--dry-run` to train. The configuration selects `nnUNetTrainer_100epochs`, `nnUNetPlans`,
`3d_fullres`, and folds 0 through 3.

## Predict with the four-fold ensemble

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

The ensemble uses folds 0 through 3 and `checkpoint_final.pth`. Remove `--dry-run` after checking
the complete command.

## Data safety

Do not commit `nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results`, `splits_final.json`, medical
images, labels, predictions, logs, or checkpoints. The repository ignore rules exclude these
artifacts by default.
