# nnU-Net segmentation reproducibility

Segmentation is a separate upstream stage. This repository does not vendor nnU-Net or claim a
study-specific modification. Instead, `scripts/segmentation/nnunet_pipeline.py` is a thin wrapper
around the official nnU-Net v2 command-line interface. Classification consumes the predicted ROI
outputs from this stage.

## Confirm the original environment first

The retained source tree's `setup.py` reports version `2.1.1`. That string alone is not enough to
prove which installed package, Git commit, trainer, plans, or configuration generated a particular
checkpoint. Before publishing commands or weights, retrieve these values from the training server:

- `python`, PyTorch, CUDA, `nnunetv2`, and operating-system versions;
- upstream nnU-Net Git commit, if installed from source;
- dataset ID/name plus the exact channel and segmentation-label mappings in `dataset.json`;
- trainer class, plans identifier, configuration (`2d`, `3d_fullres`, etc.), and trained folds;
- `plans.json`, dataset fingerprint, split definition, post-processing settings, and checkpoint;
- the exact plan, train, and predict command lines and the checksum of every released artifact.

Record these in `segmentation/model_manifest.json`. Only after this verification should
`requirements/segmentation.txt` be pinned. Do not substitute a nearby package version merely
because its number looks compatible.

## Wrapper usage

All commands require explicit raw, preprocessed, and results directories. This avoids silently
depending on environment variables inherited from an unrelated server session.

First inspect the installation:

```bash
python scripts/segmentation/nnunet_pipeline.py check \
  --raw-dir /path/to/nnUNet_raw \
  --preprocessed-dir /path/to/nnUNet_preprocessed \
  --results-dir /path/to/nnUNet_results
```

Plan and preprocess a verified dataset:

```bash
python scripts/segmentation/nnunet_pipeline.py plan \
  --raw-dir /path/to/nnUNet_raw \
  --preprocessed-dir /path/to/nnUNet_preprocessed \
  --results-dir /path/to/nnUNet_results \
  --dataset XXX \
  --configurations 2d 3d_fullres \
  --plans VERIFIED_PLANS_NAME \
  --planner VERIFIED_PLANNER_CLASS \
  --verify-dataset-integrity \
  --dry-run
```

Train one verified fold:

```bash
python scripts/segmentation/nnunet_pipeline.py train \
  --raw-dir /path/to/nnUNet_raw \
  --preprocessed-dir /path/to/nnUNet_preprocessed \
  --results-dir /path/to/nnUNet_results \
  --dataset DatasetXXX_Name \
  --configuration VERIFIED_CONFIGURATION \
  --fold 0 \
  --trainer VERIFIED_TRAINER \
  --plans VERIFIED_PLANS_NAME \
  --dry-run
```

Run inference from the official results hierarchy:

```bash
python scripts/segmentation/nnunet_pipeline.py predict \
  --raw-dir /path/to/nnUNet_raw \
  --preprocessed-dir /path/to/nnUNet_preprocessed \
  --results-dir /path/to/nnUNet_results \
  --input-dir /path/to/imagesTs \
  --output-dir /path/to/predicted_masks \
  --dataset DatasetXXX_Name \
  --configuration VERIFIED_CONFIGURATION \
  --folds 0 1 2 3 \
  --trainer VERIFIED_TRAINER \
  --plans VERIFIED_PLANS_NAME \
  --checkpoint-name checkpoint_final.pth \
  --dry-run
```

Remove `--dry-run` only after the printed command and environment have been checked. Arguments
whose upstream defaults were actually used should be omitted rather than filled with guesses.

## Model release

Keep model binaries out of ordinary Git history. Upload a ZIP or TAR archive of the verified
nnU-Net results hierarchy as a GitHub Release asset or DOI-backed archive, then publish its
SHA-256 checksum in both the release notes and `segmentation/model_manifest.json`. Verify a
download with the wrapper's `check --artifact ... --expected-sha256 ...` options.

A standalone `.pth` file is not a reproducible model release. It must be accompanied by its plans,
dataset/channel/label metadata, trainer/configuration/fold identifiers, upstream revision, exact
commands, and checksums. Never include raw scans, subject identifiers, or protected health
information in the archive.

For end-to-end evaluation, classification of held-out subjects must use predicted masks produced
without their manual labels. Any manual correction or use of `labelsTr` for classification cases
must be explicitly disclosed and evaluated as a different workflow.
