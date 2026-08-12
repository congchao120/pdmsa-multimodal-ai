# nnU-Net model artifacts

This directory contains only the publication metadata template. The study used the official
nnU-Net v2 implementation without a study-specific fork, so its source is not duplicated here.
The thin wrapper at `scripts/segmentation/nnunet_pipeline.py` invokes the official commands.

## What to publish

Publish the trained model as a versioned GitHub Release asset (or a DOI-backed archive) rather
than committing a large `.pth` file to Git. Preserve the official results hierarchy, for example:

```text
DatasetXXX_Name/
└── Trainer__Plans__Configuration/
    ├── dataset.json
    ├── plans.json
    └── fold_0/
        └── checkpoint_final.pth
```

Include every trained fold used by inference. Also include `model_manifest.json`, the verified
split definition, and any post-processing configuration. Fill every `TO_BE_FILLED` value from
the original training server; do not infer these fields from the manuscript.

A `.pth` file by itself is insufficient: nnU-Net prediction also depends on the dataset ID and
channel/label mapping, plans, trainer class, configuration, fold(s), checkpoint name, upstream
software revision, and preprocessing/post-processing contract.

## Integrity and privacy

Generate SHA-256 checksums after creating the final archive and record both the archive checksum
and per-file checksums in `model_manifest.json`. The wrapper can calculate and verify a checksum:

```bash
python scripts/segmentation/nnunet_pipeline.py check \
  --raw-dir /path/to/nnUNet_raw \
  --preprocessed-dir /path/to/nnUNet_preprocessed \
  --results-dir /path/to/nnUNet_results \
  --artifact /path/to/release-asset.zip \
  --expected-sha256 64_HEXADECIMAL_CHARACTERS
```

Do not release DICOM data, original filenames, subject IDs, filesystem paths, logs containing
identifiers, or any other protected health information. Model publication should be reviewed
under the study's ethics and data-governance requirements.
