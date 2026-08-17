# nnU-Net four-fold segmentation recipe

This directory contains metadata for a reproducible **four-fold retraining recipe**. The public
repository intentionally excludes images, manual segmentation labels, real subject identifiers,
the generated `splits_final.json`, logs, predictions, and model weights.

The recipe uses Dataset 800 (`Dataset800_PD`), the `3d_fullres` configuration, `nnUNetPlans`, and
folds 0--3. Its three channels are FLAIR, T1w, and T2w. The retained dataset metadata names the
four foreground classes only as `Region 1` through `Region 4`; anatomical names must not be
inferred until they are independently verified.

The retained training logs report 100 epochs, while the base trainer in the retained source tree
currently defaults to 200. The public recipe therefore selects the existing nnU-Net trainer
variant `nnUNetTrainer_100epochs` explicitly. No custom neural-network implementation is needed.

`model_manifest.json` records the machine-readable recipe and publication status. Full setup,
split-generation, training, and inference commands are in `docs/SEGMENTATION.md`.

## Scope and provenance

The downloaded result directory contains folds 0--4. Those five-fold checkpoints are not outputs
of this four-fold recipe and must not be renamed as folds 0--3. The public setup is a rerunnable
protocol; until all four folds have actually been trained and evaluated, it must not be presented
as the source of reported segmentation or classification results.

## Privacy

Generate `splits_final.json` locally from the authorized dataset. The generator prints counts but
not case IDs. Before publishing any later artifact, remove subject identifiers, private filesystem
paths, logs containing identifiers, raw scans, labels, and predictions, and complete the required
ethics and data-governance review.
