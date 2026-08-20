# Model card

## Intended use

Research-only differentiation of PD and MSA-P in the study population. The model is not a medical
device and must not be used as a stand-alone clinical diagnostic system.

## Inputs

Five axial striatal-ROI slices per subject, dynamically selected at relative positions
`[-2, -1, 0, 1, 2]` around that subject's largest-ROI-area center. Each classifier input is an
already fused 384 x 384 RGB PNG, for example an FDG+CFT+T2WI pseudo-color image. Its
RGB-to-modality mapping must be recorded during preprocessing and is not inferred by the
classifier. One shared ViT checkpoint per fold is applied to every relative slice position.

## Outputs

Slice-level class scores and a patient-level aggregate score. Weighted MSV uses fixed relative
position weights `[0.10, 0.20, 0.40, 0.20, 0.10]`. The positive disease class, voting method, and
threshold must be stated in each run configuration and publication table.

## Known limitations

- single-center, imbalanced cohort;
- limited number of MSA-P subjects;
- dorsal-striatal ROI restriction;
- missing modalities require a separately trained compatible model;
- no external validation in the current release;
- saliency maps describe model attribution, not causal pathology.

## Prohibited claims

Do not claim that augmentation creates independent patients, that four-fold validation solves class
imbalance, that MSV is an attention mechanism, or that the full three-channel model accepts an
unseen missing channel without retraining.
