# Model card

## Intended use

Research-only differentiation of PD and MSA-P in the study population. The model is not a medical
device and must not be used as a stand-alone clinical diagnostic system.

## Inputs

Five axial, striatal ROI slices (indices 6 through 10) per subject. Each classifier input is an
already fused RGB PNG, for example an FDG+CFT+T2WI pseudo-color image. Its RGB-to-modality mapping
must be recorded during preprocessing and is not inferred by the classifier.

## Outputs

Slice-level class scores and a patient-level aggregate score. The positive disease class, voting
method, and threshold must be stated in each run configuration and publication table.

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
