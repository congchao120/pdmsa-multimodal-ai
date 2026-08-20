# Methods-to-code alignment

This file separates confirmed study facts, retained source-code evidence, and details that still
require author confirmation. Complete the unresolved acquisition/registration fields when the
original server record becomes available.

## Confirmed by the authors

- Standard space family: MNI152, not MNI305.
- Software family: SPM.
- Atlas: the Neuromorphometrics label atlas distributed with SPM.
- Regions of interest: bilateral caudate nuclei and putamina.
- Classification input: five axial slices dynamically centered on the largest segmented ROI area.
- Slice positions: relative offsets `[-2, -1, 0, 1, 2]`, recorded with zero-based indices.
- Classifier input size: 384 x 384 with 16 x 16 patches.
- Classifier scope: one shared ViT per fold for all five slice positions.

## ViT classification configuration

- Fused RGB slices are resized to 384 x 384 and divided into 16 x 16 patches.
- Each fold initializes one ViT. Its training dataset contains all five selected slices from every
  training patient; its held-out dataset contains all five slices from every validation patient.
- One fold checkpoint produces all slice-level probabilities before patient-level aggregation.
- The runnable configuration records 150 epochs and batch size 32.

A released numerical result must archive the shared checkpoint, exact configuration, fold
assignment hash, and complete held-out slice predictions.

## MSV definition

ViT self-attention operates within a slice. MSV is a separate patient-level aggregation step and
must not be described as attention. This release supports:

- soft voting: the mean of five slice-level positive-class scores;
- hard voting: the fraction of five binary slice decisions;
- fixed weighted soft voting: a configured weighted sum of five slice probabilities.

For hard voting, each slice decision is the two-class argmax; `patient_threshold` is applied only
to the resulting positive-vote fraction. The article-aligned fixed weights are
`[0.10, 0.20, 0.40, 0.20, 0.10]`, ordered by relative offsets `[-2, -1, 0, 1, 2]`. They are fixed
in configuration and are never estimated from held-out labels by the consolidated pipeline.
