# Multimodal PET/MRI classification of PD and MSA-P

This repository organizes the classification code used to distinguish Parkinson's disease
(PD) from the parkinsonian subtype of multiple system atrophy (MSA-P). It was consolidated
from the study `sourcecode` directory into a reproducible command-line project; it is not a
byte-for-byte archive of every exploratory script.

The classification input is the **already fused RGB PNG** generated during preprocessing.
For each subject, the axial slice with the largest segmented striatal ROI is selected dynamically,
and its two real neighbors on each side form a five-slice window. One shared Vision Transformer
(ViT) is trained per cross-validation fold on all five slices from every training patient. The same
fold checkpoint produces all held-out slice probabilities, which are then combined by multi-slice
voting (MSV).


## Pipeline scope

The release contains:

- manifest-based pairing by pseudonymous subject ID and slice index;
- patient-level four-fold split generation and leakage checks;
- one shared ViT classifier per fold, trained on every selected slice;
- configurable weighted MSV and patient-level metrics;
- optional training-only augmentation and two optional class-imbalance strategies;
- Grad-CAM export for an individual slice processed by the fold-shared model;
- synthetic manifest records and unit tests (no clinical images).

MSV is a **decision-level aggregation method**, not an attention mechanism. Self-attention is
part of each ViT model. Grad-CAM is a gradient-based class-activation method and is likewise
different from attention. A slice can therefore have both a class probability and a Grad-CAM
map without either quantity being an "MSV attention score."

Atlas-guided striatal segmentation is a separate preprocessing stage. The study used MNI152
standard space with SPM and the Neuromorphometrics atlas. This repository includes a true
four-fold nnU-Net retraining recipe, a deterministic split generator, and guarded batch
training/inference commands. It does not copy the upstream nnU-Net source or distribute scans,
labels, real case assignments, or weights; see [the segmentation guide](docs/SEGMENTATION.md).

## Labels and input contract

The default label convention follows the classification source code:

- `0 = MSA-P`
- `1 = PD`

With `positive_label = 1`, sensitivity is PD recall and specificity is the true-negative rate
for MSA-P. Changing the positive label changes the clinical interpretation of these metrics and
must be reflected consistently in configurations, tables, and figures.

The classification manifest has one row per subject and selected slice:

```csv
subject_id,label,center_slice_index,slice_offset,slice_index,fused_path
SUBJ0001,1,18,-2,16,relative/path/SUBJ0001_slice016.png
SUBJ0001,1,18,-1,17,relative/path/SUBJ0001_slice017.png
SUBJ0001,1,18,0,18,relative/path/SUBJ0001_slice018.png
```

All slice indices are zero-based indices on the preprocessed canonical axial grid. Every subject
must have offsets `-2, -1, 0, 1, 2`, and `slice_index` must equal
`center_slice_index + slice_offset`. The file name is treated as an opaque path and is never used
to infer an index.

`pdmsa.roi.centered_slice_metadata` identifies the center by maximum nonzero striatal-mask area
along axis 2 and returns manifest-ready metadata for the exact center-minus-two through
center-plus-two window. The underlying `select_five_slices` helper performs the same selection.
The mask must already be reoriented to the study's canonical axial orientation. A maximum too
close to the volume edge is rejected rather than shifted or duplicated, and tied maxima use the
lowest zero-based index.

`fused_path` points to a three-channel RGB PNG prepared before classification. The RGB channel
order represents the chosen three-modality combination and must match the preprocessing record
for that experiment. The classifier does not search server directories, infer subject identity
from filenames, or fuse raw MRI/PET volumes at training time.

Use only immutable pseudonyms in the public manifest. Names, medical-record numbers, accession
numbers, acquisition dates, DICOM headers, and clinical images must remain outside the repository.
See [data/README.md](data/README.md).

## Installation

The manuscript's retained software record states Python 3.9, PyTorch 1.13.1, and CUDA 11.6.
The reorganized package is tested through CI on Python 3.10 and 3.11; its dependency metadata
therefore requires Python 3.10 or newer. PyTorch installation depends on the operating system,
GPU, CUDA runtime, and driver, so it is intentionally not forced by the generic requirements file.
Before describing a rerun as an exact reproduction, recover the original environment freeze and
confirm the environment against its checkpoint rather than treating either record as verified.

1. Create and activate an environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   python -m pip install --upgrade pip
   ```

2. Install `torch` and `torchvision` using the command generated by the
   [official PyTorch selector](https://pytorch.org/get-started/locally/), then verify the device:

   ```bash
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   ```

3. Install the classification package and its non-PyTorch dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Install optional Grad-CAM or development dependencies only when needed:

   ```bash
   python -m pip install -r requirements/explainability.txt
   python -m pip install -r requirements/dev.txt
   ```

The files contain supported version ranges because the original environment lock was not
recoverable. For a released result, save `python -m pip freeze`, the CUDA/cuDNN versions, GPU
model, configuration file, and Git commit alongside the checkpoint.

nnU-Net is intentionally absent from these requirements. Install the official nnU-Net package
in a separate segmentation environment and preserve the exact upstream version with the model
artifacts; do not assume an unavailable `nnunetv2==2.1.1` PyPI pin.

For the separate segmentation environment and the verified installation procedure, use
`requirements/segmentation.txt` together with [the segmentation guide](docs/SEGMENTATION.md).

## Classification workflow

The runnable configuration is [configs/classification.toml](configs/classification.toml).
Paths are resolved relative to that file unless documented otherwise.

The manuscript-aligned public configuration uses `google/vit-base-patch16-384`, 384-pixel inputs,
150 epochs, and a batch size of 32. Within a fold, training and validation datasets contain all
five relative slice positions for their assigned patients; the model is initialized and
checkpointed only once. The pretrained 1000-class ImageNet head is replaced by a newly initialized
two-class PD/MSA-P head, while the compatible pretrained ViT backbone weights are loaded. A
reported numerical result should archive this shared checkpoint and its exact run provenance.

1. Audit the manifest and, when images are locally available, verify every path:

   ```bash
   pdmsa-audit-manifest --config configs/classification.toml --check-files
   ```

2. Create the frozen patient-level four-fold assignments once:

   ```bash
   pdmsa-make-splits --config configs/classification.toml
   ```

3. Train one shared ViT for each fold:

   ```bash
   pdmsa-train-fold --config configs/classification.toml --fold 0
   pdmsa-train-fold --config configs/classification.toml --fold 1
   pdmsa-train-fold --config configs/classification.toml --fold 2
   pdmsa-train-fold --config configs/classification.toml --fold 3
   ```

4. Aggregate five probabilities per subject with MSV, or evaluate pooled out-of-fold results:

   ```bash
   pdmsa-aggregate-msv --input outputs/oof_slice_predictions.csv \
     --output outputs/oof_subject_predictions.csv \
     --method weighted_soft --weights 0.10 0.20 0.40 0.20 0.10 \
     --positive-label 1

   pdmsa-evaluate-oof --input outputs/oof_slice_predictions.csv \
     --output-dir outputs/oof_evaluation --expected-subjects 155 \
     --method weighted_soft --weights 0.10 0.20 0.40 0.20 0.10 \
     --positive-label 1
   ```

The weights follow relative offsets `-2, -1, 0, 1, 2`; they therefore remain aligned even when
patients have different absolute center indices. If weights are
tuned, tuning must use training/validation data only; selecting weights on the final held-out
predictions would bias the reported estimate.

## Grad-CAM

Install the explainability requirements, then run Grad-CAM against the fold-shared checkpoint and
any corresponding fused PNG from that fold:

```bash
pdmsa-gradcam --config configs/classification.toml \
  --checkpoint outputs/fdg_cft_t2wi/fold_0/best_model_weights.pth \
  --input data/private/SUBJ0001_slice018.png \
  --output-dir outputs/gradcam/SUBJ0001_slice018 \
  --target-class 0 --target-layer 8
```

If target arguments are omitted, the command uses the documented configuration defaults. Store
the raw activation map, overlay, class probabilities, target class/layer, checkpoint hash, and
input identifier together. The same fold checkpoint is used for every selected slice. A Grad-CAM
image explains one classifier decision; it is not a
probability assigned by MSV to that slice.

## nnU-Net four-fold segmentation

The runnable methods record is
[configs/segmentation_fourfold.toml](configs/segmentation_fourfold.toml). It fixes folds 0--3,
the nnU-Net-compatible split seed, `3d_fullres` plans, and the built-in
`nnUNetTrainer_150epochs` variant. Generate the private `splits_final.json` locally, preview all
commands, and then train:

```bash
python scripts/segmentation/make_fourfold_splits.py \
  --labels-dir /secure/nnUNet_raw/Dataset800_PD/labelsTr \
  --output /secure/nnUNet_preprocessed/Dataset800_PD/splits_final.json

python scripts/segmentation/nnunet_pipeline.py train-fourfold \
  --raw-dir /secure/nnUNet_raw \
  --preprocessed-dir /secure/nnUNet_preprocessed \
  --results-dir /secure/nnUNet_results \
  --study-config configs/segmentation_fourfold.toml \
  --dry-run
```


## Class imbalance options

The configuration supports `none`, `weighted_sampler`, `class_weighted_loss`, and `both`.
Weights are estimated from training subjects only. Combining a weighted sampler with a weighted
loss can over-correct the minority class, so `both` is intended for an explicitly labelled
sensitivity analysis.

Augmentation and class weighting cannot create independent subjects or eliminate the uncertainty
caused by a small MSA-P cohort. Four-fold cross-validation makes efficient use of available
subjects; patient-level splitting, balanced metrics, and confidence intervals remain necessary.

## Repository layout

```text
configs/                 Classification, registration, and segmentation configurations
data/                    Synthetic manifest plus private-data instructions
docs/                    Methods, availability, segmentation, and release notes
requirements/            Classification, explainability, and development dependencies
scripts/                 Standalone preprocessing and guarded nnU-Net helpers
segmentation/            Privacy-safe metadata and dataset.json template (no weights)
src/pdmsa/               Classification, MSV, metrics, and Grad-CAM implementation
tests/                   Unit tests and leakage checks
```

## Code, data, weights, and citation

Clinical images and direct identifiers are not distributed. The intended public repository is
<https://github.com/congchao120/pdmsa-multimodal-ai>; see
[the availability statement](docs/DATA_AVAILABILITY.md) for the manuscript wording.

No nnU-Net checkpoint is distributed in this release. If weights are published later, use a
versioned release or durable research archive rather than ordinary Git, and include checksums,
the nnU-Net revision, trainer, plans, configuration, folds, and inference metadata. A standalone
`.pth` file is not a reproducible model release. Never include training images or identifiers.

Citation metadata are provided in [CITATION.cff](CITATION.cff). No software license has yet been
selected in this release candidate; the copyright holders and institutions must approve a license
before public release. Public visibility by itself does not grant reuse rights.
