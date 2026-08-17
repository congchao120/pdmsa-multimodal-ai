# Methods-to-code alignment

This file separates confirmed study facts, retained source-code evidence, and details that still
require author confirmation. Complete the unresolved acquisition/registration fields when the
original server record becomes available.

## Confirmed by the authors

- Standard space family: MNI152, not MNI305.
- Software family: SPM.
- Atlas: the Neuromorphometrics label atlas distributed with SPM.
- Regions of interest: bilateral caudate nuclei and putamina.
- Classification input: five axial slices centered on the largest segmented ROI area.

## Required MNI/SPM details

"MNI152 using SPM and Neuromorphometrics" identifies the general framework but is not a
complete reproducible specification. Record all of the following:

1. SPM release and update revision, MATLAB version, and operating system.
2. Exact standard-space reference file. For example, state whether the registration target was
   `spm12/canonical/avg152T1.nii`, an SPM tissue-probability map such as `TPM.nii`, or another
   MNI152 image. These files are not interchangeable descriptions.
3. Exact atlas files and checksum, normally
   `spm12/tpm/labels_Neuromorphometrics.nii` with its XML label definition.
4. Reference subject modality (the manuscript currently indicates T1-weighted MRI).
5. Transform chain and direction: rigid six-degree-of-freedom initialization, 12-DOF affine,
   and any nonlinear deformation; specify which transform was inverted to map atlas labels to
   native T1 space.
6. SPM batch module/function and key settings, including objective/cost function, voxel size,
   bounding box, regularization, and whether estimation and writing were separate steps.
7. Interpolation: nearest-neighbor for discrete atlas labels; the stated interpolation for MRI
   and PET intensity images.
8. How PET and the other MRI sequences were aligned to T1, including acquisition order and
   whether all modalities were resampled to a single grid before fusion.
9. Output grid dimensions, voxel sizes, orientation convention, and affine/header handling.
10. Registration quality control, failure/repeat criteria, and the number of cases requiring
    manual intervention.

Suggested minimum Methods wording after the details are verified:

> T1-weighted MRI served as the within-subject reference. Images were aligned using SPM12
> (revision [TO VERIFY]) and normalized with respect to the MNI152 reference image [exact file,
> voxel size]. The SPM Neuromorphometrics maximum-probability atlas
> (`tpm/labels_Neuromorphometrics.nii`, [checksum]) was used to define the bilateral caudate
> nuclei and putamina. The estimated [rigid/affine/nonlinear] transform was inverted to map
> labels into native T1 space. Continuous-valued images were resampled using [interpolation],
> whereas discrete labels were resampled using nearest-neighbor interpolation. Registration
> quality was assessed by [procedure].

## ViT configuration still to verify

- The manuscript describes 384 x 384 images, 16 x 16 patches, and 150 epochs.
- Retained `vit0.py` points to a patch-32/384 model and 100 epochs.
- Retained `vit5.py`-`vit8.py` point to a patch-16/224 model and 100 epochs and appear more closely
  connected to the voting experiments.

The runnable `configs/classification.toml` therefore uses the settings in the layer-specific
patch-16/224 scripts. This is a source-code default, not proof that the submitted table came from
that checkpoint. Update the configuration and Methods together after checking the final model
directory, training log, or checkpoint metadata.

## nnU-Net four-fold recipe

The public segmentation workflow generates a four-fold `splits_final.json` (folds 0--3,
seed 12345) and uses the built-in `nnUNetTrainer_100epochs` variant. The retained result
directory contains an earlier five-fold model; those checkpoints are not redistributed or
relabeled. The four-fold configuration is a retraining recipe and must not be cited as having
produced an existing result until all four folds are actually rerun and archived.

## MSV definition

ViT self-attention operates within a slice. MSV is a separate patient-level aggregation step and
must not be described as attention. This release supports:

- soft voting: the mean of five slice-level positive-class scores;
- hard voting: the fraction of five binary slice decisions;
- fixed weighted soft voting: a configured weighted sum of five slice probabilities.

For hard voting, each slice decision is the two-class argmax; `patient_threshold` is applied only
to the resulting positive-vote fraction. The retained fixed-weight script used weights
`[0.05, 0.05, 0.80, 0.05, 0.05]` for layers 6 through 10. A separate exploratory source script
searched weights on the supplied labelled data; that search is not run by the consolidated
pipeline because tuning on the reported held-out fold would bias performance.
