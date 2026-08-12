# Code and data availability

## Manuscript-ready statement

> The classification source code, synthetic manifest example, configuration templates, and
> instructions for reproducing ViT classification, multi-slice voting, and Grad-CAM are available
> at <https://github.com/congchao120/pdmsa-multimodal-ai>. No clinical images or direct subject
> identifiers are included. The clinical imaging data cannot be publicly distributed because
> they contain potentially identifiable medical information and are governed by institutional
> ethics approval, participant consent, and institutional policy. Access to appropriately
> de-identified data may be considered upon reasonable request to the corresponding author,
> subject to approval by the responsible institution and execution of any required data-use
> agreement.

The code statement and clinical-data statement are deliberately separate. Add a versioned archive
DOI only after one has actually been issued; do not use a placeholder DOI in the submitted text.

## Checkpoints

Model weights are not tracked in Git. If an nnU-Net checkpoint is released, use a versioned GitHub
Release asset or durable research repository and provide its SHA-256 checksum, exact upstream
nnU-Net version, trainer, plans, configuration, fold, `dataset.json`, and inference instructions.
A `.pth` file without these companion records is not sufficient for reproducible inference.

Checkpoint bundles must not contain clinical images, direct identifiers, DICOM headers, server
paths, or a subject-level split file that exposes local identifiers.
