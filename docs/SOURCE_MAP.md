# Retained source-code map

The original `sourcecode` directory is retained outside this public project for audit. It should
not be copied wholesale because it contains exploratory scripts and private server paths.

The public classifier removes private paths and implements the clarified shared-model design:
all five dynamically centered slices in one fold use one 384-pixel ViT and one checkpoint before
fixed-weight patient-level MSV. Exact configuration and checkpoint provenance must accompany any
numerical reproduction claim.
