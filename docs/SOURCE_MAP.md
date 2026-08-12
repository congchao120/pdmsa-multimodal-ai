# Retained source-code map

The original `sourcecode` directory is retained outside this public project for audit. It should
not be copied wholesale because it contains exploratory scripts and private server paths.

| Study stage | Retained source evidence | Consolidated implementation |
|---|---|---|
| Four-fold split | `固定划分四折二.py` | `pdmsa.splits`, with subject-level assignment and full coverage checks |
| Three-channel fusion | `三合一图像.py` | Classifier reads the resulting pre-fused RGB PNG through `fused_path` |
| ViT training | `vit0.py`–`vit8.py` | One independent ViT per layer and fold in `pdmsa.training` |
| Five-slice voting | `投票/投票3.0.py`, `投票5.0.py` | Fixed weighted, soft, and hard MSV in `pdmsa.voting` |
| Segmentation | copied upstream `nnunetv2/` tree | Official nnU-Net plus the thin runner in `scripts/segmentation/` |
| Explainability | `vit33.py` | Corrected Hugging Face ViT Grad-CAM with auditable artifacts |
| Metrics | several one-off scripts | Explicit positive class, patient-level metrics, and bootstrap CIs |

The consolidation removes hard-coded paths and incomplete debugging code. Data augmentation,
weighted sampling, and class-weighted cross-entropy are optional additions and are disabled in the
source-aligned configuration. The final 224-versus-384 checkpoint and exact environment still need
to be confirmed from the original server before claiming exact numerical reproduction.
