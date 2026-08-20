from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int, augmentation: dict[str, Any], training: bool):
    """Build preprocessing for the shared 384-pixel ViT.

    Every pre-fused RGB slice is resized to the configured square input and
    ImageNet-normalized. Augmentation remains an explicitly enabled option and
    is disabled in the article-aligned configuration.
    """
    from torchvision import transforms

    operations: list[Any] = [transforms.Resize((image_size, image_size), antialias=True)]
    if training and bool(augmentation.get("enabled", False)):
        operations.append(
            transforms.RandomAffine(
                degrees=float(augmentation.get("rotation_degrees", 5.0)),
                translate=(
                    float(augmentation.get("translation_fraction", 0.02)),
                    float(augmentation.get("translation_fraction", 0.02)),
                ),
                scale=(
                    float(augmentation.get("scale_min", 0.95)),
                    float(augmentation.get("scale_max", 1.05)),
                ),
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=0,
            )
        )
        flip_probability = float(augmentation.get("horizontal_flip_probability", 0.0))
        if flip_probability > 0:
            operations.append(transforms.RandomHorizontalFlip(p=flip_probability))
        brightness = float(augmentation.get("brightness", 0.0))
        contrast = float(augmentation.get("contrast", 0.0))
        if brightness > 0 or contrast > 0:
            operations.append(
                transforms.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=0,
                    hue=0,
                )
            )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transforms.Compose(operations)


class FusedRGBSliceDataset:
    """Load one pre-fused RGB PNG for each subject and slice.

    Fusion is intentionally not repeated inside the classifier. The source
    scripts consumed already prepared three-channel PNG files, where channel
    meaning depends on the modality-combination experiment.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        data_root: str | Path,
        image_column: str = "fused_path",
        transform=None,
    ) -> None:
        if image_column not in frame.columns:
            raise ValueError(f"Frame is missing image column {image_column!r}")
        self.frame = frame.reset_index(drop=True).copy()
        self.data_root = Path(data_root).expanduser().resolve()
        self.image_column = str(image_column)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def _resolve(self, value: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.data_root / path
        return path.resolve()

    def __getitem__(self, index: int):
        import torch

        record = self.frame.iloc[index]
        path = self._resolve(str(record[self.image_column]))
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        else:
            from torchvision.transforms.functional import pil_to_tensor

            image = pil_to_tensor(image).float() / 255.0

        label = torch.tensor(int(record["label"]), dtype=torch.long)
        metadata = {
            "subject_id": str(record["subject_id"]),
            "center_slice_index": int(record["center_slice_index"]),
            "slice_offset": int(record["slice_offset"]),
            "slice_index": int(record["slice_index"]),
            # Do not propagate private server roots into prediction tables.
            "input_file": Path(str(record[self.image_column])).name,
        }
        return image, label, metadata


class MultimodalSliceDataset(FusedRGBSliceDataset):
    """Backward-compatible name for callers migrating from the first release.

    New code should use :class:`FusedRGBSliceDataset`. This adapter deliberately
    accepts only a single path column; on-the-fly modality stacking is not the
    behavior of the retained source classification scripts.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        channels: list[str],
        data_root: str | Path,
        image_size: int | None = None,
        normalization: str | None = None,
        transform=None,
    ) -> None:
        if len(channels) != 1:
            raise ValueError(
                "The source-aligned classifier expects one pre-fused RGB path column"
            )
        super().__init__(frame, data_root, image_column=channels[0], transform=transform)
