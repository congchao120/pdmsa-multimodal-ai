from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from PIL import Image


def load_grayscale(
    path: str | Path,
    size: int | tuple[int, int] | None = None,
) -> np.ndarray:
    """Load one source image as the 8-bit plane used in pseudo-color fusion."""
    with Image.open(path) as loaded:
        image = loaded.convert("L")
    if size is not None:
        target = (size, size) if isinstance(size, int) else tuple(size)
        image = image.resize(target, resample=Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def fuse_channels(
    paths: Iterable[str | Path],
    size: int | tuple[int, int] | None = None,
) -> Image.Image:
    """Stack three explicitly ordered grayscale planes into one RGB PNG.

    ``paths`` is ordered as red, green, and blue. Pixel intensities are copied
    directly without per-image renormalization. The caller must record which modality occupies
    each channel. Explicit paths replace unsafe positional ``os.listdir`` pairing.
    """
    path_list = list(paths)
    if len(path_list) != 3:
        raise ValueError(f"Exactly three channel paths are required, found {len(path_list)}")
    arrays = [load_grayscale(path, size=size) for path in path_list]
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        raise ValueError(f"Channel shapes do not match: {sorted(shapes)}")
    return Image.fromarray(np.stack(arrays, axis=-1), mode="RGB")
