from __future__ import annotations

import numpy as np


def select_five_slices(mask: np.ndarray, axis: int = 2) -> list[int]:
    """Select the largest-ROI slice and a boundary-safe contiguous five-slice window."""
    array = np.asarray(mask)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D mask, found shape {array.shape}")
    binary = array != 0
    if not binary.any():
        raise ValueError("Cannot select slices from an empty segmentation mask")
    if array.shape[axis] < 5:
        raise ValueError("The selected axis contains fewer than five slices")

    reduction_axes = tuple(index for index in range(3) if index != axis)
    areas = binary.sum(axis=reduction_axes)
    center = int(np.argmax(areas))
    start = max(0, center - 2)
    start = min(start, array.shape[axis] - 5)
    return list(range(start, start + 5))


def bounding_box_2d(mask_slice: np.ndarray, padding: int = 5) -> tuple[int, int, int, int]:
    """Return a clipped `(row_start, row_end, col_start, col_end)` box."""
    binary = np.asarray(mask_slice) != 0
    if binary.ndim != 2:
        raise ValueError("mask_slice must be two-dimensional")
    rows, columns = np.nonzero(binary)
    if rows.size == 0:
        raise ValueError("Cannot create a bounding box from an empty mask slice")
    row_start = max(int(rows.min()) - padding, 0)
    row_end = min(int(rows.max()) + padding + 1, binary.shape[0])
    col_start = max(int(columns.min()) - padding, 0)
    col_end = min(int(columns.max()) + padding + 1, binary.shape[1])
    return row_start, row_end, col_start, col_end


def crop_slice(image_slice: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    row_start, row_end, col_start, col_end = box
    return np.asarray(image_slice)[row_start:row_end, col_start:col_end]
