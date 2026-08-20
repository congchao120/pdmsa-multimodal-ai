from __future__ import annotations

import numpy as np

from .manifest import DEFAULT_SLICE_OFFSETS


def select_five_slices(mask: np.ndarray, axis: int = 2) -> list[int]:
    """Return the largest-ROI axial index and its exact ``-2..+2`` window.

    Indices are zero-based. The input mask must already be reoriented to the
    study's canonical axial grid, with ``axis=2`` identifying the axial axis.
    Ties are resolved by ``numpy.argmax`` (the lowest index). A center without
    two real neighboring slices on both sides is rejected rather than shifted
    or duplicated, preserving the exact center-minus/plus-two definition.
    """
    array = np.asarray(mask)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D mask, found shape {array.shape}")
    if not isinstance(axis, (int, np.integer)) or int(axis) not in {0, 1, 2}:
        raise ValueError("axis must be one of the canonical array axes 0, 1, or 2")
    axis = int(axis)
    binary = array != 0
    if not binary.any():
        raise ValueError("Cannot select slices from an empty segmentation mask")
    if array.shape[axis] < 5:
        raise ValueError("The selected axis contains fewer than five slices")

    reduction_axes = tuple(index for index in range(3) if index != axis)
    areas = binary.sum(axis=reduction_axes)
    center = int(np.argmax(areas))
    if center < 2 or center > array.shape[axis] - 3:
        raise ValueError(
            f"Largest-ROI center {center} does not have two real neighbors on both sides"
        )
    return list(range(center - 2, center + 3))


def centered_slice_metadata(mask: np.ndarray, axis: int = 2) -> list[dict[str, int]]:
    """Build manifest-ready metadata for a subject-specific centered window.

    This is the preprocessing entry point used to replace fixed absolute slice
    numbers. The returned rows can be joined to that subject's five fused-image
    paths before the classification manifest is written.
    """
    indices = select_five_slices(mask, axis=axis)
    center = indices[2]
    return [
        {
            "center_slice_index": center,
            "slice_offset": offset,
            "slice_index": slice_index,
        }
        for offset, slice_index in zip(DEFAULT_SLICE_OFFSETS, indices, strict=True)
    ]


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
