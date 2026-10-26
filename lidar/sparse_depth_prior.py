"""Sparse projected-LiDAR inverse-depth prior for BlockGaussian training.

This is an extracted, reusable form of the masked depth term used in the
research training patch. It does not implement LiDAR normal supervision.
Only *training* views may be supplied to this prior; callers must enforce
the experiment's train/test split when generating projected LiDAR depths.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def load_inverse_depth_target(
    npz_path: str | Path,
    image_height: int,
    image_width: int,
    scene_scale: float = 1.0,
    min_valid_pixels: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a sparse LiDAR camera-Z map and return (inverse-Z, valid mask).

    The depth NPZ must contain ``depth`` and ``valid_mask`` arrays. Nearest-
    neighbor resizing matches the original SceneDataset patch. ``scene_scale``
    must match the BlockGaussian scene scale and the LiDAR/COLMAP coordinate
    registration must have been verified before training.
    """
    if scene_scale <= 0:
        raise ValueError("scene_scale must be positive")
    with np.load(npz_path, allow_pickle=False) as data:
        depth = np.asarray(data["depth"], dtype=np.float32)
        mask = np.asarray(data["valid_mask"], dtype=bool)
    if depth.ndim != 2 or mask.shape != depth.shape:
        raise ValueError("depth and valid_mask must be matching 2-D arrays")
    shape = (int(image_height), int(image_width))
    if depth.shape != shape:
        import cv2

        size = (shape[1], shape[0])
        depth = cv2.resize(depth, size, interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    mask &= np.isfinite(depth) & (depth > 1e-6)
    if int(mask.sum()) < min_valid_pixels:
        mask[:] = False
    target = np.zeros(shape, dtype=np.float32)
    target[mask] = 1.0 / (depth[mask] * float(scene_scale) + 1e-8)
    return torch.from_numpy(target), torch.from_numpy(mask)


def linear_lidar_weight(
    iteration: int,
    start: int,
    ramp_end: int,
    initial: float,
    final: float,
) -> float:
    """The training patch's zero-before-start, linearly ramped loss weight."""
    if iteration < start:
        return 0.0
    if ramp_end <= start:
        return float(final)
    alpha = float(np.clip((iteration - start) / (ramp_end - start), 0.0, 1.0))
    return float(initial + alpha * (final - initial))


def masked_inverse_depth_l1(
    rendered_inverse_depth: torch.Tensor,
    lidar_inverse_depth: torch.Tensor,
    lidar_valid_mask: torch.Tensor,
    min_valid_pixels: int = 1,
) -> tuple[torch.Tensor | None, int]:
    """Return (mean L1 loss, supervised pixel count), or (None, count).

    Inputs are single-view arrays in identical camera coordinates and pixel
    grids. No loss is applied to absent, nonfinite, or nonpositive depths.
    """
    if rendered_inverse_depth.shape != lidar_inverse_depth.shape:
        raise ValueError("rendered and LiDAR inverse-depth shapes differ")
    if lidar_valid_mask.shape != rendered_inverse_depth.shape:
        raise ValueError("LiDAR mask shape differs from rendered depth")
    valid = (
        lidar_valid_mask.bool()
        & torch.isfinite(rendered_inverse_depth)
        & torch.isfinite(lidar_inverse_depth)
        & (rendered_inverse_depth > 0)
        & (lidar_inverse_depth > 0)
    )
    count = int(valid.sum().item())
    if count < min_valid_pixels:
        return None, count
    return torch.abs(rendered_inverse_depth[valid] - lidar_inverse_depth[valid]).mean(), count
