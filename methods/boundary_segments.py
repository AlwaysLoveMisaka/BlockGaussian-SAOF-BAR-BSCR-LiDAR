"""Finite shared x/z edges for BlockGaussian's axis-aligned block layout.

Unlike an infinite x or z plane, an edge is restricted to the actual span
shared by two neighbouring block bounding boxes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def extract_segments(blocks_info_path: str | Path, tol: float = 1e-4) -> list[dict]:
    info = json.loads(Path(blocks_info_path).read_text())
    boxes = {}
    for key, value in info.items():
        if key == "num_blocks":
            continue
        bbx = np.asarray(value["bbx"], dtype=np.float64)
        if bbx.shape != (8, 3):
            raise ValueError(f"Unexpected bbox shape for block {key}: {bbx.shape}")
        boxes[int(key)] = (float(bbx[:, 0].min()), float(bbx[:, 0].max()),
                           float(bbx[:, 2].min()), float(bbx[:, 2].max()))
    if len(boxes) != info["num_blocks"]:
        raise ValueError("Block count mismatch")
    segments = []
    ids = sorted(boxes)
    for index, left_id in enumerate(ids):
        ax0, ax1, az0, az1 = boxes[left_id]
        for right_id in ids[index + 1:]:
            bx0, bx1, bz0, bz1 = boxes[right_id]
            z0, z1 = max(az0, bz0), min(az1, bz1)
            x0, x1 = max(ax0, bx0), min(ax1, bx1)
            if z1 - z0 > tol and (abs(ax1 - bx0) <= tol or abs(bx1 - ax0) <= tol):
                coordinate = (ax1 + bx0) / 2 if abs(ax1 - bx0) <= tol else (bx1 + ax0) / 2
                segments.append({"axis": "x", "coordinate": coordinate,
                                 "lower": z0, "upper": z1, "blocks": [left_id, right_id]})
            if x1 - x0 > tol and (abs(az1 - bz0) <= tol or abs(bz1 - az0) <= tol):
                coordinate = (az1 + bz0) / 2 if abs(az1 - bz0) <= tol else (bz1 + az0) / 2
                segments.append({"axis": "z", "coordinate": coordinate,
                                 "lower": x0, "upper": x1, "blocks": [left_id, right_id]})
    if not segments:
        raise ValueError("No finite shared block boundaries found")
    return segments


def mask_numpy(xyz: np.ndarray, segments: list[dict], width: float):
    if width < 0 or not segments:
        raise ValueError("A nonnegative width and at least one segment are required")
    x, z = xyz[:, 0], xyz[:, 2]
    distance = np.full(len(xyz), np.inf, dtype=np.float32)
    for edge in segments:
        coordinate, lower, upper = edge["coordinate"], edge["lower"], edge["upper"]
        if edge["axis"] == "x":
            parallel, transverse = z, np.abs(x - coordinate)
        else:
            parallel, transverse = x, np.abs(z - coordinate)
        endpoint = np.maximum(np.maximum(lower - parallel, parallel - upper), 0.0)
        distance = np.minimum(distance, np.hypot(transverse, endpoint))
    return distance <= width, distance


def mask_torch(xyz, segments: list[dict], width: float):
    import torch

    if width < 0 or not segments:
        raise ValueError("A nonnegative width and at least one segment are required")
    x, z = xyz[:, 0], xyz[:, 2]
    distance = torch.full_like(x, float("inf"))
    for edge in segments:
        coordinate, lower, upper = edge["coordinate"], edge["lower"], edge["upper"]
        if edge["axis"] == "x":
            parallel, transverse = z, torch.abs(x - coordinate)
        else:
            parallel, transverse = x, torch.abs(z - coordinate)
        endpoint = torch.maximum(torch.maximum(lower - parallel, parallel - upper),
                                 torch.zeros_like(parallel))
        distance = torch.minimum(distance, torch.hypot(transverse, endpoint))
    return distance <= width, distance
