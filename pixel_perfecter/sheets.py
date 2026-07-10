"""Sprite-sheet frame splitting on reconstructed 1x art.

Reconstruction runs once over the whole sheet, so every frame shares one
grid, one pitch, and (with palette options) one palette by construction.
This module only has to find the frames: connected regions of non-background
pixels on the small 1x image, separated by background gutters.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from . import colors as color_ops

Box = Tuple[int, int, int, int]  # x0, y0, x1, y1 (exclusive)


def split_frames(
    art: np.ndarray,
    bg_tolerance: int = 12,
    min_size: int = 4,
    padding: int = 1,
) -> List[Box]:
    """Frame bounding boxes of a reconstructed 1x sheet, in reading order.

    Background is the dominant border color (or transparency when an alpha
    channel is present). Frames are connected components of foreground after
    a small closing, so sprites with 1px-disjoint parts stay together while
    frames separated by gutters split.
    """
    h, w = art.shape[:2]
    if art.shape[2] == 4:
        fg = art[:, :, 3] > color_ops.ALPHA_THRESHOLD
    else:
        bg = np.array(color_ops.most_common_boundary_color(art), dtype=np.int16)
        diff = np.abs(art[:, :, :3].astype(np.int16) - bg)
        fg = np.any(diff > bg_tolerance, axis=-1)

    mask = cv2.morphologyEx(
        fg.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    num, labels = cv2.connectedComponents(mask)

    boxes: List[Box] = []
    for label in range(1, num):
        ys, xs = np.where(labels == label)
        if xs.size == 0:
            continue
        x0 = max(0, int(xs.min()) - padding)
        y0 = max(0, int(ys.min()) - padding)
        x1 = min(w, int(xs.max()) + 1 + padding)
        y1 = min(h, int(ys.max()) + 1 + padding)
        if (x1 - x0) < min_size or (y1 - y0) < min_size:
            continue
        boxes.append((x0, y0, x1, y1))

    # Reading order: row bands (half the median frame height), then left.
    if boxes:
        band = max(1, int(np.median([b[3] - b[1] for b in boxes])) // 2)
        boxes.sort(key=lambda b: (b[1] // band, b[0]))
    return boxes


def crop_frames(art: np.ndarray, boxes: List[Box]) -> List[np.ndarray]:
    return [art[y0:y1, x0:x1].copy() for x0, y0, x1, y1 in boxes]
