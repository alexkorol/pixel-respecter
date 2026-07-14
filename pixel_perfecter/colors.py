"""Color and transparency handling for pixel art reconstruction.

Alpha-aware cell colors, palette quantization, and background-transparency
helpers. The transparency-majority rule and boundary-color background
detection follow Kenneth J. Allen's *proper-pixel-art*
(https://github.com/KennethJAllen/proper-pixel-art, MIT license).
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Optional, Tuple

import numpy as np
from PIL import Image

# Alpha >= this counts as opaque (0-255).
ALPHA_THRESHOLD = 128

# A cell becomes transparent when at least this fraction of it is transparent.
TRANSPARENCY_MAJORITY = 0.5


def quantize_image(
    img: np.ndarray, num_colors: int = 64
) -> Tuple[np.ndarray, np.ndarray]:
    """Quantize an RGB image to ``num_colors``; return (labels, palette).

    No dithering — dither noise is exactly what pixel art reconstruction
    must avoid (quantize-before-downsample, per proper-pixel-art).
    """
    pil = Image.fromarray(img[:, :, :3], "RGB")
    num = int(min(256, max(2, num_colors)))
    try:
        quant = pil.quantize(colors=num, method=Image.Quantize.MAXCOVERAGE,
                             dither=Image.Dither.NONE)
    except Exception:
        quant = pil.quantize(colors=num, method=Image.Quantize.MEDIANCUT,
                             dither=Image.Dither.NONE)
    labels = np.asarray(quant, dtype=np.int32)
    used = int(labels.max()) + 1
    palette = np.array(quant.getpalette()[: used * 3], dtype=np.uint8).reshape(-1, 3)
    return labels, palette


def cell_alpha_rigid(
    alpha: np.ndarray,
    cell: int,
    offset: Tuple[int, int],
    alpha_threshold: int = ALPHA_THRESHOLD,
    majority: float = TRANSPARENCY_MAJORITY,
) -> np.ndarray:
    """Per-cell alpha for a rigid grid: 0 or 255 by transparency majority.

    Vectorized; one output value per whole cell.
    """
    h, w = alpha.shape[:2]
    ox, oy = offset
    gh = (h - oy) // cell
    gw = (w - ox) // cell
    if gh < 1 or gw < 1:
        return np.full((max(gh, 0), max(gw, 0)), 255, dtype=np.uint8)
    crop = alpha[oy : oy + gh * cell, ox : ox + gw * cell]
    opaque = (crop >= alpha_threshold).reshape(gh, cell, gw, cell)
    opaque_frac = opaque.mean(axis=(1, 3))
    return np.where(opaque_frac > (1.0 - majority), 255, 0).astype(np.uint8)


def cell_alpha_mesh(
    alpha: np.ndarray,
    mesh: Tuple[list, list],
    alpha_threshold: int = ALPHA_THRESHOLD,
    majority: float = TRANSPARENCY_MAJORITY,
) -> np.ndarray:
    """Per-cell alpha for a variable mesh: 0 or 255 by transparency majority."""
    from . import mesh as mesh_ops

    h, w = alpha.shape[:2]
    ids, grid_h, grid_w = mesh_ops._cell_ids((h, w), mesh)
    flat_ids = ids.ravel()
    num_cells = grid_h * grid_w
    counts = np.maximum(np.bincount(flat_ids, minlength=num_cells), 1)
    opaque = np.bincount(
        flat_ids, weights=(alpha.ravel() >= alpha_threshold), minlength=num_cells
    )
    frac = opaque / counts
    return np.where(frac > (1.0 - majority), 255, 0).astype(np.uint8).reshape(
        grid_h, grid_w
    )


def dominant_color_binned(rgb_pixels: np.ndarray, bin_size: int = 52) -> Tuple[int, int, int]:
    """Dominant color of a pixel set via offset binning (proper-pixel-art).

    RGB space is bucketed on two grids — one standard, one shifted by half a
    bin — and the larger winning bucket's median color is returned. The dual
    grid avoids colors straddling a single grid's boundary splitting their
    vote.
    """
    if len(rgb_pixels) == 0:
        return (0, 0, 0)
    if len(rgb_pixels) == 1:
        r, g, b = rgb_pixels[0]
        return (int(r), int(g), int(b))
    if len(rgb_pixels) <= 3:
        median = np.median(rgb_pixels, axis=0).astype(np.uint8)
        return (int(median[0]), int(median[1]), int(median[2]))

    num_bins = 255 // bin_size + 1
    # np.intp, not int64: bincount rejects int64 on 32-bit targets (wasm32).
    pixels = rgb_pixels.astype(np.intp)

    bins1 = pixels // bin_size
    idx1 = bins1[:, 0] * num_bins**2 + bins1[:, 1] * num_bins + bins1[:, 2]
    counts1 = np.bincount(idx1, minlength=num_bins**3)
    dom1 = int(counts1.argmax())

    offset = bin_size // 2
    bins2 = np.minimum(pixels + offset, 255) // bin_size
    idx2 = bins2[:, 0] * num_bins**2 + bins2[:, 1] * num_bins + bins2[:, 2]
    counts2 = np.bincount(idx2, minlength=num_bins**3)
    dom2 = int(counts2.argmax())

    mask = idx1 == dom1 if counts1[dom1] >= counts2[dom2] else idx2 == dom2
    median = np.median(rgb_pixels[mask], axis=0).astype(np.uint8)
    return (int(median[0]), int(median[1]), int(median[2]))


def most_common_boundary_color(img: np.ndarray) -> Tuple[int, int, int]:
    """The RGB color appearing most often on the image border."""
    rgb = img[:, :, :3]
    border = np.concatenate([
        rgb[0, :], rgb[-1, :], rgb[1:-1, 0], rgb[1:-1, -1],
    ])
    counts = Counter(map(tuple, border))
    color = counts.most_common(1)[0][0]
    return (int(color[0]), int(color[1]), int(color[2]))


def make_background_transparent(img: np.ndarray) -> np.ndarray:
    """Zero the alpha of every pixel matching the dominant border color."""
    h, w = img.shape[:2]
    if img.shape[2] == 4:
        rgba = img.copy()
    else:
        rgba = np.dstack([img, np.full((h, w), 255, dtype=np.uint8)])
    bg = most_common_boundary_color(img)
    matches = np.all(rgba[:, :, :3] == np.array(bg, dtype=np.uint8), axis=-1)
    rgba[matches, 3] = 0
    return rgba


def make_border_connected_transparent(
    img: np.ndarray,
    tolerance: Optional[float] = None,
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """Make the border-connected dominant-color cluster transparent.

    Unlike :func:`make_background_transparent`, this preserves disconnected
    interior pixels that happen to reuse the background color. That is safer
    for item/NPC sprites where a dark background can also appear in outlines
    or internal details. By default the RGB tolerance adapts to the 90th
    percentile of border variation, which removes dithered/noisy generated
    backgrounds rather than only one exact palette entry.
    """
    h, w = img.shape[:2]
    if h < 1 or w < 1:
        raise ValueError("image must contain at least one pixel")
    if img.shape[2] == 4:
        rgba = img.copy()
    else:
        rgba = np.dstack([img[:, :, :3], np.full((h, w), 255, dtype=np.uint8)])
    bg = most_common_boundary_color(rgba)
    rgb = rgba[:, :, :3].astype(np.float32)
    center = np.asarray(bg, dtype=np.float32)
    border = np.concatenate([
        rgb[0, :], rgb[-1, :], rgb[1:-1, 0], rgb[1:-1, -1],
    ])
    border_distance = np.sqrt(np.sum((border - center) ** 2, axis=1))
    if tolerance is None:
        tolerance = min(
            64.0,
            max(2.0, float(np.percentile(border_distance, 90)) + 2.0),
        )
    tolerance = max(0.0, float(tolerance))
    distance = np.sqrt(np.sum((rgb - center) ** 2, axis=2))
    matches = distance <= tolerance
    connected = np.zeros((h, w), dtype=bool)
    pending = deque()
    for x in range(w):
        pending.append((0, x))
        if h > 1:
            pending.append((h - 1, x))
    for y in range(1, h - 1):
        pending.append((y, 0))
        if w > 1:
            pending.append((y, w - 1))
    while pending:
        y, x = pending.popleft()
        if connected[y, x] or not matches[y, x]:
            continue
        connected[y, x] = True
        if y > 0:
            pending.append((y - 1, x))
        if y + 1 < h:
            pending.append((y + 1, x))
        if x > 0:
            pending.append((y, x - 1))
        if x + 1 < w:
            pending.append((y, x + 1))
    rgba[:, :, 3] = np.where(connected, 0, 255).astype(np.uint8)
    return rgba, bg


def crop_transparent_border(
    img: np.ndarray, padding: int = 0
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Crop an RGBA sprite to its opaque bounds plus cell padding.

    Returns ``(cropped, (x0, y0, x1, y1))`` using half-open bounds in the
    original 1x grid. RGB inputs have no transparent border and therefore
    return unchanged. Fully transparent inputs also remain unchanged so a
    mistaken transparency choice never collapses a sprite to an empty image.
    """
    image = np.asarray(img)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("expected an HxWx3 or HxWx4 image")
    h, w = image.shape[:2]
    if h < 1 or w < 1:
        raise ValueError("image must contain at least one pixel")
    if image.shape[2] != 4:
        return image.copy(), (0, 0, w, h)
    opaque_y, opaque_x = np.nonzero(image[:, :, 3] > 0)
    if len(opaque_x) == 0:
        return image.copy(), (0, 0, w, h)
    padding = max(0, int(padding))
    # Bounds intentionally may extend beyond the original canvas. Production
    # sprites need the requested transparent buffer even when opaque content
    # touches a generated edge.
    x0 = int(opaque_x.min()) - padding
    y0 = int(opaque_y.min()) - padding
    x1 = int(opaque_x.max()) + 1 + padding
    y1 = int(opaque_y.max()) + 1 + padding
    cropped = np.zeros((y1 - y0, x1 - x0, image.shape[2]), dtype=image.dtype)
    src_x0, src_y0 = max(0, x0), max(0, y0)
    src_x1, src_y1 = min(w, x1), min(h, y1)
    dst_x0, dst_y0 = src_x0 - x0, src_y0 - y0
    cropped[
        dst_y0 : dst_y0 + (src_y1 - src_y0),
        dst_x0 : dst_x0 + (src_x1 - src_x0),
    ] = image[src_y0:src_y1, src_x0:src_x1]
    return cropped, (x0, y0, x1, y1)


def flatten_alpha(
    img: np.ndarray,
    alpha_threshold: int = ALPHA_THRESHOLD,
    background: Optional[Tuple[int, int, int]] = None,
) -> np.ndarray:
    """Composite RGBA onto a solid background that won't collide with content.

    Transparent regions must not contribute edges or colors during grid
    detection; replacing them with a far-away flat color keeps them inert
    (clamp-alpha idea from proper-pixel-art).
    """
    if img.shape[2] != 4:
        return img[:, :, :3]
    rgb = img[:, :, :3]
    opaque = img[:, :, 3] >= alpha_threshold
    if background is None:
        background = _farthest_background(rgb[opaque])
    out = np.empty_like(rgb)
    out[:] = np.array(background, dtype=np.uint8)
    out[opaque] = rgb[opaque]
    return out


_BACKGROUND_CANDIDATES = np.array([
    (255, 0, 255), (0, 255, 255), (255, 255, 0),
    (0, 255, 0), (255, 128, 0), (128, 0, 255),
    (255, 255, 255), (0, 0, 0),
], dtype=np.int64)


def _farthest_background(opaque_pixels: np.ndarray) -> Tuple[int, int, int]:
    """Pick the candidate color farthest from the image's common colors."""
    if len(opaque_pixels) == 0:
        return (255, 0, 255)
    sample = opaque_pixels
    if len(sample) > 4096:
        step = len(sample) // 4096
        sample = sample[::step]
    sample = sample.astype(np.int64)
    # min distance from each candidate to any sampled color
    diffs = _BACKGROUND_CANDIDATES[:, None, :] - sample[None, :, :]
    dists = (diffs ** 2).sum(axis=-1).min(axis=1)
    best = _BACKGROUND_CANDIDATES[int(dists.argmax())]
    return (int(best[0]), int(best[1]), int(best[2]))
