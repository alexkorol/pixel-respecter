"""Vectorized rigid-grid primitives for pixel art reconstruction.

This module is the numeric engine behind :class:`PixelArtReconstructor`.
Everything here operates on numpy arrays without per-cell Python loops, so
scoring dozens of candidate grids costs milliseconds instead of seconds.

Key ideas:

* Work in *label space*: map the image to (palette, label) form once, then
  per-cell modal colors become a single ``np.bincount`` over
  ``cell_id * num_colors + label``.
* Score candidate grids with ``grid_quality_score`` (inter-cell contrast
  divided by intra-cell variance), which peaks at the true cell size and
  resolves harmonic ambiguity (e.g. 5 vs 10) toward the native resolution.
* Estimate grid phase (offset) by folding edge projections modulo the cell
  size instead of brute-forcing offset pairs.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

# Cap for the exact-color labeling path. Above this many unique colors the
# image is quantized (pixel art virtually never needs more for mode-picking).
MAX_EXACT_COLORS = 1024

# bincount table budget (cells x colors). Above this, labels are re-quantized.
MAX_BINCOUNT_CELLS = 30_000_000


def label_image(
    img: np.ndarray, max_colors: int = MAX_EXACT_COLORS
) -> Tuple[np.ndarray, np.ndarray]:
    """Map an RGB image to ``(labels, palette)``.

    Uses exact unique colors when there are few (clean pixel art), otherwise
    quantizes with PIL (no dither) so modal-color selection still operates on
    a small, meaningful palette — quantize-before-downsample, as in Kenneth
    Allen's proper-pixel-art.

    Returns:
        labels: (H, W) int32 array of palette indices.
        palette: (K, 3) uint8 array of RGB colors.
    """
    h, w = img.shape[:2]
    flat = img[:, :, :3].reshape(-1, 3).astype(np.uint32)
    packed = (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]
    uniq, inverse = np.unique(packed, return_inverse=True)

    if len(uniq) <= max_colors:
        palette = np.stack(
            [(uniq >> 16) & 0xFF, (uniq >> 8) & 0xFF, uniq & 0xFF], axis=1
        ).astype(np.uint8)
        return inverse.reshape(h, w).astype(np.int32), palette

    pil = Image.fromarray(img[:, :, :3], "RGB")
    num = int(min(256, max(2, max_colors)))
    try:
        quant = pil.quantize(colors=num, method=Image.Quantize.MAXCOVERAGE,
                             dither=Image.Dither.NONE)
    except Exception:
        quant = pil.quantize(colors=num, method=Image.Quantize.MEDIANCUT,
                             dither=Image.Dither.NONE)
    labels = np.asarray(quant, dtype=np.int32)
    raw_palette = quant.getpalette()
    used = labels.max() + 1
    palette = np.array(raw_palette[: used * 3], dtype=np.uint8).reshape(-1, 3)
    return labels, palette


def _cell_view(arr: np.ndarray, cell: int, offset: Tuple[int, int]) -> np.ndarray:
    """Crop ``arr`` to whole cells and reshape to (gh, cell, gw, cell, ...)."""
    h, w = arr.shape[:2]
    ox, oy = offset
    gh = (h - oy) // cell
    gw = (w - ox) // cell
    if gh < 1 or gw < 1:
        raise ValueError(f"cell={cell} offset={offset} leaves no whole cells")
    crop = arr[oy : oy + gh * cell, ox : ox + gw * cell]
    new_shape = (gh, cell, gw, cell) + crop.shape[2:]
    return crop.reshape(new_shape)


def cell_margin(cell: int) -> int:
    """Interior margin trimmed from each cell to suppress halo/AA bleed."""
    if cell < 4:
        return 0
    return max(1, min(cell // 6, (cell - 1) // 2))


def cell_vote_table(
    labels: np.ndarray,
    palette: np.ndarray,
    cell: int,
    offset: Tuple[int, int],
    row: int,
    column: int,
    trim_margin: bool = True,
) -> dict:
    """Return the modal-vote candidates for one reconstruction cell.

    The inspected pixels and trimming rule are exactly those used by
    :func:`downsample_mode`.  The result is deliberately JSON-friendly so
    browser tools can capture the decision context without reimplementing
    reconstruction. Candidates are sorted by vote count, then palette label.

    This function is introspection-only: it does not mutate labels, palette,
    or any reconstruction state.
    """
    cell = int(cell)
    view = _cell_view(labels, cell, offset)
    gh, gw = int(view.shape[0]), int(view.shape[2])
    row, column = int(row), int(column)
    if not (0 <= row < gh and 0 <= column < gw):
        raise IndexError(
            f"cell ({row}, {column}) outside reconstruction {gh}x{gw}"
        )

    margin = cell_margin(cell) if trim_margin else 0
    if margin and cell - 2 * margin >= 1:
        pixels = view[
            row, margin : cell - margin,
            column, margin : cell - margin,
        ]
    else:
        margin = 0
        pixels = view[row, :, column, :]

    # np.intp is required by np.bincount on wasm32/Pyodide.
    counts = np.bincount(
        pixels.reshape(-1).astype(np.intp, copy=False),
        minlength=len(palette),
    )
    used = np.flatnonzero(counts)
    ordered = sorted(used.tolist(), key=lambda label: (-counts[label], label))
    total = int(counts.sum())
    ox, oy = (int(offset[0]), int(offset[1]))
    x0, y0 = ox + column * cell, oy + row * cell
    candidates = [
        {
            "label": int(label),
            "color": [int(channel) for channel in palette[label, :3]],
            "count": int(counts[label]),
            "share": float(counts[label] / total) if total else 0.0,
        }
        for label in ordered
    ]
    return {
        "row": row,
        "column": column,
        "source_bounds": [x0, y0, x0 + cell, y0 + cell],
        "margin": margin,
        "total": total,
        "winner": candidates[0]["label"] if candidates else None,
        "vote_margin": (
            candidates[0]["share"]
            - (candidates[1]["share"] if len(candidates) > 1 else 0.0)
        ) if candidates else 0.0,
        "candidates": candidates,
    }


def downsample_mode(
    labels: np.ndarray,
    palette: np.ndarray,
    cell: int,
    offset: Tuple[int, int],
    trim_margin: bool = True,
) -> np.ndarray:
    """Collapse each grid cell to its modal palette color (vectorized).

    One output pixel per cell. Cell interiors are trimmed by
    :func:`cell_margin` so anti-aliased borders don't vote.
    """
    num_colors = len(palette)
    view = _cell_view(labels, cell, offset)
    gh, _, gw = view.shape[0], view.shape[1], view.shape[2]

    margin = cell_margin(cell) if trim_margin else 0
    if margin and cell - 2 * margin >= 1:
        view = view[:, margin : cell - margin, :, margin : cell - margin]

    cells = view.transpose(0, 2, 1, 3).reshape(gh * gw, -1)

    if gh * gw * num_colors > MAX_BINCOUNT_CELLS:
        if num_colors > 256:
            # Palette too large for the bincount table: re-quantize to 256.
            # (Recurse at most once - the next call lands in the banded
            # branch below. Recursing unconditionally used to loop forever
            # when the CELL count alone blew the budget, exhausting stack
            # or memory on huge/fine-pitched images.)
            small_labels, small_palette = _requantize(labels, palette, 256)
            return downsample_mode(
                small_labels, small_palette, cell, offset, trim_margin
            )
        # Palette already small: bincount in bands of cell rows so the
        # table stays within budget regardless of image size.
        band = max(1, MAX_BINCOUNT_CELLS // (gw * num_colors))
        out = np.empty((gh, gw, 3), dtype=palette.dtype)
        for y0 in range(0, gh, band):
            y1 = min(gh, y0 + band)
            sub = cells[y0 * gw : y1 * gw]
            n = sub.shape[0]
            ids = sub + np.arange(n, dtype=np.intp)[:, None] * num_colors
            counts = np.bincount(ids.ravel(), minlength=n * num_colors)
            modes = counts.reshape(n, num_colors).argmax(axis=1)
            out[y0:y1] = palette[modes].reshape(y1 - y0, gw, 3)
        return out

    # np.intp keeps bincount happy on 32-bit targets (Pyodide/wasm32); the
    # MAX_BINCOUNT_CELLS budget above guarantees ids fit.
    ids = cells + np.arange(gh * gw, dtype=np.intp)[:, None] * num_colors
    counts = np.bincount(ids.ravel(), minlength=gh * gw * num_colors)
    modes = counts.reshape(gh * gw, num_colors).argmax(axis=1)
    return palette[modes].reshape(gh, gw, 3)


def _requantize(
    labels: np.ndarray, palette: np.ndarray, num: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce an oversized palette by re-quantizing the palette colors."""
    img = palette[labels]
    return label_image(img, max_colors=num)


def upscale_nn(art: np.ndarray, scale: int) -> np.ndarray:
    """Nearest-neighbor upscale (exact block replication)."""
    return np.repeat(np.repeat(art, scale, axis=0), scale, axis=1)


def grid_quality_score(
    img: np.ndarray, cell: int, offset: Tuple[int, int] = (0, 0)
) -> float:
    """Score how well a rigid grid explains the image. Higher is better.

    score = inter-cell contrast / sqrt(intra-cell variance)

    * correct cell size: uniform cells (low intra) that differ from their
      neighbors (high inter) -> high score
    * half the true size (harmonic): intra stays low but half the neighbor
      pairs are identical -> inter halves -> score halves
    * double the true size: intra blows up -> score collapses
    """
    h, w = img.shape[:2]
    ox, oy = offset
    if cell < 1 or (h - oy) // cell < 2 or (w - ox) // cell < 2:
        return 0.0

    view = _cell_view(img[:, :, :3].astype(np.float32), cell, offset)
    # axes: (gh, cell_y, gw, cell_x, channel)
    means = view.mean(axis=(1, 3))
    sq_means = (view ** 2).mean(axis=(1, 3))
    intra = float(np.mean(sq_means - means ** 2))

    h_diff = np.abs(means[:, :-1] - means[:, 1:]).sum(axis=-1)
    v_diff = np.abs(means[:-1, :] - means[1:, :]).sum(axis=-1)
    pairs = np.concatenate([h_diff.ravel(), v_diff.ravel()])
    inter = float(pairs.mean())

    # Statistical support: a contrast mean over a handful of giant cells is
    # not comparable to one over thousands of fine-pitch pairs. Without it,
    # sprites on flat backgrounds crown huge cells (every pair straddles
    # subject and background) over the true pitch, whose mean contrast is
    # diluted by identical background neighbors. Negligible above ~2000
    # pairs, strong below ~200.
    support = float(np.sqrt(pairs.size / (pairs.size + 400.0)))

    if intra < 0.1:
        return inter * 100.0 * support
    return inter / float(np.sqrt(intra)) * support


def intra_cell_variance(
    img: np.ndarray, cell: int, offset: Tuple[int, int] = (0, 0)
) -> float:
    """Mean per-cell color variance (lower = better grid fit). Vectorized."""
    h, w = img.shape[:2]
    ox, oy = offset
    cell = max(int(cell), 1)
    if (h - oy) < cell or (w - ox) < cell:
        return float("inf")
    view = _cell_view(img[:, :, :3].astype(np.float32), cell, offset)
    means = view.mean(axis=(1, 3))
    sq_means = (view ** 2).mean(axis=(1, 3))
    return float(np.mean(sq_means - means ** 2))


def fold_phase(edge_strength: np.ndarray, cell: int) -> Tuple[int, float]:
    """Best grid phase for a 1D edge-projection signal, by modular folding.

    Grid lines sit at ``offset + k*cell``; edges of a well-aligned grid pile
    up in one residue class mod ``cell``. Returns (offset, alignment) where
    alignment is the folded peak relative to the mean (>=1.0; higher = best).
    """
    n = len(edge_strength)
    if n < cell or cell < 1:
        return 0, 0.0
    positions = np.arange(n) % cell
    folded = np.bincount(positions, weights=edge_strength, minlength=cell)
    # Normalize: residues appearing one extra time get a fair share.
    counts = np.bincount(positions, minlength=cell)
    folded = folded / np.maximum(counts, 1)
    mean = float(folded.mean())
    if mean <= 0:
        return 0, 0.0
    # Grid lines mark cell *boundaries*: a cell starting at offset has its
    # leading boundary at offset, so the offset equals the peak residue.
    best = int(np.argmax(folded))
    return best % cell, float(folded[best]) / mean


def detect_nn_scale(
    img: np.ndarray, max_scale: Optional[int] = None, min_cells: int = 4
) -> int:
    """Largest integer ``s`` such that the image is an exact s× NN upscale.

    Walks candidate scales from large to small; verifies each with a full
    vectorized block-uniformity check. Returns 1 when the image is not an
    exact upscale (anti-aliased art returns 1 — use grid scoring for those).
    """
    h, w = img.shape[:2]
    limit = min(h, w) // min_cells
    if max_scale is not None:
        limit = min(limit, max_scale)
    rgb = img[:, :, :3]
    # Cheap pre-check window: a scale that fails here fails everywhere.
    probe = rgb[: min(h, 128), : min(w, 128)]
    for scale in range(limit, 1, -1):
        if h % scale or w % scale:
            continue
        ph = (probe.shape[0] // scale) * scale
        pw = (probe.shape[1] // scale) * scale
        if ph >= scale and pw >= scale:
            pview = probe[:ph, :pw].reshape(ph // scale, scale, pw // scale, scale, 3)
            if not (pview == pview[:, :1, :, :1]).all():
                continue
        view = rgb.reshape(h // scale, scale, w // scale, scale, 3)
        if (view == view[:, :1, :, :1]).all():
            return scale
    return 1


def diff_projections(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Boundary-mass projections: how much the image changes at each line.

    Returns (proj_x, proj_y) where ``proj_x[j]`` is the total absolute color
    difference across the vertical line at ``x = j`` (between columns j-1
    and j). Sharper than Canny projections — the peak sits exactly on the
    cell boundary instead of smearing across adjacent pixels.
    """
    rgb = img[:, :, :3].astype(np.int16)
    h, w = rgb.shape[:2]
    proj_x = np.zeros(w, dtype=np.float64)
    proj_y = np.zeros(h, dtype=np.float64)
    proj_x[1:] = np.abs(np.diff(rgb, axis=1)).sum(axis=(0, 2))
    proj_y[1:] = np.abs(np.diff(rgb, axis=0)).sum(axis=(1, 2))
    return proj_x, proj_y


def autocorr_periods(
    projections: List[np.ndarray],
    min_period: int = 2,
    max_period: Optional[int] = None,
) -> List[Tuple[int, float]]:
    """Candidate periods from autocorrelation of 1D boundary projections.

    Returns up to 8 (period, strength) pairs sorted by strength.
    """
    results: dict[int, float] = {}
    for proj in projections:
        cap = max_period or min(128, len(proj) // 4)
        signal = proj.astype(np.float64) - proj.mean()
        if not np.any(signal):
            continue
        corr = np.correlate(signal, signal, mode="full")[len(signal) - 1 :]
        if corr[0] <= 0:
            continue
        corr = corr / corr[0]
        upper = min(len(corr) - 1, cap)
        for lag in range(max(min_period, 2), upper):
            if corr[lag] > corr[lag - 1] and corr[lag] >= corr[lag + 1]:
                strength = float(corr[lag])
                if strength > 0.15:
                    results[lag] = max(results.get(lag, 0.0), strength)
    ranked = sorted(results.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:8]


def build_diff_masks(
    original: np.ndarray,
    rec_up: np.ndarray,
    cell: int,
    offset: Tuple[int, int],
    tolerance: int = 0,
) -> dict:
    """Vectorized halo/core diff context (same keys as the legacy version)."""
    if tolerance > 0:
        channel_diff = np.abs(original.astype(np.int16) - rec_up.astype(np.int16))
        diff = np.any(channel_diff > tolerance, axis=-1)
    else:
        diff = np.any(original != rec_up, axis=-1)

    h, w = diff.shape
    cell = max(int(cell), 1)
    ox, oy = offset

    ys = np.arange(h)
    xs = np.arange(w)
    in_y = ys >= oy
    in_x = xs >= ox
    grid_mask = in_y[:, None] & in_x[None, :]

    margin = max(1, min(cell // 6, (cell - 1) // 2)) if cell > 2 else 0
    if margin and cell - 2 * margin >= 1:
        cy = (ys - oy) % cell
        cx = (xs - ox) % cell
        core_y = in_y & (cy >= margin) & (cy < cell - margin)
        core_x = in_x & (cx >= margin) & (cx < cell - margin)
        core_mask = core_y[:, None] & core_x[None, :]
    else:
        core_mask = np.zeros_like(grid_mask)

    core_mask &= grid_mask
    halo_mask = grid_mask & ~core_mask

    return {
        "diff": diff,
        "grid_mask": grid_mask,
        "core_mask": core_mask,
        "halo_mask": halo_mask,
        "core_diff": diff & core_mask,
        "halo_diff": diff & halo_mask,
        "outside_diff": diff & ~grid_mask,
    }
