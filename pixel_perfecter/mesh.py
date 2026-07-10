"""Mesh-based pixel grid detection.

Detects the pixel grid as two lists of line coordinates (a *mesh*) instead of
a rigid ``(cell_size, offset)`` pair. A mesh handles images whose pixel art
was upscaled by a non-integer factor — where cell widths alternate between,
say, 7 and 8 source pixels — which no rigid grid can represent.

The algorithm follows Kenneth J. Allen's *proper-pixel-art*
(https://github.com/KennethJAllen/proper-pixel-art, MIT license):

1. Canny edge detection on the (optionally upscaled) image
2. Morphological closing to bridge small gaps in the edge map
3. Probabilistic Hough transform to find near-vertical/horizontal lines
4. Cluster nearby lines and take cluster medians
5. Estimate the pixel width from the trimmed median of line gaps
6. "Homogenize": subdivide large gaps so the mesh covers the whole image

Differences from the original: gap subdivision uses rounding (not
truncation) to avoid cumulative drift, and the mesh is validated/scored by
the caller against other candidate grids rather than trusted outright.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

Lines = List[int]
Mesh = Tuple[Lines, Lines]


@dataclass
class MeshConfig:
    """Tunable parameters for mesh detection."""

    canny_thresholds: Tuple[int, int] = (50, 150)
    closure_kernel_size: int = 3
    cluster_threshold: int = 4
    angle_threshold_deg: float = 15.0
    trim_outlier_fraction: float = 0.2
    hough_threshold: int = 100
    hough_min_line_frac: float = 0.25  # of min(h, w)
    hough_max_line_gap: int = 10


def close_edges(edges: np.ndarray, kernel_size: int) -> np.ndarray:
    """Morphological closing to fill small gaps in the edge map."""
    if kernel_size <= 1:
        return edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)


def cluster_lines(lines: Lines, threshold: int) -> Lines:
    """Merge lines closer than ``threshold``; keep each cluster's median."""
    if not lines:
        return []
    lines = sorted(lines)
    clusters = [[lines[0]]]
    for line in lines[1:]:
        if abs(line - clusters[-1][-1]) <= threshold:
            clusters[-1].append(line)
        else:
            clusters.append([line])
    return [int(np.median(c)) for c in clusters]


def detect_grid_lines(
    edges: np.ndarray, config: Optional[MeshConfig] = None
) -> Mesh:
    """Hough-detect near-vertical/horizontal lines; cluster; include borders."""
    cfg = config or MeshConfig()
    height, width = edges.shape

    closed = close_edges(edges, cfg.closure_kernel_size)
    min_len = max(8, int(min(height, width) * cfg.hough_min_line_frac))
    hough_lines = cv2.HoughLinesP(
        closed,
        rho=1.0,
        theta=np.pi / 180.0,
        threshold=cfg.hough_threshold,
        minLineLength=min_len,
        maxLineGap=cfg.hough_max_line_gap,
    )

    # Image borders are grid lines that Hough cannot see.
    lines_x: Lines = [0, width - 1]
    lines_y: Lines = [0, height - 1]
    if hough_lines is None:
        return lines_x, lines_y

    angle_threshold = np.deg2rad(cfg.angle_threshold_deg)
    for x1, y1, x2, y2 in hough_lines[:, 0]:
        angle = abs(np.arctan2(y2 - y1, x2 - x1))
        if angle > np.pi / 2 - angle_threshold:
            lines_x.append(round((x1 + x2) / 2))
        elif angle < angle_threshold:
            lines_y.append(round((y1 + y2) / 2))

    return (
        cluster_lines(lines_x, cfg.cluster_threshold),
        cluster_lines(lines_y, cfg.cluster_threshold),
    )


def estimate_pixel_width(mesh: Mesh, trim_outlier_fraction: float = 0.2) -> float:
    """Pixel width from the trimmed median of line gaps (0.0 when unknown)."""
    gaps: List[np.ndarray] = []
    for lines in mesh:
        if len(lines) > 1:
            gaps.append(np.diff(sorted(lines)))
    if not gaps:
        return 0.0
    all_gaps = np.concatenate(gaps)
    low = np.percentile(all_gaps, 100 * trim_outlier_fraction)
    high = np.percentile(all_gaps, 100 * (1 - trim_outlier_fraction))
    middle = all_gaps[(all_gaps >= low) & (all_gaps <= high)]
    if len(middle) == 0:
        middle = all_gaps
    return float(np.median(middle))


def homogenize_lines(lines: Lines, pixel_width: float) -> Lines:
    """Fill gaps between detected lines so spacing is approximately even.

    Each gap is split into ``round(gap / pixel_width)`` sections; section
    boundaries are placed by rounding so error never accumulates past 0.5px.
    """
    if pixel_width <= 0 or len(lines) < 2:
        return list(lines)
    lines = sorted(lines)
    completed: List[int] = []
    for start, end in zip(lines[:-1], lines[1:]):
        gap = end - start
        sections = max(1, int(round(gap / pixel_width)))
        section_width = gap / sections
        completed.extend(
            int(start + round(n * section_width)) for n in range(sections)
        )
    completed.append(lines[-1])
    # Defend against duplicates from rounding tiny sections.
    return sorted(set(completed))


def is_trivial(mesh: Mesh) -> bool:
    """True when only borders (and at most one extra line) were found."""
    return len(mesh[0]) <= 3 and len(mesh[1]) <= 3


@dataclass
class MeshResult:
    """A detected mesh plus quality statistics."""

    lines_x: Lines
    lines_y: Lines
    pixel_width: float          # trimmed-median gap (original-image pixels)
    scale: int                  # upscale factor detection ran at
    gap_cv: float               # coefficient of variation of mesh gaps
    line_support: float         # detected lines / expected lines (0..1+)
    rigid_drift: float = 0.0    # max accumulated error of the best rigid fit
    max_residual: float = 0.0   # max wobble of lines around the linear fit

    @property
    def mesh(self) -> Mesh:
        return self.lines_x, self.lines_y

    @property
    def is_regular(self) -> bool:
        """True when a rigid (cell_size, offset) grid can represent the mesh.

        A mesh is *irregular* when the true pitch is fractional — a rigid
        integer grid drifts off the lines by more than half a pixel
        somewhere — or when the lines wobble around any linear fit.
        """
        return self.rigid_drift <= 0.6 and self.max_residual <= 1.5


def _gap_stats(mesh: Mesh) -> Tuple[float, float]:
    """Return (mean, coefficient of variation) of all interior gaps."""
    gaps: List[np.ndarray] = []
    for lines in mesh:
        if len(lines) > 2:
            gaps.append(np.diff(sorted(lines)))
    if not gaps:
        return 0.0, float("inf")
    all_gaps = np.concatenate(gaps).astype(np.float64)
    mean = float(all_gaps.mean())
    if mean <= 0:
        return 0.0, float("inf")
    return mean, float(all_gaps.std() / mean)


def _rigid_fit_drift(lines: Lines) -> Tuple[float, float]:
    """How badly the best rigid grid fits these lines.

    Fits position ~ intercept + pitch * index on interior lines (borders may
    be partial cells), then reports:

    * drift: |pitch - nearest integer| accumulated across all lines — the
      worst-case misalignment of an integer-pitch grid
    * max_residual: worst deviation of any line from the linear fit
    """
    if len(lines) < 3:
        return 0.0, 0.0
    interior = np.asarray(lines[1:-1] if len(lines) > 4 else lines, dtype=np.float64)
    if len(interior) < 2:
        return 0.0, 0.0
    idx = np.arange(len(interior), dtype=np.float64)
    pitch, intercept = np.polyfit(idx, interior, 1)
    residuals = interior - (intercept + pitch * idx)
    drift = abs(pitch - round(pitch)) * max(len(interior) - 1, 1)
    return float(drift), float(np.max(np.abs(residuals)))


def detect_mesh(
    gray: np.ndarray,
    upscale: int = 2,
    config: Optional[MeshConfig] = None,
) -> Optional[MeshResult]:
    """Detect the pixel mesh of a grayscale image.

    Runs at ``upscale``x first (helps Canny separate fine grids, per
    proper-pixel-art); falls back to 1x when that finds nothing. Returns
    ``None`` when no usable mesh exists.
    """
    cfg = config or MeshConfig()

    for scale in ([upscale, 1] if upscale > 1 else [1]):
        if scale > 1:
            work = cv2.resize(
                gray, (gray.shape[1] * scale, gray.shape[0] * scale),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            work = gray
        edges = cv2.Canny(work, *cfg.canny_thresholds)

        scaled_cfg = MeshConfig(
            canny_thresholds=cfg.canny_thresholds,
            closure_kernel_size=cfg.closure_kernel_size,
            cluster_threshold=cfg.cluster_threshold * scale,
            angle_threshold_deg=cfg.angle_threshold_deg,
            trim_outlier_fraction=cfg.trim_outlier_fraction,
            hough_threshold=cfg.hough_threshold,
            hough_min_line_frac=cfg.hough_min_line_frac,
            hough_max_line_gap=cfg.hough_max_line_gap * scale,
        )
        initial = detect_grid_lines(edges, scaled_cfg)
        if is_trivial(initial):
            continue

        width_scaled = estimate_pixel_width(initial, cfg.trim_outlier_fraction)
        if width_scaled < 2:
            continue

        lines_x = homogenize_lines(initial[0], width_scaled)
        lines_y = homogenize_lines(initial[1], width_scaled)

        # Scale line coordinates back to the original image.
        if scale > 1:
            lines_x = sorted(set(int(round(x / scale)) for x in lines_x))
            lines_y = sorted(set(int(round(y / scale)) for y in lines_y))

        mesh: Mesh = (lines_x, lines_y)
        gap_mean, gap_cv = _gap_stats(mesh)
        if gap_mean < 1.0:
            continue

        h, w = gray.shape[:2]
        expected = w / gap_mean + h / gap_mean
        detected = len(initial[0]) + len(initial[1])
        support = detected / expected if expected > 0 else 0.0

        drift_x, resid_x = _rigid_fit_drift(lines_x)
        drift_y, resid_y = _rigid_fit_drift(lines_y)

        return MeshResult(
            lines_x=lines_x,
            lines_y=lines_y,
            pixel_width=width_scaled / scale,
            scale=scale,
            gap_cv=gap_cv,
            line_support=min(1.5, support),
            rigid_drift=max(drift_x, drift_y),
            max_residual=max(resid_x, resid_y),
        )

    return None


# ---------------------------------------------------------------------------
# Projection-based mesh construction (robust on dense art where Hough fails)
# ---------------------------------------------------------------------------


def estimate_float_pitch(proj: np.ndarray, approx: int) -> float:
    """Sub-pixel grid pitch from the autocorrelation of an edge projection.

    The autocorrelation of a grid-aligned edge projection peaks at every
    multiple of the pitch. Locating the K-th peak and dividing by K measures
    the pitch with ~1/K pixel precision; parabolic interpolation around the
    peak adds sub-integer resolution. Refines iteratively so large K can be
    used safely.
    """
    if approx < 2 or len(proj) < approx * 3:
        return float(approx)
    signal = proj.astype(np.float64) - proj.mean()
    if not np.any(signal):
        return float(approx)
    corr = np.correlate(signal, signal, mode="full")[len(signal) - 1 :]
    if corr[0] <= 0:
        return float(approx)

    pitch = float(approx)
    uncertainty = 0.5  # detection was integer-precise
    max_lag = len(corr) - 2
    for _ in range(3):
        # Largest K whose search window (pitch uncertainty + 2) stays
        # unambiguous (less than half a pitch) and inside the signal.
        k_window = int((pitch / 2 - 1) / max(uncertainty, 1e-6))
        k_signal = int(max_lag / pitch)
        k = max(1, min(k_window, k_signal, 32))
        if k <= 1:
            break
        center = k * pitch
        half = min(pitch / 2 - 1, k * uncertainty + 2)
        lo = max(1, int(np.floor(center - half)))
        hi = min(max_lag, int(np.ceil(center + half)))
        if hi <= lo:
            break
        lag = lo + int(np.argmax(corr[lo : hi + 1]))
        # Parabolic sub-bin refinement.
        if 1 <= lag < len(corr) - 1:
            c_prev, c_mid, c_next = corr[lag - 1], corr[lag], corr[lag + 1]
            denom = c_prev - 2 * c_mid + c_next
            if abs(denom) > 1e-12:
                shift = 0.5 * (c_prev - c_next) / denom
                if abs(shift) <= 1:
                    lag = lag + shift
        new_pitch = float(lag) / k
        if abs(new_pitch - pitch) > max(1.0, pitch * 0.2):
            break  # rejected: window caught a neighboring structure
        pitch = new_pitch
        uncertainty = 1.5 / k
    return pitch


def projection_mesh(proj: np.ndarray, pitch: float, snap: int = 1) -> Lines:
    """Build mesh lines for one axis by snapping an even grid to edge peaks.

    Finds the grid phase maximizing edge mass under the lines, places lines
    every ``pitch`` (float), and snaps each to the strongest edge within
    ``snap`` pixels. Lines without local edge support stay at their expected
    position, so sparse regions don't break the mesh.
    """
    n = len(proj)
    if pitch < 2 or n < pitch * 2:
        return [0, max(1, n - 1)]
    num_cells = max(1, int(round(n / pitch)))

    ks = np.arange(num_cells + 1, dtype=np.float64)
    best_phi, best_mass = 0.0, -1.0
    for phi in np.arange(0.0, pitch, 0.25):
        pos = np.round(phi + ks * pitch).astype(int)
        pos = pos[(pos > 0) & (pos < n - 1)]
        if len(pos) == 0:
            continue
        mass = float(proj[pos].sum())
        if mass > best_mass:
            best_mass, best_phi = mass, phi

    lines = {0, n - 1}
    for k in ks:
        expected = int(round(best_phi + k * pitch))
        if expected <= 0 or expected >= n - 1:
            continue
        lo = max(1, expected - snap)
        hi = min(n - 2, expected + snap)
        window = proj[lo : hi + 1]
        snapped = lo + int(np.argmax(window))
        lines.add(snapped if window.max() > 0 else expected)
    return sorted(lines)


# ---------------------------------------------------------------------------
# Mesh-based downsampling and scoring (variable cell sizes)
# ---------------------------------------------------------------------------


def _cell_ids(shape: Tuple[int, int], mesh: Mesh) -> Tuple[np.ndarray, int, int]:
    """Map every pixel to its mesh cell id. Returns (ids, grid_h, grid_w)."""
    h, w = shape
    lines_x = np.asarray(mesh[0], dtype=np.int64)
    lines_y = np.asarray(mesh[1], dtype=np.int64)
    # Interior boundaries split cells; pixels outside [first, last) clamp.
    col = np.clip(np.searchsorted(lines_x, np.arange(w), side="right") - 1,
                  0, len(lines_x) - 2)
    row = np.clip(np.searchsorted(lines_y, np.arange(h), side="right") - 1,
                  0, len(lines_y) - 2)
    grid_w = len(lines_x) - 1
    grid_h = len(lines_y) - 1
    ids = row[:, None] * grid_w + col[None, :]
    return ids, grid_h, grid_w


def downsample_mesh(
    labels: np.ndarray, palette: np.ndarray, mesh: Mesh
) -> np.ndarray:
    """Collapse each (possibly unevenly sized) mesh cell to its modal color."""
    h, w = labels.shape
    ids, grid_h, grid_w = _cell_ids((h, w), mesh)
    num_colors = len(palette)
    num_cells = grid_h * grid_w

    combined = ids.astype(np.int64) * num_colors + labels
    counts = np.bincount(combined.ravel(), minlength=num_cells * num_colors)
    modes = counts.reshape(num_cells, num_colors).argmax(axis=1)
    return palette[modes].reshape(grid_h, grid_w, 3)


def mesh_quality_score(img: np.ndarray, mesh: Mesh) -> float:
    """Inter/intra score for a variable mesh (same scale as grid version)."""
    h, w = img.shape[:2]
    ids, grid_h, grid_w = _cell_ids((h, w), mesh)
    if grid_h < 2 or grid_w < 2:
        return 0.0
    num_cells = grid_h * grid_w
    flat_ids = ids.ravel()
    pixels = img[:, :, :3].reshape(-1, 3).astype(np.float64)

    counts = np.bincount(flat_ids, minlength=num_cells).astype(np.float64)
    counts = np.maximum(counts, 1)
    sums = np.stack(
        [np.bincount(flat_ids, weights=pixels[:, c], minlength=num_cells)
         for c in range(3)], axis=1)
    sq_sums = np.stack(
        [np.bincount(flat_ids, weights=pixels[:, c] ** 2, minlength=num_cells)
         for c in range(3)], axis=1)

    means = sums / counts[:, None]
    variances = sq_sums / counts[:, None] - means ** 2
    intra = float(np.mean(variances))

    mean_grid = means.reshape(grid_h, grid_w, 3)
    h_diff = np.abs(mean_grid[:, :-1] - mean_grid[:, 1:]).sum(axis=-1)
    v_diff = np.abs(mean_grid[:-1, :] - mean_grid[1:, :]).sum(axis=-1)
    pairs = np.concatenate([h_diff.ravel(), v_diff.ravel()])
    inter = float(pairs.mean())

    # Same statistical-support weight as grid_quality_score: a mesh of a few
    # giant cells straddling subject and background must not outscore a fine
    # mesh judged over thousands of pairs.
    support = float(np.sqrt(pairs.size / (pairs.size + 400.0)))

    if intra < 0.1:
        return inter * 100.0 * support
    return inter / float(np.sqrt(intra)) * support
