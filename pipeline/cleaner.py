"""Clean extracted sprites into pixel-perfect assets using pixel-perfecter.

Takes raw extracted sprites (possibly with anti-aliasing, halo glow,
JPEG artifacts, scaling artifacts) and produces clean pixel-perfect
versions at 1x native resolution.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from pixel_perfecter import colors as color_ops
from pixel_perfecter import grid as grid_ops
from pixel_perfecter.reconstructor import (
    PixelArtReconstructor,
    _compute_intra_cell_variance,
)

logger = logging.getLogger(__name__)

# Maximum acceptable intra-cell variance.  Lower means cells are more
# uniform which indicates correct grid detection.  AI-generated pixel art
# typically has variance 20-300; detailed scenes can go to 500.
# Non-pixel-art images are typically 700+.
MAX_CELL_VARIANCE = 600.0

# Minimum acceptable cell_size for the pipeline
MIN_CELL_SIZE = 4


@dataclass
class CleanedAsset:
    """A pixel-perfect cleaned sprite."""
    image_1x: np.ndarray       # native resolution (1 pixel per grid cell)
    image_upscaled: np.ndarray  # nearest-neighbor upscaled for training
    cell_size: int             # detected grid cell size in source
    offset: Tuple[int, int]    # grid offset in source
    metrics: dict              # reconstruction quality metrics
    source_path: str = ""
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


def _detect_nearest_neighbor_scale(img: np.ndarray) -> int:
    """Detect if an image has been nearest-neighbor upscaled.

    Returns the detected scale factor (1 if no upscaling detected).
    Checks for repeating pixel blocks — a telltale sign of NN upscale.
    Tests largest scales first so we find the true native resolution.

    Uses a two-tier check to avoid false positives on AI-generated pixel
    art where smooth gradients can produce low ptp at small block sizes:
      - Blocks must be strictly identical (ptp=0) at >= 60% rate, AND
      - Blocks must be near-identical (ptp<=2) at >= 90% rate.
    True NN upscaling produces ~100% ptp=0 blocks; AI gradients produce
    <1% ptp=0 blocks even when ptp<=2 is high.
    """
    h, w = img.shape[:2]
    if h < 8 or w < 8:
        return 1

    gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY)

    # test LARGEST scale factors first — we want the true native resolution
    for scale in [16, 12, 10, 8, 6, 5, 4, 3, 2]:
        if h % scale != 0 or w % scale != 0:
            continue
        # require at least 4 cells per dimension
        if h // scale < 4 or w // scale < 4:
            continue

        # sample a section and check that scale x scale blocks are uniform
        sample_h = min(h, 64 * scale) // scale * scale
        sample_w = min(w, 64 * scale) // scale * scale
        blocks = gray[:sample_h, :sample_w].reshape(
            sample_h // scale, scale, sample_w // scale, scale
        )
        ptp = blocks.max(axis=(1, 3)).astype(np.int16) - blocks.min(axis=(1, 3))

        strict_ratio = float(np.mean(ptp == 0))
        loose_ratio = float(np.mean(ptp <= 2))

        # True NN upscaling: most blocks are exactly identical
        # AI-generated: smooth gradients make blocks look near-uniform but
        # NOT exactly identical.
        if strict_ratio >= 0.60 and loose_ratio >= 0.90:
            return scale  # largest matching scale is the true upscale factor

    return 1


def _downscale_nn(img: np.ndarray, scale: int) -> np.ndarray:
    """Downscale by sampling every `scale`-th pixel."""
    return img[::scale, ::scale].copy()


# Standard pixel art cell sizes to try as fallbacks.
# For a 1024x1024 image these give: 256, 128, 64, 32, 16 cells per dim.
STANDARD_CELL_SIZES = [4, 8, 16, 32, 64]


def _grid_quality_score(img: np.ndarray, cell_size: int, offset: Tuple[int, int] = (0, 0)) -> float:
    """Score grid quality: inter-cell contrast / sqrt(intra-cell variance).

    Peaks at the correct cell size because:
    - At correct size: cells are internally uniform (low intra) but
      differ from neighbors (high inter)
    - At too-small sizes: both inter and intra are low (subdividing one pixel)
    - At too-large sizes: intra is high (spanning multiple pixels)

    Delegates to the vectorized implementation in pixel_perfecter.grid.
    """
    return grid_ops.grid_quality_score(img, cell_size, offset)


def _pick_best_cell_size(
    rgb: np.ndarray,
    detected_size: int,
    detected_offset: Tuple[int, int],
) -> Tuple[int, Tuple[int, int]]:
    """Refine detected cell size by snapping to a nearby standard size.

    Pixel art almost always uses standard cell sizes (8, 16, 32, 64).
    If the detected size is within 25% of a standard, snap to it and
    verify with the grid quality metric.  Uses detected size as fallback.

    Returns (cell_size, offset).
    """
    h, w = rgb.shape[:2]

    # Find the nearest valid standard size
    best_std = None
    best_dist = float("inf")
    for std in STANDARD_CELL_SIZES:
        if std < MIN_CELL_SIZE or h // std < 4 or w // std < 4:
            continue
        dist = abs(detected_size - std)
        pct = dist / max(std, 1)
        if pct <= 0.25 and dist < best_dist:
            best_dist = dist
            best_std = std

    if best_std is None or best_std == detected_size:
        return detected_size, detected_offset

    # Compare quality scores: snap only if the standard size is at least
    # as good as the detected size.
    det_score = _grid_quality_score(rgb, detected_size, detected_offset)
    std_score = _grid_quality_score(rgb, best_std, (0, 0))

    # Standard size gets a 10% bonus (clean round numbers are preferred)
    if std_score * 1.10 >= det_score:
        logger.info(
            "Snapping cell_size %d → %d (quality %.2f → %.2f)",
            detected_size, best_std, det_score, std_score,
        )
        return best_std, (0, 0)

    return detected_size, detected_offset


def _extract_alpha_1x(
    alpha_channel: np.ndarray,
    cell_size: int,
    offset: Tuple[int, int],
    grid_h: int,
    grid_w: int,
) -> np.ndarray:
    """Extract 1x alpha by per-cell transparency majority (vectorized)."""
    cells = color_ops.cell_alpha_rigid(alpha_channel, cell_size, offset)
    # Match the reconstruction's grid exactly; missing edge cells are opaque.
    alpha_1x = np.full((grid_h, grid_w), 255, dtype=np.uint8)
    ch = min(grid_h, cells.shape[0])
    cw = min(grid_w, cells.shape[1])
    alpha_1x[:ch, :cw] = cells[:ch, :cw]
    return alpha_1x


def clean_sprite(
    img: np.ndarray,
    source_path: str = "",
    use_hough: bool = True,
    upscale_target: int = 256,
    max_core_diff: float = 8.0,
    max_cell_variance: float = MAX_CELL_VARIANCE,
) -> Optional[CleanedAsset]:
    """Clean a single sprite image into a pixel-perfect asset.

    Args:
        img: RGB or RGBA numpy array of the sprite.
        source_path: Path to original file (for logging).
        use_hough: Use Hough-based grid detection.
        upscale_target: Target size for the upscaled training version.
        max_core_diff: Maximum acceptable core diff percentage for strict
            comparison (used for clean NN-upscaled sprites).
        max_cell_variance: Maximum acceptable intra-cell variance.  This is
            the primary quality metric for AI-generated pixel art.

    Returns:
        CleanedAsset or None if cleaning failed/rejected.
    """
    has_alpha = img.shape[2] == 4
    alpha_channel = None

    if has_alpha:
        alpha_channel = img[:, :, 3]
        rgb = img[:, :, :3]
    else:
        rgb = img

    h, w = rgb.shape[:2]
    if h < 4 or w < 4:
        logger.debug("Sprite too small: %dx%d", w, h)
        return None

    # Step 1: FAST PATH for exact nearest-neighbor upscales. The detector
    # requires every block to be perfectly uniform, so the direct downsample
    # is the true 1x art by construction — no reconstruction needed. Fuzzy
    # near-NN images (AI renders, non-integer upscales) fall through to the
    # full reconstructor, which handles them correctly.
    nn_scale = grid_ops.detect_nn_scale(rgb)
    if nn_scale >= 2:
        logger.info("Exact %dx NN upscale in %s, using fast path",
                    nn_scale, source_path)
        image_1x = _downscale_nn(rgb, nn_scale)
        if alpha_channel is not None:
            # Alpha may be fuzzier than color; use per-cell majority.
            alpha_1x = color_ops.cell_alpha_rigid(alpha_channel, nn_scale, (0, 0))
            image_1x = np.dstack([image_1x, alpha_1x])

        h1, w1 = image_1x.shape[:2]
        up_scale = max(1, upscale_target // max(h1, w1))
        image_up = cv2.resize(image_1x, (w1 * up_scale, h1 * up_scale),
                              interpolation=cv2.INTER_NEAREST)

        return CleanedAsset(
            image_1x=image_1x,
            image_upscaled=image_up,
            cell_size=nn_scale,
            offset=(0, 0),
            metrics={"clean_ratio": 1.0, "method": "nn_fast_path",
                      "percent_diff_core": 0.0,
                      "intra_cell_variance": 0.0},
            source_path=source_path,
            warnings=[],
        )
    # end NN detection

    # Step 2: run pixel-perfecter reconstruction (expects RGB)
    try:
        rec = PixelArtReconstructor(image=rgb, debug=False)
        rec.use_hough = use_hough
        result = rec.run(mode="global")
    except Exception as e:
        logger.warning("Reconstruction failed for %s: %s", source_path, e)
        return None

    detected_size = rec.cell_size
    detected_offset = rec.offset if rec.offset else (0, 0)
    # Mesh reconstructions use variable cell widths (non-integer upscales);
    # rigid-grid refinement and fallbacks below do not apply to them.
    is_mesh = bool(rec.last_metrics and rec.last_metrics.get("grid_kind") == "mesh")

    if detected_size is None or detected_size < 2:
        detected_size = 4
        detected_offset = (0, 0)

    # Step 2b: refine cell size using grid quality scoring against standard sizes
    if is_mesh:
        cell_size, offset = detected_size, detected_offset
    else:
        cell_size, offset = _pick_best_cell_size(rgb, detected_size, detected_offset)

    if cell_size < MIN_CELL_SIZE:
        logger.debug("No valid grid detected in %s (cell_size=%s)", source_path, cell_size)
        return None

    if cell_size != detected_size:
        logger.info(
            "Refined cell_size %d → %d for %s",
            detected_size, cell_size, source_path,
        )
        # Re-run reconstruction with the refined cell size
        try:
            rec.cell_size = cell_size
            rec.offset = offset
            result = rec._empirical_pixel_reconstruction()
        except Exception as e:
            logger.warning("Re-reconstruction failed for %s: %s", source_path, e)
            return None

    # Step 3: evaluate quality using metrics from the reconstructor
    metrics = rec.last_metrics or {}

    # Ensure intra-cell variance is available (it's the primary quality metric)
    cell_variance = _compute_intra_cell_variance(rgb, cell_size, offset)
    metrics["intra_cell_variance"] = cell_variance
    metrics["grid_quality_score"] = _grid_quality_score(rgb, cell_size, offset)

    # Step 3b: if variance is too high, try all standard sizes as a fallback.
    # This helps with sprite sheets and images where the auto-detector picks
    # a non-standard size that doesn't align well.  (Rigid grids only — a
    # mesh's rigid-equivalent variance is misleading by construction.)
    if not is_mesh and cell_variance > max_cell_variance:
        best_fallback = None
        for cs in STANDARD_CELL_SIZES:
            if cs < MIN_CELL_SIZE or h // cs < 4 or w // cs < 4:
                continue
            fb_var = _compute_intra_cell_variance(rgb, cs, (0, 0))
            if fb_var < cell_variance and fb_var <= max_cell_variance:
                if best_fallback is None or fb_var < best_fallback[1]:
                    best_fallback = (cs, fb_var)

        if best_fallback is not None:
            fallback_cs, fallback_var = best_fallback
            logger.info(
                "Fallback: cell_size %d (var=%.0f) → %d (var=%.0f) for %s",
                cell_size, cell_variance, fallback_cs, fallback_var, source_path,
            )
            cell_size = fallback_cs
            offset = (0, 0)
            cell_variance = fallback_var
            # Re-run reconstruction with fallback size
            try:
                rec.cell_size = cell_size
                rec.offset = offset
                result = rec._empirical_pixel_reconstruction()
            except Exception:
                pass
            metrics["intra_cell_variance"] = cell_variance
            metrics["grid_quality_score"] = _grid_quality_score(rgb, cell_size, offset)
            metrics["method"] = "standard_fallback"

    cell_variance = metrics.get("intra_cell_variance", float("inf"))
    core_diff = metrics.get("percent_diff_core", 100.0)
    tolerant_core_diff = metrics.get("percent_diff_core_tolerant", core_diff)

    if is_mesh:
        # Rigid intra-cell variance doesn't describe a variable mesh fit;
        # gate mesh reconstructions on the tolerant diff instead.
        cell_variance = min(cell_variance, max_cell_variance) \
            if tolerant_core_diff <= 50.0 else cell_variance
        metrics["method"] = "mesh"

    warnings = []
    if isinstance(metrics.get("warnings"), list):
        warnings = metrics["warnings"]
    elif isinstance(metrics.get("warnings"), str):
        warnings = [metrics["warnings"]] if metrics["warnings"] else []

    # If cell_size == 1 after NN downscale, the image is already at 1x
    if cell_size == 1 and nn_scale > 1:
        image_1x = rgb.copy()
        if alpha_channel is not None:
            alpha_1x = np.where(alpha_channel > 127, 255, 0).astype(np.uint8)
            image_1x = np.dstack([image_1x, alpha_1x])

        h1, w1 = image_1x.shape[:2]
        up_scale = max(1, upscale_target // max(h1, w1))
        image_up = cv2.resize(image_1x, (w1 * up_scale, h1 * up_scale),
                              interpolation=cv2.INTER_NEAREST)
        return CleanedAsset(
            image_1x=image_1x,
            image_upscaled=image_up,
            cell_size=nn_scale,
            offset=(0, 0),
            metrics=metrics,
            source_path=source_path,
            warnings=warnings,
        )

    # Quality gate: accept based on EITHER strict diff (for clean sprites)
    # OR intra-cell variance (for AI-generated content with anti-aliasing).
    strict_pass = core_diff <= max_core_diff
    variance_pass = cell_variance <= max_cell_variance
    tolerant_pass = tolerant_core_diff <= 50.0  # within tolerance after ±20 channel

    if not strict_pass and not variance_pass:
        logger.info(
            "Rejecting %s: core_diff=%.1f%%, cell_variance=%.1f (thresholds: %.1f%%, %.1f)",
            source_path, core_diff, cell_variance, max_core_diff, max_cell_variance,
        )
        return None

    if not strict_pass:
        logger.info(
            "Accepting %s via variance metric: cell_variance=%.1f, "
            "tolerant_core_diff=%.1f%%, strict_core_diff=%.1f%%",
            source_path, cell_variance, tolerant_core_diff, core_diff,
        )
        metrics["method"] = "variance_accepted"

    # Step 4: extract 1x pixel art
    # The reconstructor returns a small RGB image (num_cells_y x num_cells_x
    # x 3) which IS the 1x pixel art.
    if result is None:
        return None

    image_1x = result

    # handle alpha: per-cell transparency majority over the same grid
    grid_h, grid_w = image_1x.shape[:2]
    if alpha_channel is not None:
        if is_mesh and getattr(rec, "mesh_lines", None):
            alpha_1x = color_ops.cell_alpha_mesh(alpha_channel, rec.mesh_lines)
            if alpha_1x.shape != (grid_h, grid_w):  # defensive
                alpha_1x = np.full((grid_h, grid_w), 255, dtype=np.uint8)
        else:
            alpha_1x = _extract_alpha_1x(alpha_channel, cell_size, offset, grid_h, grid_w)
        image_1x = np.dstack([image_1x, alpha_1x])

    # account for NN downscale in reported cell size
    effective_cell = cell_size * nn_scale if nn_scale > 1 else cell_size

    # Step 5: create upscaled version for training
    h1, w1 = image_1x.shape[:2]
    scale = max(1, upscale_target // max(h1, w1))
    if scale < 1:
        scale = 1
    image_up = cv2.resize(image_1x, (w1 * scale, h1 * scale),
                          interpolation=cv2.INTER_NEAREST)

    return CleanedAsset(
        image_1x=image_1x,
        image_upscaled=image_up,
        cell_size=effective_cell,  # report original cell size
        offset=offset,
        metrics=metrics,
        source_path=source_path,
        warnings=warnings,
    )


def bulk_clean(
    sprite_dir: Path,
    output_dir: Path,
    use_hough: bool = True,
    upscale_target: int = 256,
    max_core_diff: float = 8.0,
) -> List[Tuple[Path, CleanedAsset]]:
    """Process all sprites in a directory.

    Returns list of (output_path, CleanedAsset) for successful cleanings.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
    files = sorted(
        p for p in sprite_dir.iterdir()
        if p.suffix.lower() in extensions
    )

    for fpath in files:
        try:
            pil = Image.open(fpath)
            if pil.mode == "RGBA":
                img = np.array(pil)
            elif pil.mode == "P":
                img = np.array(pil.convert("RGBA"))
            else:
                img = np.array(pil.convert("RGB"))
        except Exception as e:
            logger.warning("Failed to load %s: %s", fpath, e)
            continue

        asset = clean_sprite(
            img,
            source_path=str(fpath),
            use_hough=use_hough,
            upscale_target=upscale_target,
            max_core_diff=max_core_diff,
        )

        if asset is None:
            logger.info("Skipped %s (cleaning failed or rejected)", fpath.name)
            continue

        # save 1x and upscaled versions
        stem = fpath.stem
        out_1x = output_dir / f"{stem}_1x.png"
        out_up = output_dir / f"{stem}_up.png"

        channels = asset.image_1x.shape[2] if len(asset.image_1x.shape) == 3 else 1
        if channels == 4:
            Image.fromarray(asset.image_1x, "RGBA").save(out_1x)
            Image.fromarray(asset.image_upscaled, "RGBA").save(out_up)
        else:
            Image.fromarray(asset.image_1x, "RGB").save(out_1x)
            Image.fromarray(asset.image_upscaled, "RGB").save(out_up)

        results.append((out_1x, asset))
        logger.info("Cleaned %s → %s (cell=%d, variance=%.1f, core_diff=%.1f%%)",
                     fpath.name, out_1x.name,
                     asset.cell_size,
                     asset.metrics.get("intra_cell_variance", -1),
                     asset.metrics.get("percent_diff_core", -1))

    return results
