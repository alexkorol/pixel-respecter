"""Unified Pixel Art Reconstruction Pipeline.

Detects the pixel grid of an upscaled / degraded pixel art image and
collapses every grid cell to its dominant color, recovering the art at its
native 1x resolution.

Detection gathers candidate cell sizes from several independent sources —
exact nearest-neighbor scale checking, Hough-based mesh detection,
autocorrelation of edge projections, and a phase-folding sweep — then scores
every candidate with a single quality metric (inter-cell contrast over
intra-cell variance). No single detector is trusted blindly; the best
explanation of the image wins. When the detected mesh is irregular (the
image was upscaled by a non-integer factor) reconstruction uses the variable
mesh directly instead of forcing a rigid grid.

The mesh detection chain and the quantize-before-downsample strategy follow
Kenneth J. Allen's proper-pixel-art
(https://github.com/KennethJAllen/proper-pixel-art, MIT license).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from . import colors as color_ops
from . import grid as grid_ops
from . import mesh as mesh_ops


def unsharp_prepass(
    img: np.ndarray, sigma: float = 2.0, amount: float = 1.5
) -> np.ndarray:
    """Unsharp-mask a soft source before reconstruction.

    Softly rendered "pixel art" (bloom-heavy or blurry AI output) loses cell
    detail to averaging; sharpening first recovers visibly crisper cells.
    Not applied automatically: on already-crisp sources it amplifies noise,
    and no current metric can tell the cases apart (the tolerant diff gets
    *worse* on sharpened sources even when the output looks better).
    """
    rgb = img[:, :, :3]
    blur = cv2.GaussianBlur(rgb, (0, 0), sigma)
    sharp = cv2.addWeighted(rgb, 1 + amount, blur, -amount, 0)
    out = np.clip(sharp, 0, 255).astype(np.uint8)
    if img.shape[2] == 4:
        return np.dstack([out, img[:, :, 3]])
    return out

# Refinement explores phase candidates around the folded-phase estimate.
_OFFSET_NEIGHBORHOOD = (-1, 0, 1)

# Candidate sizes whose quality score is within this fraction of the best
# are considered ties; the largest size (truest native resolution) wins.
_SCORE_TIE_FRACTION = 0.02

# A mesh reconstruction must beat the best rigid score by this factor to be
# chosen — rigid grids are simpler and downstream tooling prefers them.
_MESH_PREFERENCE_FACTOR = 1.10

# Candidates whose intra-cell variance exceeds this are not believable pixel
# grids (real winners measure well under 300 even for AI art; photographic
# content measures 600+ at every coarse size). Matches the cleaner's gate.
_VARIANCE_CEILING = 600.0

# Peakiness gate: the harmonic tie-break (and trusting the argmax at all) is
# only sound when the score curve actually peaks. best/median >= this over
# the gated candidates counts as peaked; flatter curves get the finer-pitch
# preference instead — on dense fine-pitched art the curve is nearly flat
# and the argmax lands on an arbitrary coarse size that destroys detail.
_PEAK_MEDIAN_FACTOR = 1.25

# Under flat evidence, prefer the finest candidate scoring at least this
# fraction of the best. A slight oversample preserves the art; an
# over-coarse pick is unrecoverable.
_FLAT_KEEP_FRACTION = 0.80

# An always-proposed fine candidate with intra-cell variance below this is
# just a clean subdivision of coarser art (uniform blocks split into uniform
# quarters score artificially high); it may not outvote the organic winner.
_SUBDIVISION_VARIANCE = 60.0


class PixelArtReconstructor:
    """Consolidated pixel art reconstruction pipeline.

    Three-step process:

    1. Candidate grid detection (NN-scale check, Hough mesh, autocorrelation,
       phase-fold sweep)
    2. Unified scoring — every candidate (cell size, offset) pair is ranked
       by the inter/intra cell quality metric
    3. Modal-color downsampling in palette label space (vectorized)
    """

    def __init__(
        self,
        image_path: Optional[str] = None,
        image: Optional[np.ndarray] = None,
        debug: bool = False,
    ):
        if image_path is None and image is None:
            raise ValueError("Either image_path or image must be provided.")
        self.image_path = image_path
        self.image = image.copy() if image is not None else None
        self.cell_size: Optional[int] = None
        self.offset: Optional[Tuple[int, int]] = None
        self.grid_result: Optional[np.ndarray] = None
        self.debug = debug
        self.last_metrics: Optional[dict] = None
        self.progress_callback: Optional[Callable[[int, int, str], None]] = None
        self.region_summaries: List[dict] = []
        self.mode: str = "global"
        self.use_hough: bool = True
        self.initial_upscale: int = 2
        self._hough_confidence: float = 0.0
        self.mesh_result: Optional[mesh_ops.MeshResult] = None
        self.mesh_lines: Optional[mesh_ops.Mesh] = None
        self._labels: Optional[np.ndarray] = None
        self._palette: Optional[np.ndarray] = None
        # Full candidate table from the last _fit_grid call (diagnostics).
        self.last_fit_debug: Optional[dict] = None
        self._edge_map: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def run(self, mode: str = "global") -> np.ndarray:
        """Execute the reconstruction pipeline.

        Args:
            mode: "global" fits one grid to the whole image and returns the
                1x reconstruction. "adaptive" fits per-region grids and
                returns a full-resolution composite.
        """
        self._ensure_image_loaded()
        mode = mode.lower()
        if mode not in {"global", "adaptive"}:
            raise ValueError(f"Unsupported reconstruction mode: {mode}")
        if self.debug:
            print(f"Processing {self.image_path or '<in-memory image>'} [{mode}]")
        if mode == "adaptive":
            return self._run_adaptive()
        return self._run_global()

    def debug_grid_detection(self, save_dir=None):
        """Save (or show) edge-projection and autocorrelation debug plots."""
        import matplotlib

        try:
            matplotlib.use("Agg")
        except Exception:
            pass
        import matplotlib.pyplot as plt
        import os

        self._ensure_image_loaded()
        edges = self._edges()
        h_proj = edges.sum(axis=0)
        v_proj = edges.sum(axis=1)

        def _autocorr(signal: np.ndarray) -> np.ndarray:
            signal = signal.astype(np.float64) - signal.mean()
            corr = np.correlate(signal, signal, mode="full")[len(signal) - 1 :]
            return corr

        fig, axs = plt.subplots(2, 2, figsize=(10, 6))
        axs[0, 0].plot(h_proj)
        axs[0, 0].set_title("Horizontal Edge Projection")
        axs[0, 1].plot(_autocorr(h_proj))
        axs[0, 1].set_title("Horizontal Autocorrelation")
        axs[1, 0].plot(v_proj)
        axs[1, 0].set_title("Vertical Edge Projection")
        axs[1, 1].plot(_autocorr(v_proj))
        axs[1, 1].set_title("Vertical Autocorrelation")
        name = os.path.basename(self.image_path) if self.image_path else "in_memory"
        fig.suptitle(f"Grid Debug: {name}")
        fig.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, name + "_grid_debug.png"))
            plt.close(fig)
        else:
            plt.show()

    # ------------------------------------------------------------------
    # loading and shared maps
    # ------------------------------------------------------------------

    def _ensure_image_loaded(self) -> None:
        if self.image is not None:
            return
        if self.image_path is None:
            raise ValueError("No image data available for reconstruction.")
        self.image = self._load_image()
        if self.image is None:
            raise ValueError(f"Could not load image: {self.image_path}")

    def _load_image(self) -> Optional[np.ndarray]:
        try:
            with Image.open(self.image_path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                return np.array(img)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Error loading image: {exc}")
            return None

    def _label_space(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._labels is None or self._palette is None:
            self._labels, self._palette = grid_ops.label_image(self.image)
        return self._labels, self._palette

    def _edges(self) -> np.ndarray:
        if self._edge_map is None:
            gray = cv2.cvtColor(self.image[:, :, :3], cv2.COLOR_RGB2GRAY)
            filtered = cv2.bilateralFilter(gray, d=5, sigmaColor=30, sigmaSpace=3)
            self._edge_map = cv2.Canny(filtered, 50, 150)
        return self._edge_map

    def _report_progress(self, current: int, total: int, label: str) -> None:
        if not self.progress_callback:
            return
        try:
            self.progress_callback(current, total, label)
        except Exception as exc:  # pylint: disable=broad-except
            if self.debug:
                print(f"DEBUG: Progress callback failed: {exc}")

    # ------------------------------------------------------------------
    # global pipeline
    # ------------------------------------------------------------------

    def _run_global(self) -> np.ndarray:
        best = self._fit_grid(self.image)
        self.cell_size = best["cell_size"]
        self.offset = best["offset"]
        self._hough_confidence = best.get("mesh_support", 0.0)
        self.mesh_result = best.get("mesh_result")
        self.mesh_lines = best.get("mesh_lines") if best.get("use_mesh") else None
        self.mode = "global"
        self.region_summaries = []

        if best.get("use_mesh"):
            labels, palette = self._label_space()
            reconstruction = mesh_ops.downsample_mesh(
                labels, palette, best["mesh_lines"]
            )
            grid_kind = "mesh"
        else:
            reconstruction = self._empirical_pixel_reconstruction()
            grid_kind = "rigid"

        self.grid_result = reconstruction
        self.last_metrics = self._evaluate_metrics_local(
            reconstruction, self.cell_size, self.offset
        )
        self.last_metrics["mode"] = "global"
        self.last_metrics["grid_kind"] = grid_kind
        self.last_metrics["regions"] = 1
        if best.get("no_grid"):
            # The believability gate judges rigid candidates by raw variance,
            # which false-alarms on noisy but genuinely grid-aligned art. The
            # final say goes to the round-trip: a reconstruction that explains
            # the source within tolerance IS a reliable grid.
            core_tol = self.last_metrics.get("percent_diff_core_tolerant", 100.0)
            if core_tol >= 15.0:
                self.last_metrics["no_grid"] = True
                self.last_metrics["warnings"].append(
                    "No reliable pixel grid detected - output is best-effort."
                )
        if self.debug:
            print(
                f"  cell={self.cell_size} offset={self.offset} kind={grid_kind} "
                f"score={best['score']:.1f}"
            )
        return reconstruction

    def _fit_grid(self, img: np.ndarray) -> Dict[str, object]:
        """Find the best grid for ``img``: candidates -> unified scoring."""
        h, w = img.shape[:2]
        # Boundary-mass projections: peaks sit exactly on grid lines.
        h_proj, v_proj = grid_ops.diff_projections(img)

        max_size = max(2, min(192, min(h, w) // 4))
        candidates: Dict[int, str] = {}

        def _add(size: float, source: str) -> None:
            size = int(round(size))
            if 2 <= size <= max_size and size not in candidates:
                candidates[size] = source

        # Source 1: exact nearest-neighbor scale (free certainty when clean).
        nn_scale = grid_ops.detect_nn_scale(img)
        if nn_scale > 1:
            _add(nn_scale, "nn")

        # Source 2: Hough mesh detection (proper-pixel-art chain).
        mesh_result: Optional[mesh_ops.MeshResult] = None
        if self.use_hough:
            gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY)
            filtered = cv2.bilateralFilter(gray, d=5, sigmaColor=30, sigmaSpace=3)
            mesh_result = mesh_ops.detect_mesh(
                filtered, upscale=max(1, self.initial_upscale)
            )
            if mesh_result is not None:
                _add(mesh_result.pixel_width, "mesh")
                _add(mesh_result.pixel_width / 2, "mesh/2")
                _add(mesh_result.pixel_width * 2, "mesh*2")

        # Source 3: autocorrelation peaks of the boundary projections.
        for period, _strength in grid_ops.autocorr_periods(
            [h_proj, v_proj], max_period=max_size
        ):
            _add(period, "autocorr")

        # Source 4: phase-fold sweep (cheap O(W+H) per size).
        sweep: List[Tuple[float, int]] = []
        for size in range(2, max_size + 1):
            _, align_x = grid_ops.fold_phase(h_proj, size)
            _, align_y = grid_ops.fold_phase(v_proj, size)
            sweep.append((align_x + align_y, size))
        sweep.sort(reverse=True)
        for _, size in sweep[:8]:
            _add(size, "fold")

        # Source 5: always-proposed fine pitches. Dense fine-pitched art
        # often has no autocorr/mesh peak at all, so nothing else proposes
        # the fine end of the range and selection degenerates to whichever
        # coarse sizes happened to be suggested. These can only win outright
        # or under the flat-evidence rule in the selection below.
        for size in range(3, 9):
            _add(size, "fine")

        if not candidates:
            return {
                "cell_size": 4, "offset": (0, 0), "score": 0.0,
                "use_mesh": False, "mesh_result": None, "mesh_support": 0.0,
            }

        # Unified scoring: every candidate size with a few phase offsets.
        scored: List[Tuple[float, int, Tuple[int, int]]] = []
        ordered = sorted(candidates)
        total = len(ordered)
        for idx, size in enumerate(ordered):
            self._report_progress(idx, total, f"Scoring {size}px grid")
            phase_x, _ = grid_ops.fold_phase(h_proj, size)
            phase_y, _ = grid_ops.fold_phase(v_proj, size)
            tried = set()
            best_for_size: Optional[Tuple[float, int, Tuple[int, int]]] = None
            for dx in _OFFSET_NEIGHBORHOOD:
                for dy in _OFFSET_NEIGHBORHOOD:
                    offset = ((phase_x + dx) % size, (phase_y + dy) % size)
                    if offset in tried:
                        continue
                    tried.add(offset)
                    score = self._score_candidate(img, size, offset)
                    entry = (score, size, offset)
                    if best_for_size is None or score > best_for_size[0]:
                        best_for_size = entry
            if (0, 0) not in tried:
                score = self._score_candidate(img, size, (0, 0))
                if best_for_size is None or score > best_for_size[0]:
                    best_for_size = (score, size, (0, 0))
            if best_for_size is not None:
                scored.append(best_for_size)
                if self.debug:
                    print(
                        f"DEBUG: size={size} ({candidates[size]}) "
                        f"offset={best_for_size[2]} score={best_for_size[0]:.1f}"
                    )
        self._report_progress(total, total, "Scoring complete")

        scored.sort(key=lambda item: item[0], reverse=True)

        # Believability gate: a winning grid must leave cells reasonably
        # uniform. Photographic / non-grid content scores arbitrarily by
        # contrast alone and would otherwise crown giant cells.
        no_grid = False
        gated = []
        variances: Dict[Tuple[int, Tuple[int, int]], float] = {}
        for score, size, offset in scored:
            var = self._candidate_variance(img, size, offset)
            variances[(size, offset)] = var
            if var <= _VARIANCE_CEILING:
                gated.append((score, size, offset))
        self.last_fit_debug = {
            "candidates": [
                {
                    "size": size,
                    "source": candidates.get(size, "?"),
                    "offset": list(offset),
                    "score": score,
                    "variance": variances[(size, offset)],
                    "gated_out": variances[(size, offset)] > _VARIANCE_CEILING,
                }
                for score, size, offset in scored
            ],
            "mesh_candidates": [],
        }
        if not gated:
            # Nothing believable: not grid-aligned pixel art. Degrade
            # gracefully to the most uniform (finest-explaining) candidate.
            # The always-proposed fine sizes don't compete here: they are
            # trivially uniform and would mask the no-grid condition.
            no_grid = True
            organic_scored = [
                item for item in scored if candidates.get(item[1]) != "fine"
            ]
            gated = sorted(
                organic_scored or scored,
                key=lambda item: (variances[(item[1], item[2])], item[1]),
            )[:1]

        # Organic candidates carry positive evidence from a detector; the
        # always-proposed fine sizes are a safety net and get less trust.
        organic_gated = [
            item for item in gated if candidates.get(item[1]) != "fine"
        ]
        median_score = float(np.median([item[0] for item in gated]))
        peaked = (
            median_score > 0
            and gated[0][0] >= _PEAK_MEDIAN_FACTOR * median_score
        )

        def _harmonic_tiebreak(
            pool: List[Tuple[float, int, Tuple[int, int]]],
            base: Tuple[float, int, Tuple[int, int]],
        ) -> Tuple[float, int, Tuple[int, int]]:
            # Prefer the largest size whose score is within a whisker of the
            # base (5 vs 10 both score "perfect"; 10 is the native
            # resolution).
            pick = base
            for entry in pool:
                if (
                    entry[0] >= base[0] * (1 - _SCORE_TIE_FRACTION)
                    and entry[1] > pick[1]
                ):
                    pick = entry
            return pick

        # fine_spike: a fine candidate with genuine texture (not a clean
        # subdivision) that outscores every organic candidate. Adopted only
        # if no mesh explains the image better (checked further down).
        fine_spike: Optional[Tuple[float, int, Tuple[int, int]]] = None
        if no_grid:
            best_score, best_size, best_offset = gated[0]
            mesh_ref = best_score
        elif peaked:
            pool = organic_gated or gated
            base = max(pool, key=lambda item: item[0])
            best_score, best_size, best_offset = _harmonic_tiebreak(pool, base)
            mesh_ref = base[0]
            top = gated[0]
            if (
                candidates.get(top[1]) == "fine"
                and top[0] > base[0]
                and variances[(top[1], top[2])] > _SUBDIVISION_VARIANCE
            ):
                fine_spike = top
        else:
            # Flat curve: no size is meaningfully better than the rest, so
            # the argmax is arbitrary. Take the finest decent candidate.
            keep = [
                item
                for item in gated
                if item[0] >= gated[0][0] * _FLAT_KEEP_FRACTION and item[1] >= 3
            ] or gated[:1]
            best_score, best_size, best_offset = min(
                keep, key=lambda item: item[1]
            )
            mesh_ref = gated[0][0]
            # Flat, fine-only evidence is no evidence of a grid at all.
            no_grid = not organic_gated
        if self.debug:
            print(
                f"DEBUG: selection mode="
                f"{'no_grid' if no_grid else 'peaked' if peaked else 'flat'} "
                f"median={median_score:.2f} pick={best_size} "
                f"score={best_score:.2f}"
            )

        result: Dict[str, object] = {
            "cell_size": best_size,
            "offset": best_offset,
            "score": best_score,
            "no_grid": no_grid,
            "use_mesh": False,
            "mesh_lines": None,
            "mesh_result": mesh_result,
            "mesh_support": mesh_result.line_support if mesh_result else 0.0,
        }

        # Irregular grids: a non-integer upscale (cells alternating e.g.
        # 15/16 px) cannot be represented by any rigid (cell, offset) pair.
        # Build variable meshes and switch when one explains the image
        # decisively better than the best rigid grid.
        mesh_candidates: List[Tuple[str, mesh_ops.Mesh, float]] = []

        # Seed float-pitch estimation from the winner AND from the strongest
        # fine-pitch candidate: when a coarse harmonic wins the rigid vote,
        # the true (often fractional) pitch is still among the fine sizes.
        seed_sizes = {best_size}
        # ... and from the top-scoring organic candidate even when the
        # variance gate rejected it: noisy-but-real pitches (e.g. a 16px
        # grid the rigid gate can't believe) still seed a projection mesh.
        organic_scored_best = next(
            (item for item in scored if candidates.get(item[1]) != "fine"),
            None,
        )
        if organic_scored_best is not None:
            seed_sizes.add(organic_scored_best[1])
        fine = [item for item in gated if item[1] <= 48 and item[1] != best_size]
        if fine:
            seed_sizes.add(max(fine)[1])
        # Half the winner, too: a fractional pitch like 15.5 shows up in the
        # autocorr comb only as its 31/62/... multiples, so no fine rigid
        # candidate exists to seed from. Integer half-pitches are harmless -
        # the fractionality gate below discards them.
        if best_size >= 16:
            seed_sizes.add(best_size // 2)

        seen_pitches: set = set()
        for seed in seed_sizes:
            pitch_x = mesh_ops.estimate_float_pitch(h_proj, seed)
            pitch_y = mesh_ops.estimate_float_pitch(v_proj, seed)
            frac = max(abs(pitch_x - round(pitch_x)), abs(pitch_y - round(pitch_y)))
            key = (round(pitch_x, 1), round(pitch_y, 1))
            if frac > 0.03 and key not in seen_pitches:
                seen_pitches.add(key)
                proj_mesh = (
                    mesh_ops.projection_mesh(h_proj, pitch_x),
                    mesh_ops.projection_mesh(v_proj, pitch_y),
                )
                if len(proj_mesh[0]) > 3 and len(proj_mesh[1]) > 3:
                    pitch = (pitch_x + pitch_y) / 2
                    mesh_candidates.append(("projection", proj_mesh, pitch))

        if (
            mesh_result is not None
            and not mesh_result.is_regular
            and mesh_result.pixel_width <= max_size
            and len(mesh_result.lines_x) > 3
            and len(mesh_result.lines_y) > 3
        ):
            mesh_candidates.append(
                ("hough", mesh_result.mesh, mesh_result.pixel_width)
            )

        engaged_meshes: List[Tuple[float, mesh_ops.Mesh, float]] = []
        for kind, candidate_mesh, pitch in mesh_candidates:
            mesh_score = mesh_ops.mesh_quality_score(img, candidate_mesh)
            self.last_fit_debug["mesh_candidates"].append(
                {
                    "kind": kind,
                    "pitch": pitch,
                    "score": mesh_score,
                    "lines": [
                        [int(value) for value in candidate_mesh[0]],
                        [int(value) for value in candidate_mesh[1]],
                    ],
                }
            )
            if self.debug:
                print(
                    f"DEBUG: {kind} mesh score={mesh_score:.1f} "
                    f"(pitch={pitch:.2f}) vs ref {mesh_ref:.1f}"
                )
            if mesh_score > mesh_ref * _MESH_PREFERENCE_FACTOR:
                engaged_meshes.append((mesh_score, candidate_mesh, pitch))

        if engaged_meshes:
            # A mesh must decisively beat the rigid *evidence* (mesh_ref),
            # not the possibly-deliberately-finer rigid pick. Among engaged
            # meshes the best score wins; under flat evidence prefer the
            # finest pitch within the keep fraction (same policy as rigid).
            best_mesh_score = max(item[0] for item in engaged_meshes)
            if peaked or no_grid:
                mesh_score, candidate_mesh, pitch = max(
                    engaged_meshes, key=lambda item: item[0]
                )
            else:
                mesh_score, candidate_mesh, pitch = min(
                    (
                        item for item in engaged_meshes
                        if item[0] >= best_mesh_score * _FLAT_KEEP_FRACTION
                    ),
                    key=lambda item: item[2],
                )
            result.update(
                use_mesh=True,
                mesh_lines=candidate_mesh,
                score=mesh_score,
                cell_size=max(2, int(round(pitch))),
                offset=(
                    int(candidate_mesh[0][0]),
                    int(candidate_mesh[1][0]),
                ),
                # A mesh that beats the rigid evidence decisively IS a
                # reliable grid; the rigid believability gate no longer
                # applies to it.
                no_grid=False,
            )
        elif fine_spike is not None:
            spike = _harmonic_tiebreak(gated, fine_spike)
            result.update(
                cell_size=spike[1], offset=spike[2], score=spike[0]
            )

        self.last_fit_debug.update(
            median_score=median_score,
            peaked=peaked,
            rigid_pick={"size": best_size, "score": best_score},
            final_pick={
                "size": result["cell_size"],
                "score": result["score"],
                "use_mesh": result["use_mesh"],
                "no_grid": result["no_grid"],
            },
        )
        return result

    @staticmethod
    def _score_candidate(
        img: np.ndarray, size: int, offset: Tuple[int, int], max_dim: int = 768
    ) -> float:
        """Quality score on a bounded crop so cost stays constant."""
        h, w = img.shape[:2]
        ox, oy = offset
        if h > max_dim or w > max_dim:
            crop = img[oy : min(h, oy + max_dim), ox : min(w, ox + max_dim)]
            return grid_ops.grid_quality_score(crop, size, (0, 0))
        return grid_ops.grid_quality_score(img, size, offset)

    @staticmethod
    def _candidate_variance(
        img: np.ndarray, size: int, offset: Tuple[int, int], max_dim: int = 768
    ) -> float:
        """Intra-cell variance on the same bounded crop as the score."""
        h, w = img.shape[:2]
        ox, oy = offset
        if h > max_dim or w > max_dim:
            crop = img[oy : min(h, oy + max_dim), ox : min(w, ox + max_dim)]
            return grid_ops.intra_cell_variance(crop, size, (0, 0))
        return grid_ops.intra_cell_variance(img, size, offset)

    # ------------------------------------------------------------------
    # reconstruction primitives
    # ------------------------------------------------------------------

    def _empirical_pixel_reconstruction(self) -> np.ndarray:
        """Map each rigid grid cell to its modal color (one output pixel)."""
        if self.image is None:
            raise ValueError("Image must be loaded before reconstruction.")
        cell = int(self.cell_size or 4)
        offset = tuple(self.offset or (0, 0))
        labels, palette = self._label_space()
        return grid_ops.downsample_mode(labels, palette, cell, offset)

    def _refine_artifacts(self) -> np.ndarray:
        """Remove isolated single-pixel artifacts from the reconstruction.

        Conservative majority filter on the 1x art: a pixel is replaced only
        when at least 7 of its 8 neighbors agree on a different color.
        """
        if self.grid_result is None:
            return self.image
        art = self.grid_result.copy()
        h, w = art.shape[:2]
        if h < 3 or w < 3:
            return art

        packed = (
            (art[:, :, 0].astype(np.uint32) << 16)
            | (art[:, :, 1].astype(np.uint32) << 8)
            | art[:, :, 2].astype(np.uint32)
        )
        center = packed[1:-1, 1:-1]
        neighbors = np.stack([
            packed[dy : dy + h - 2, dx : dx + w - 2]
            for dy in (0, 1, 2)
            for dx in (0, 1, 2)
            if not (dy == 1 and dx == 1)
        ])
        # If >=7 of 8 neighbors share one color, that color matches at least
        # one of the first two neighbors, so two reference checks suffice.
        replacement = center.copy()
        replaced = np.zeros(center.shape, dtype=bool)
        for ref in (neighbors[0], neighbors[1]):
            count = (neighbors == ref[None]).sum(axis=0)
            hit = (count >= 7) & (ref != center) & ~replaced
            replacement[hit] = ref[hit]
            replaced |= hit

        out = art[1:-1, 1:-1].copy()
        out[..., 0] = (replacement >> 16) & 0xFF
        out[..., 1] = (replacement >> 8) & 0xFF
        out[..., 2] = replacement & 0xFF
        art[1:-1, 1:-1] = out
        return art

    # ------------------------------------------------------------------
    # adaptive pipeline
    # ------------------------------------------------------------------

    def _run_adaptive(self) -> np.ndarray:
        regions = self._segment_regions()
        if not regions:
            if self.debug:
                print("  No distinct regions found; falling back to global.")
            small = self._run_global()
            h, w = self.image.shape[:2]
            return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        height, width = self.image.shape[:2]
        composite = self.image.copy()
        summaries: List[dict] = []
        total = len(regions)

        for idx, region in enumerate(regions, 1):
            x0, y0, x1, y1 = region["bbox"]
            mask = region["mask"].astype(bool)
            patch = self.image[y0:y1, x0:x1]
            self._report_progress(idx - 1, total, f"Region {idx}/{total}")

            entry = {
                "index": idx,
                "status": "ok",
                "cell_size": None,
                "offset": None,
                "core_diff": None,
                "percent_diff": 0.0,
                "mean_error": 0.0,
                "area": int(np.count_nonzero(mask)),
            }
            try:
                local = PixelArtReconstructor(image=patch, debug=False)
                local.use_hough = self.use_hough
                local.initial_upscale = self.initial_upscale
                fit = local._fit_grid(patch)
                local.cell_size = fit["cell_size"]
                local.offset = fit["offset"]
                small = local._empirical_pixel_reconstruction()
                rec_up = cv2.resize(
                    small, (patch.shape[1], patch.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                diff_map = np.any(patch != rec_up, axis=-1)
                masked = max(int(np.count_nonzero(mask)), 1)
                percent = 100.0 * int(np.count_nonzero(diff_map & mask)) / masked
                error = np.abs(
                    patch.astype(np.int32) - rec_up.astype(np.int32)
                ).mean(axis=-1)
                entry.update(
                    cell_size=fit["cell_size"],
                    offset=fit["offset"],
                    core_diff=percent,
                    percent_diff=percent,
                    mean_error=float(error[mask].mean()) if masked else 0.0,
                )
                target = composite[y0:y1, x0:x1]
                target[mask] = rec_up[mask]
                composite[y0:y1, x0:x1] = target
            except Exception as exc:  # pylint: disable=broad-except
                if self.debug:
                    print(f"  [WARN] Region {idx} failed: {exc}")
                entry["status"] = "fallback"
            summaries.append(entry)

        self.cell_size = None
        self.offset = None
        self.mesh_lines = None
        self.region_summaries = summaries
        self.mode = "adaptive"
        self.grid_result = composite

        try:
            self.last_metrics = self._evaluate_metrics_local(composite, 1, (0, 0))
        except Exception:  # pylint: disable=broad-except
            self.last_metrics = None
        if self.last_metrics is not None:
            self.last_metrics["mode"] = "adaptive"
            self.last_metrics["regions"] = len(summaries)

        self._report_progress(total, total, "Refinement complete")
        if self.debug:
            print(f"  Processed {total} region(s).")
        return composite

    def _segment_regions(self) -> List[dict]:
        """Connected components over dilated edges -> candidate regions."""
        self._ensure_image_loaded()
        edges = self._edges()
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        num_labels, labels = cv2.connectedComponents(dilated)
        height, width = edges.shape
        regions: List[dict] = []
        for label in range(1, num_labels):
            mask = labels == label
            if int(mask.sum()) < 64:
                continue
            ys, xs = np.where(mask)
            y0 = max(int(ys.min()) - 2, 0)
            y1 = min(int(ys.max()) + 3, height)
            x0 = max(int(xs.min()) - 2, 0)
            x1 = min(int(xs.max()) + 3, width)
            if (y1 - y0) < 8 or (x1 - x0) < 8:
                continue
            regions.append({"bbox": (x0, y0, x1, y1), "mask": mask[y0:y1, x0:x1]})
        regions.sort(key=lambda seg: -int(np.count_nonzero(seg["mask"])))
        return regions

    # ------------------------------------------------------------------
    # metrics
    # ------------------------------------------------------------------

    def _evaluate_metrics_local(
        self, reconstruction: np.ndarray, cell_size: int, offset: Tuple[int, int]
    ) -> dict:
        """Compute difference metrics without reloading images from disk."""
        if self.image is None:
            raise ValueError("Image must be loaded before evaluating metrics.")
        original = self.image
        h, w = original.shape[:2]
        rec_up = cv2.resize(reconstruction, (w, h), interpolation=cv2.INTER_NEAREST)

        cell_size = max(int(cell_size or 1), 1)
        offset = tuple(offset or (0, 0))

        diff_ctx = _build_diff_context(original, rec_up, cell_size, offset)
        metrics = _compute_overlay_metrics(diff_ctx, cell_size, offset)

        tolerant_ctx = _build_diff_context(
            original, rec_up, cell_size, offset, tolerance=20
        )
        tolerant = _compute_overlay_metrics(tolerant_ctx, cell_size, offset)
        metrics["percent_diff_core_tolerant"] = tolerant["percent_diff_core"]
        metrics["percent_diff_total_tolerant"] = tolerant["percent_diff_total"]
        # Warnings must come from the tolerant diff: strict exact-color
        # mismatch is ~100% on any noisy/anti-aliased source and says nothing
        # about alignment quality.
        metrics["warnings"] = tolerant["warnings"]
        metrics["intra_cell_variance"] = _compute_intra_cell_variance(
            original, cell_size, offset
        )
        return metrics


# ---------------------------------------------------------------------------
# module-level metric / diagnostic helpers (public contract)
# ---------------------------------------------------------------------------


def _compute_intra_cell_variance(
    original: np.ndarray, cell_size: int, offset: Tuple[int, int]
) -> float:
    """Mean intra-cell color variance (lower = better grid fit)."""
    return grid_ops.intra_cell_variance(original, max(int(cell_size), 1), offset)


def _build_diff_context(
    original: np.ndarray,
    rec_up: np.ndarray,
    cell_size: int,
    offset: Tuple[int, int],
    tolerance: int = 0,
) -> Dict[str, np.ndarray]:
    """Masks separating halo and core regions for diagnostics (vectorized)."""
    return grid_ops.build_diff_masks(
        original, rec_up, max(int(cell_size), 1), offset, tolerance
    )


def _assemble_overlay(
    original: np.ndarray, overlay_view: np.ndarray, rec_up: np.ndarray
) -> np.ndarray:
    """Stack original, diagnostic overlay, and reconstruction panels."""
    return np.hstack([original, overlay_view, rec_up])


def _render_diagnostic_overlays(
    original: np.ndarray, rec_up: np.ndarray, diff_ctx: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """Generate multiple overlay variants for inspection."""
    overlays: Dict[str, np.ndarray] = {}

    raw_overlay = original.copy()
    raw_overlay[diff_ctx["diff"]] = [255, 0, 0]
    overlays["raw"] = _assemble_overlay(original, raw_overlay, rec_up)

    combined_overlay = original.copy()
    combined_overlay[diff_ctx["halo_diff"]] = [255, 255, 0]
    combined_overlay[diff_ctx["core_diff"]] = [255, 0, 0]
    overlays["combined"] = _assemble_overlay(original, combined_overlay, rec_up)

    halo_suppressed = original.copy()
    halo_suppressed[diff_ctx["core_diff"]] = [255, 0, 0]
    overlays["halo_suppressed"] = _assemble_overlay(original, halo_suppressed, rec_up)

    core_only = original.astype(np.float32)
    core_only[~diff_ctx["core_mask"]] *= 0.25
    core_only = np.clip(core_only, 0, 255).astype(np.uint8)
    core_only[diff_ctx["core_diff"]] = [255, 0, 0]
    overlays["core_only"] = _assemble_overlay(original, core_only, rec_up)

    return overlays


def _compute_overlay_metrics(
    diff_ctx: Dict[str, np.ndarray], cell_size: int, offset: Tuple[int, int]
) -> dict:
    """Difference statistics using halo/core separation."""
    total_pixels = diff_ctx["diff"].size
    total_diff = int(np.sum(diff_ctx["diff"]))
    core_pixels = int(np.sum(diff_ctx["core_mask"]))
    halo_pixels = int(np.sum(diff_ctx["halo_mask"]))
    outside_pixels = total_pixels - int(np.sum(diff_ctx["grid_mask"]))

    core_diff = int(np.sum(diff_ctx["core_diff"]))
    halo_diff = int(np.sum(diff_ctx["halo_diff"]))
    outside_diff = int(np.sum(diff_ctx["outside_diff"]))

    percent_total = 100.0 * total_diff / max(total_pixels, 1)
    percent_core = 100.0 * core_diff / max(core_pixels, 1) if core_pixels else 0.0
    percent_halo = 100.0 * halo_diff / max(halo_pixels, 1) if halo_pixels else 0.0
    percent_outside = (
        100.0 * outside_diff / max(outside_pixels, 1) if outside_pixels else 0.0
    )

    h, w = diff_ctx["diff"].shape
    grid_ratio = cell_size / max(1, min(h, w))

    warnings: List[str] = []
    if percent_core > 5.0:
        warnings.append(f"Core mismatch {percent_core:.1f}% - inspect alignment.")
    if percent_halo > 15.0 and percent_core < 3.0:
        warnings.append("Differences dominated by halo glow.")
    if percent_outside > 2.0:
        warnings.append("Differences detected outside projected grid.")
    if grid_ratio > 0.25:
        warnings.append(
            f"Grid size {cell_size} is large relative to image size {min(h, w)}."
        )
    if cell_size < 4:
        warnings.append("Grid size is very small (may indicate noise).")

    return {
        "cell_size": cell_size,
        "offset": offset,
        "percent_diff_total": float(percent_total),
        "percent_diff_core": float(percent_core),
        "percent_diff_halo": float(percent_halo),
        "percent_diff_outside": float(percent_outside),
        "percent_diff": float(percent_total),  # Backwards compatibility
        "grid_ratio": float(grid_ratio),
        "warnings": warnings,
    }


def build_validation_diagnostics(
    original_path: str,
    reconstructed: np.ndarray,
    cell_size: int,
    offset: Tuple[int, int],
) -> Dict[str, object]:
    """Produce overlays and metrics for validation/GUI tooling."""
    with Image.open(original_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        original = np.array(img)

    h, w = original.shape[:2]
    rec_up = cv2.resize(reconstructed, (w, h), interpolation=cv2.INTER_NEAREST)
    diff_ctx = _build_diff_context(original, rec_up, cell_size, offset)
    overlays = _render_diagnostic_overlays(original, rec_up, diff_ctx)
    metrics = _compute_overlay_metrics(diff_ctx, cell_size, offset)

    # As in _evaluate_metrics_local: warnings judge the tolerant diff, since
    # strict exact-color mismatch is ~100% on any noisy source.
    tolerant_ctx = _build_diff_context(original, rec_up, cell_size, offset,
                                       tolerance=20)
    tolerant = _compute_overlay_metrics(tolerant_ctx, cell_size, offset)
    metrics["percent_diff_core_tolerant"] = tolerant["percent_diff_core"]
    metrics["percent_diff_total_tolerant"] = tolerant["percent_diff_total"]
    metrics["warnings"] = tolerant["warnings"]

    return {
        "original": original,
        "upscaled_reconstruction": rec_up,
        "overlays": overlays,
        "diff_context": diff_ctx,
        "metrics": metrics,
    }


def create_validation_overlay(
    original_path: str,
    reconstructed: np.ndarray,
    cell_size: int,
    offset: Tuple[int, int],
    mode: str = "combined",
) -> np.ndarray:
    """Return a specific diagnostic overlay variant."""
    diagnostics = build_validation_diagnostics(
        original_path, reconstructed, cell_size, offset
    )
    overlays: Dict[str, np.ndarray] = diagnostics["overlays"]  # type: ignore[assignment]
    if mode not in overlays:
        mode = "combined"
    return overlays[mode]


def _evaluate_with_parameters(
    image_path: Path,
    cell_size: int,
    offset: Tuple[int, int],
    debug: bool,
) -> dict:
    """Evaluate reconstruction quality for explicit parameters."""
    candidate = PixelArtReconstructor(str(image_path), debug=debug)
    candidate._ensure_image_loaded()
    candidate.cell_size = cell_size
    candidate.offset = offset
    result = candidate._empirical_pixel_reconstruction()
    diagnostics = build_validation_diagnostics(
        str(image_path), result, cell_size, offset
    )
    overlays: Dict[str, np.ndarray] = diagnostics["overlays"]  # type: ignore[assignment]
    metrics = diagnostics["metrics"]  # type: ignore[assignment]
    return {
        "result": result,
        "overlay": overlays["combined"],
        "overlays": overlays,
        "metrics": metrics,
        "reconstructor": candidate,
    }


def process_all_images(debug: bool = False, use_ml: bool = False, adaptive: bool = False):
    """Process all images in the input directory."""
    input_dir = Path("input")
    output_dir = Path("output")
    grid_debug_dir = output_dir / "grid_debug"

    output_dir.mkdir(exist_ok=True)
    grid_debug_dir.mkdir(exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    image_files = [
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    if not image_files:
        print("No image files found in input/ directory")
        return

    suggest_fn = None
    if use_ml:
        try:
            from ml.inference import suggest_parameters as _suggest_parameters  # type: ignore

            suggest_fn = _suggest_parameters
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[WARN] ML suggestions unavailable: {exc}")
            suggest_fn = None

    print(f"Found {len(image_files)} image(s) to process")
    metrics_records: List[dict] = []

    for image_file in image_files:
        try:
            reconstructor = PixelArtReconstructor(str(image_file), debug=debug)
            mode = "adaptive" if adaptive else "global"
            result = reconstructor.run(mode=mode)

            cell_size_for_metrics = reconstructor.cell_size or 1
            offset_for_metrics = reconstructor.offset or (0, 0)
            diagnostics = build_validation_diagnostics(
                str(image_file), result, cell_size_for_metrics, offset_for_metrics
            )
            overlays: Dict[str, np.ndarray] = diagnostics["overlays"]  # type: ignore[assignment]
            metrics = diagnostics["metrics"]  # type: ignore[assignment]

            best = {
                "result": result,
                "overlay": overlays["combined"],
                "overlays": overlays,
                "metrics": metrics,
                "reconstructor": reconstructor,
            }
            ml_applied = False

            if suggest_fn and not adaptive:
                suggestions = suggest_fn(str(image_file))
                for suggestion in suggestions:
                    candidate = _evaluate_with_parameters(
                        image_path=image_file,
                        cell_size=suggestion.cell_size,
                        offset=suggestion.offset,
                        debug=debug,
                    )
                    if (
                        candidate["metrics"]["percent_diff"] + 0.1
                        < best["metrics"]["percent_diff"]
                    ):
                        candidate["metrics"]["warnings"].append(
                            f"Applied ML suggestion (conf={suggestion.confidence:.2f})"
                        )
                        best = candidate
                        ml_applied = True

            result = best["result"]
            overlay = best["overlay"]
            metrics = best["metrics"]
            reconstructor = best["reconstructor"]

            output_path = output_dir / f"{image_file.stem}_reconstructed{image_file.suffix}"
            Image.fromarray(result).save(output_path)

            overlay_path = output_dir / f"{image_file.stem}_validation{image_file.suffix}"
            Image.fromarray(overlay).save(overlay_path)

            reconstructor.debug_grid_detection(save_dir=str(grid_debug_dir))

            print(f"Saved: {output_path} and {overlay_path}")
            if metrics["warnings"]:
                print("  [WARN] " + " ".join(metrics["warnings"]))
            else:
                print("  Output appears reasonable.")
            if ml_applied:
                print("  [INFO] ML suggestion applied.")

            metrics_records.append(
                {
                    "image": image_file.name,
                    "cell_size": metrics.get("cell_size"),
                    "offset_x": metrics.get("offset", (0, 0))[0],
                    "offset_y": metrics.get("offset", (0, 0))[1],
                    "percent_diff": round(metrics.get("percent_diff", 0.0), 4),
                    "percent_diff_core": round(metrics.get("percent_diff_core", 0.0), 4),
                    "percent_diff_halo": round(metrics.get("percent_diff_halo", 0.0), 4),
                    "percent_diff_outside": round(
                        metrics.get("percent_diff_outside", 0.0), 4
                    ),
                    "grid_ratio": round(metrics.get("grid_ratio", 0.0), 4),
                    "warnings": " | ".join(metrics.get("warnings", []))
                    if metrics.get("warnings")
                    else "",
                    "ml_used": "yes" if ml_applied else ("n/a" if adaptive else "no"),
                    "mode": reconstructor.mode,
                    "regions": len(reconstructor.region_summaries)
                    if reconstructor.region_summaries
                    else 1,
                }
            )

        except Exception as e:  # pylint: disable=broad-except
            print(f"Error processing {image_file}: {e}")

    if metrics_records:
        metrics_path = output_dir / "metrics.csv"
        with metrics_path.open("w", newline="") as csvfile:
            fieldnames = [
                "image",
                "cell_size",
                "offset_x",
                "offset_y",
                "percent_diff",
                "percent_diff_core",
                "percent_diff_halo",
                "percent_diff_outside",
                "grid_ratio",
                "warnings",
                "mode",
                "regions",
                "ml_used",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics_records)
        print(f"\nMetrics written to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process all images with pixel reconstructor."
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument(
        "--ml",
        action="store_true",
        help="Use ML suggestions to refine reconstructions when available.",
    )
    parser.add_argument(
        "--adaptive", action="store_true", help="Use adaptive per-region grids."
    )
    args = parser.parse_args()
    process_all_images(debug=args.debug, use_ml=args.ml, adaptive=args.adaptive)
