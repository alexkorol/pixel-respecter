"""Tests for the rewritten reconstruction core.

Covers the vectorized grid primitives, mesh detection (including
non-integer upscale factors), color/alpha handling, and the end-to-end
reconstructor contract on synthetic ground-truth sprites.
"""

from __future__ import annotations

import numpy as np
import pytest

from pixel_perfecter import colors as color_ops
from pixel_perfecter import grid as grid_ops
from pixel_perfecter import mesh as mesh_ops
from pixel_perfecter.reconstructor import (
    PixelArtReconstructor,
    _build_diff_context,
    _compute_intra_cell_variance,
    unsharp_prepass,
)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def make_art(grid: int = 32, colors: int = 12, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    palette = rng.integers(0, 256, size=(colors, 3), dtype=np.uint8)
    return palette[rng.integers(0, colors, size=(grid, grid))]


def upscale(art: np.ndarray, scale: int) -> np.ndarray:
    return np.repeat(np.repeat(art, scale, axis=0), scale, axis=1)


def upscale_fractional(art: np.ndarray, pitch: float) -> np.ndarray:
    """Upscale by a non-integer factor via rounded cumulative boundaries."""
    n = art.shape[0]
    bounds = np.round(np.arange(n + 1) * pitch).astype(int)
    idx = np.repeat(np.arange(n), np.diff(bounds))
    return art[np.ix_(idx, idx)]


def make_flat_bg_art(grid: int = 64, sprite: int = 24, seed: int = 7) -> np.ndarray:
    """Sprite centered on a large flat background (typical AI output)."""
    rng = np.random.default_rng(seed)
    palette = rng.integers(0, 256, size=(12, 3), dtype=np.uint8)
    art = np.full((grid, grid, 3), (40, 42, 60), dtype=np.uint8)
    lo = (grid - sprite) // 2
    art[lo : lo + sprite, lo : lo + sprite] = palette[
        rng.integers(0, 12, size=(sprite, sprite))
    ]
    return art


# ---------------------------------------------------------------------------
# grid primitives
# ---------------------------------------------------------------------------


class TestLabelImage:
    def test_exact_small_palette(self):
        art = make_art(grid=8, colors=5)
        labels, palette = grid_ops.label_image(art)
        assert len(palette) <= 5 * 2
        assert np.array_equal(palette[labels], art)

    def test_quantizes_large_palettes(self):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        labels, palette = grid_ops.label_image(img, max_colors=64)
        assert len(palette) <= 256
        assert labels.shape == (64, 64)


class TestDetectNNScale:
    def test_exact_scales(self):
        art = make_art(grid=16)
        for scale in (2, 3, 5, 8):
            assert grid_ops.detect_nn_scale(upscale(art, scale)) == scale

    def test_returns_largest_uniform_scale(self):
        art = make_art(grid=16)
        # 2x2-uniform art upscaled 3x is a true 6x upscale of the 1x art
        chunky = upscale(art, 2)
        assert grid_ops.detect_nn_scale(upscale(chunky, 3)) == 6

    def test_noise_is_not_nn(self):
        rng = np.random.default_rng(1)
        img = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        assert grid_ops.detect_nn_scale(img) == 1

    def test_blur_is_not_nn(self):
        import cv2

        img = cv2.GaussianBlur(upscale(make_art(), 8), (3, 3), 1.0)
        assert grid_ops.detect_nn_scale(img) == 1


class TestDownsampleMode:
    def test_exact_roundtrip(self):
        art = make_art()
        labels, palette = grid_ops.label_image(upscale(art, 6))
        out = grid_ops.downsample_mode(labels, palette, 6, (0, 0))
        assert np.array_equal(out, art)

    def test_majority_wins_with_noise(self):
        art = make_art(grid=8, colors=4)
        img = upscale(art, 8)
        # corrupt one corner pixel per cell; mode must still win
        img[::8, ::8] = (255, 255, 255)
        labels, palette = grid_ops.label_image(img)
        out = grid_ops.downsample_mode(labels, palette, 8, (0, 0))
        assert np.array_equal(out, art)

    def test_offset_grid(self):
        art = make_art()
        img = upscale(art, 7)[3:, 5:]
        labels, palette = grid_ops.label_image(img)
        out = grid_ops.downsample_mode(labels, palette, 7, (2, 4))
        assert np.array_equal(out, art[1:, 1:])


class TestGridQualityScore:
    def test_native_size_beats_harmonics(self):
        img = upscale(make_art(), 10)
        s_native = grid_ops.grid_quality_score(img, 10)
        s_half = grid_ops.grid_quality_score(img, 5)
        s_double = grid_ops.grid_quality_score(img, 20)
        assert s_native > s_half
        assert s_native > s_double

    def test_correct_offset_beats_wrong(self):
        img = upscale(make_art(), 9)[4:, 4:]
        s_right = grid_ops.grid_quality_score(img, 9, (5, 5))
        s_wrong = grid_ops.grid_quality_score(img, 9, (0, 0))
        assert s_right > s_wrong


class TestFoldPhase:
    def test_recovers_grid_phase(self):
        img = upscale(make_art(), 9)[4:, 4:]  # boundaries at 5, 14, 23, ...
        proj_x, _ = grid_ops.diff_projections(img)
        phase, alignment = grid_ops.fold_phase(proj_x, 9)
        assert phase == 5
        assert alignment > 2.0


class TestIntraCellVarianceVectorized:
    def test_zero_for_perfect_grid(self):
        img = upscale(make_art(), 8)
        assert _compute_intra_cell_variance(img, 8, (0, 0)) == 0.0

    def test_higher_for_wrong_size(self):
        img = upscale(make_art(), 8)
        right = _compute_intra_cell_variance(img, 8, (0, 0))
        wrong = _compute_intra_cell_variance(img, 7, (0, 0))
        assert wrong > right


class TestDiffContext:
    def test_mask_partition(self):
        original = upscale(make_art(grid=8), 8)
        ctx = _build_diff_context(original, original.copy(), 8, (0, 0))
        # grid splits into core + halo exactly
        assert np.array_equal(ctx["core_mask"] | ctx["halo_mask"], ctx["grid_mask"])
        assert not np.any(ctx["core_mask"] & ctx["halo_mask"])
        assert not np.any(ctx["diff"])

    def test_tolerance_forgives_small_deltas(self):
        original = upscale(make_art(grid=8), 8)
        shifted = np.clip(original.astype(np.int16) + 10, 0, 255).astype(np.uint8)
        strict = _build_diff_context(original, shifted, 8, (0, 0))
        tolerant = _build_diff_context(original, shifted, 8, (0, 0), tolerance=20)
        assert strict["diff"].sum() > 0
        assert tolerant["diff"].sum() == 0


# ---------------------------------------------------------------------------
# mesh
# ---------------------------------------------------------------------------


class TestMeshHelpers:
    def test_cluster_lines_merges_near_duplicates(self):
        assert mesh_ops.cluster_lines([0, 1, 2, 50, 51, 100], threshold=4) == [1, 50, 100]

    def test_homogenize_fills_missing_lines(self):
        # detected every other line of a 10px grid
        lines = [0, 20, 40, 60]
        completed = mesh_ops.homogenize_lines(lines, 10.0)
        assert completed == [0, 10, 20, 30, 40, 50, 60]

    def test_homogenize_no_drift_on_fractional_pitch(self):
        lines = [0, 155]  # 10 cells of 15.5
        completed = mesh_ops.homogenize_lines(lines, 15.5)
        gaps = np.diff(completed)
        assert set(gaps.tolist()) <= {15, 16}
        assert len(completed) == 11

    def test_estimate_float_pitch(self):
        art = make_art()
        n = art.shape[0]
        for pitch in (15.5, 9.7):
            img = upscale_fractional(art, pitch)
            # what the rounded boundaries actually realize on average
            realized = round(n * pitch) / n
            proj_x, _ = grid_ops.diff_projections(img)
            est = mesh_ops.estimate_float_pitch(proj_x, int(round(pitch)))
            # sub-pixel accurate (integer detection is only +-0.5) and not
            # locked to an integer
            assert abs(est - realized) < 0.1, f"pitch {pitch}: estimated {est}"
            assert abs(est - round(est)) > 0.1

    def test_projection_mesh_snaps_to_boundaries(self):
        art = make_art()
        img = upscale_fractional(art, 15.5)
        truth = np.round(np.arange(33) * 15.5).astype(int)
        proj_x, _ = grid_ops.diff_projections(img)
        lines = mesh_ops.projection_mesh(proj_x, 15.5)
        interior_truth = set(truth[1:-1].tolist())
        interior_lines = [l for l in lines if 0 < l < img.shape[1] - 1]
        hits = sum(1 for l in interior_lines if l in interior_truth)
        assert hits >= len(interior_truth) - 1

    def test_rigid_fit_drift_flags_fractional_pitch(self):
        regular = list(range(0, 160, 16))
        drift, _ = mesh_ops._rigid_fit_drift(regular)
        assert drift < 0.6
        fractional = np.round(np.arange(11) * 15.5).astype(int).tolist()
        drift_frac, _ = mesh_ops._rigid_fit_drift(fractional)
        assert drift_frac > 0.6

    def test_downsample_mesh_exact(self):
        art = make_art()
        img = upscale_fractional(art, 15.5)
        truth_lines = np.round(np.arange(33) * 15.5).astype(int).tolist()
        truth_lines[-1] = img.shape[0] - 1
        labels, palette = grid_ops.label_image(img)
        out = mesh_ops.downsample_mesh(labels, palette, (truth_lines, truth_lines))
        assert np.array_equal(out, art)


# ---------------------------------------------------------------------------
# colors
# ---------------------------------------------------------------------------


class TestColors:
    def test_cell_alpha_rigid_majority(self):
        alpha = np.zeros((16, 16), dtype=np.uint8)
        alpha[:8, :] = 255  # top half opaque
        cells = color_ops.cell_alpha_rigid(alpha, 8, (0, 0))
        assert cells.shape == (2, 2)
        assert cells[0].tolist() == [255, 255]
        assert cells[1].tolist() == [0, 0]

    def test_dominant_color_binned_resists_outliers(self):
        pixels = np.array([[200, 30, 40]] * 60 + [[10, 250, 10]] * 5, dtype=np.uint8)
        color = color_ops.dominant_color_binned(pixels)
        assert color == (200, 30, 40)

    def test_make_background_transparent(self):
        img = np.full((10, 10, 3), (1, 2, 3), dtype=np.uint8)
        img[4:6, 4:6] = (200, 100, 50)
        out = color_ops.make_background_transparent(img)
        assert out.shape[2] == 4
        assert out[0, 0, 3] == 0
        assert out[4, 4, 3] == 255

    def test_quantize_image_reduces_palette(self):
        rng = np.random.default_rng(3)
        img = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
        labels, palette = color_ops.quantize_image(img, num_colors=8)
        assert len(palette) <= 8
        assert labels.max() < len(palette)

    def test_flatten_alpha_keeps_opaque_pixels(self):
        rgba = np.zeros((8, 8, 4), dtype=np.uint8)
        rgba[2:6, 2:6] = (90, 120, 150, 255)
        flat = color_ops.flatten_alpha(rgba)
        assert flat.shape == (8, 8, 3)
        assert tuple(flat[3, 3]) == (90, 120, 150)
        assert tuple(flat[0, 0]) != (0, 0, 0)  # replaced, not left black


# ---------------------------------------------------------------------------
# end-to-end reconstructor contract
# ---------------------------------------------------------------------------


class TestReconstructorEndToEnd:
    def test_clean_integer_upscale_exact(self):
        art = make_art()
        rec = PixelArtReconstructor(image=upscale(art, 10))
        out = rec.run("global")
        assert rec.cell_size == 10
        assert np.array_equal(out, art)

    def test_cropped_offset_grid(self):
        art = make_art()
        rec = PixelArtReconstructor(image=upscale(art, 10)[3:, 3:])
        out = rec.run("global")
        assert rec.cell_size == 10
        assert rec.offset == (7, 7)
        assert np.array_equal(out, art[1:, 1:])

    @pytest.mark.parametrize("pitch", [15.5, 9.7])
    def test_fractional_upscale_uses_mesh(self, pitch):
        art = make_art()
        rec = PixelArtReconstructor(image=upscale_fractional(art, pitch))
        out = rec.run("global")
        assert rec.last_metrics["grid_kind"] == "mesh"
        assert out.shape == art.shape
        assert np.array_equal(out, art)

    def test_noisy_upscale_within_tolerance(self):
        rng = np.random.default_rng(11)
        art = make_art()
        img = upscale(art, 10).astype(np.float64)
        img = np.clip(img + rng.normal(0, 5, img.shape), 0, 255).astype(np.uint8)
        rec = PixelArtReconstructor(image=img)
        out = rec.run("global")
        assert rec.cell_size == 10
        assert out.shape == art.shape
        deltas = np.abs(out.astype(int) - art.astype(int))
        assert float(np.mean(np.all(deltas <= 10, axis=-1))) > 0.99

    def test_blurred_ai_style(self):
        import cv2

        art = make_art()
        img = cv2.GaussianBlur(upscale(art, 10), (3, 3), 1.0)
        rec = PixelArtReconstructor(image=img)
        out = rec.run("global")
        assert rec.cell_size == 10
        assert out.shape == art.shape

    def test_flat_background_sprite_keeps_fine_pitch(self):
        # Sprites on flat backgrounds used to crown giant cells: at the true
        # pitch most neighbor pairs are identical background (contrast
        # diluted to ~0), while a handful of giant cells straddling subject
        # and background scored huge contrast. The statistical-support
        # weight in the quality scores prevents that.
        rng = np.random.default_rng(3)
        art = make_flat_bg_art()
        img = upscale(art, 16).astype(np.float64)
        img = np.clip(img + rng.normal(0, 5, img.shape), 0, 255).astype(np.uint8)
        rec = PixelArtReconstructor(image=img)
        out = rec.run("global")
        assert rec.cell_size <= 32, f"giant pitch {rec.cell_size} won"
        assert out.shape[0] >= 32, "output collapsed to a handful of cells"
        assert not rec.last_metrics.get("no_grid", False)

    def test_score_prefers_fine_pitch_over_giant_cells(self):
        rng = np.random.default_rng(3)
        art = make_flat_bg_art()
        img = upscale(art, 16).astype(np.float64)
        img = np.clip(img + rng.normal(0, 5, img.shape), 0, 255).astype(np.uint8)
        fine = grid_ops.grid_quality_score(img, 16, (0, 0))
        giant = grid_ops.grid_quality_score(img, 176, (0, 0))
        assert fine > giant

    def test_mesh_score_prefers_fine_pitch_over_giant_cells(self):
        rng = np.random.default_rng(3)
        art = make_flat_bg_art()
        img = upscale(art, 16).astype(np.float64)
        img = np.clip(img + rng.normal(0, 5, img.shape), 0, 255).astype(np.uint8)
        n = img.shape[0]
        fine_mesh = (
            np.arange(0, n + 1, 16, dtype=float),
            np.arange(0, n + 1, 16, dtype=float),
        )
        giant_mesh = (
            np.arange(0, n + 1, 176, dtype=float),
            np.arange(0, n + 1, 176, dtype=float),
        )
        fine = mesh_ops.mesh_quality_score(img, fine_mesh)
        giant = mesh_ops.mesh_quality_score(img, giant_mesh)
        assert fine > giant

    def test_downsample_mode_banded_when_cells_exceed_budget(self, monkeypatch):
        # When cell count alone blows the bincount budget, downsample_mode
        # must band the computation instead of re-quantizing forever
        # (used to recurse infinitely / exhaust memory on fine-pitched or
        # huge images).
        art = make_art(grid=24)
        img = grid_ops.upscale_nn(art, 4)
        labels, palette = grid_ops.label_image(img)
        expected = grid_ops.downsample_mode(labels, palette, 4, (0, 0))
        monkeypatch.setattr(grid_ops, "MAX_BINCOUNT_CELLS", 64)
        banded = grid_ops.downsample_mode(labels, palette, 4, (0, 0))
        assert np.array_equal(banded, expected)

    def test_unsharp_prepass_recovers_blurred_grid(self):
        import cv2

        art = make_art()
        img = cv2.GaussianBlur(upscale(art, 10), (5, 5), 2.0)
        sharpened = unsharp_prepass(img)
        rec = PixelArtReconstructor(image=sharpened)
        out = rec.run("global")
        assert rec.cell_size == 10
        assert out.shape == art.shape

    def test_unsharp_prepass_preserves_alpha_and_shape(self):
        art = make_art()
        rgba = np.dstack([art, np.full(art.shape[:2], 77, dtype=np.uint8)])
        out = unsharp_prepass(rgba)
        assert out.shape == rgba.shape
        assert np.array_equal(out[:, :, 3], rgba[:, :, 3])

    def test_noisy_source_warnings_use_tolerant_diff(self):
        # Noise makes the strict exact-color diff ~100% even when the grid is
        # perfectly aligned; warnings must reflect the tolerant diff instead.
        rng = np.random.default_rng(11)
        art = make_art()
        img = upscale(art, 10).astype(np.float64)
        img = np.clip(img + rng.normal(0, 5, img.shape), 0, 255).astype(np.uint8)
        rec = PixelArtReconstructor(image=img)
        rec.run("global")
        metrics = rec.last_metrics
        assert metrics["percent_diff_core_tolerant"] < 5.0
        assert not any("Core mismatch" in w for w in metrics["warnings"])

    def test_metrics_contract(self):
        rec = PixelArtReconstructor(image=upscale(make_art(), 8))
        rec.run("global")
        metrics = rec.last_metrics
        for key in (
            "cell_size", "offset", "percent_diff", "percent_diff_core",
            "percent_diff_halo", "percent_diff_outside", "grid_ratio",
            "warnings", "percent_diff_core_tolerant", "intra_cell_variance",
            "mode", "regions", "grid_kind",
        ):
            assert key in metrics, f"missing metrics key: {key}"
        assert metrics["percent_diff"] == 0.0
        assert metrics["intra_cell_variance"] == 0.0

    def test_manual_parameters_respected(self):
        art = make_art()
        rec = PixelArtReconstructor(image=upscale(art, 6))
        rec.cell_size = 6
        rec.offset = (0, 0)
        out = rec._empirical_pixel_reconstruction()
        assert np.array_equal(out, art)

    def test_refine_artifacts_removes_isolated_pixel(self):
        art = np.full((9, 9, 3), (50, 90, 130), dtype=np.uint8)
        art[4, 4] = (250, 10, 10)
        rec = PixelArtReconstructor(image=upscale(art, 4))
        rec.grid_result = art
        cleaned = rec._refine_artifacts()
        assert tuple(cleaned[4, 4]) == (50, 90, 130)

    def test_refine_artifacts_preserves_checkerboard(self):
        art = np.zeros((8, 8, 3), dtype=np.uint8)
        art[(np.indices((8, 8)).sum(axis=0) % 2) == 1] = (255, 255, 255)
        rec = PixelArtReconstructor(image=upscale(art, 4))
        rec.grid_result = art
        assert np.array_equal(rec._refine_artifacts(), art)

    def test_adaptive_mode_smoke(self):
        art = make_art(grid=16)
        rec = PixelArtReconstructor(image=upscale(art, 8))
        out = rec.run("adaptive")
        assert out.shape == (128, 128, 3)
        assert rec.mode == "adaptive"
        assert isinstance(rec.region_summaries, list)

    def test_uniform_image_does_not_crash(self):
        img = np.full((64, 64, 3), (120, 130, 140), dtype=np.uint8)
        rec = PixelArtReconstructor(image=img)
        out = rec.run("global")
        assert out.ndim == 3


class TestPeakinessGate:
    """Selection under flat vs peaked score curves (wild-corpus fix).

    On dense fine-pitched art the quality score is nearly flat across
    pitches, so the argmax (and the largest-within-a-whisker harmonic
    tie-break) lands on an arbitrary coarse size that destroys the art.
    The gate detects the flat curve and prefers the finest decent pitch;
    fine sizes 3-8 are always proposed so that end of the range competes.
    """

    @staticmethod
    def _mixed_pitch_image() -> np.ndarray:
        """Two pitches side by side + noise/blur: flat multi-pitch evidence."""
        import cv2

        rng = np.random.default_rng(7)
        palette = rng.integers(0, 256, size=(12, 3), dtype=np.uint8)

        def art(pitch: int, n: int = 480, seed: int = 1) -> np.ndarray:
            r = np.random.default_rng(seed)
            g = n // pitch + 1
            a = palette[r.integers(0, 12, size=(g, g))]
            return np.repeat(np.repeat(a, pitch, 0), pitch, 1)[:n, :n]

        img = np.concatenate([art(6, seed=2), art(14, seed=3)], axis=1)
        img = img.astype(np.float64)
        img += np.random.default_rng(4).normal(0, 6, img.shape)
        return cv2.GaussianBlur(
            np.clip(img, 0, 255).astype(np.uint8), (3, 3), 1.2
        )

    def test_fine_pitches_always_proposed(self):
        # Coarse blocky art: no detector proposes anything below 16, so the
        # fine end of the range must be injected to compete at all.
        rng = np.random.default_rng(3)
        art = make_art(grid=8, colors=6, seed=3)
        img = upscale(art, 32).astype(np.float64)
        img = np.clip(img + rng.normal(0, 2, img.shape), 0, 255).astype(np.uint8)
        rec = PixelArtReconstructor(image=img)
        rec.run("global")
        proposed = {c["size"] for c in rec.last_fit_debug["candidates"]}
        assert set(range(3, 9)) <= proposed

    def test_flat_evidence_prefers_fine_pitch(self):
        rec = PixelArtReconstructor(image=self._mixed_pitch_image())
        rec.run("global")
        assert rec.last_fit_debug["peaked"] is False
        assert rec.cell_size <= 8, (
            f"flat evidence must err fine, picked {rec.cell_size}"
        )
        assert not rec.last_metrics.get("no_grid", False)

    def test_clean_subdivision_does_not_outvote_native(self):
        # 8x8 art upscaled x32: cells of 8 subdivide the uniform 32-blocks
        # with near-zero variance and an inflated score. The native 32 must
        # still win (fine sizes are a safety net, not a harmonic shortcut).
        rng = np.random.default_rng(3)
        art = make_art(grid=8, colors=6, seed=3)
        img = upscale(art, 32).astype(np.float64)
        img = np.clip(img + rng.normal(0, 2, img.shape), 0, 255).astype(np.uint8)
        rec = PixelArtReconstructor(image=img)
        rec.run("global")
        assert rec.cell_size == 32
        assert rec.last_fit_debug["peaked"] is True

    def test_requires_an_image(self):
        with pytest.raises(ValueError):
            PixelArtReconstructor()
