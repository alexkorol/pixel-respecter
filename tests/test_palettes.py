"""Tests for target-palette snapping and color reduction."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from pixel_perfecter import palettes


def make_art(grid: int = 32, colors: int = 12, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    palette = rng.integers(0, 256, size=(colors, 3), dtype=np.uint8)
    return palette[rng.integers(0, colors, size=(grid, grid))]


def colors_of(img: np.ndarray) -> set:
    return {tuple(c) for c in img.reshape(-1, img.shape[-1])[:, :3]}


class TestGetPalette:
    def test_known_palettes(self):
        assert palettes.get_palette("pico8").shape == (16, 3)
        assert palettes.get_palette("db32").shape == (32, 3)
        assert palettes.get_palette("gameboy").shape == (4, 3)

    def test_name_normalization(self):
        assert np.array_equal(
            palettes.get_palette("PICO-8"), palettes.get_palette("pico8")
        )

    def test_unknown_palette_raises(self):
        with pytest.raises(KeyError, match="Unknown palette"):
            palettes.get_palette("commodore")


class TestSnapToPalette:
    def test_output_colors_subset_of_palette(self):
        art = make_art()
        target = palettes.get_palette("pico8")
        snapped = palettes.snap_to_palette(art, target)
        assert snapped.shape == art.shape
        assert colors_of(snapped) <= {tuple(c) for c in target}

    def test_exact_palette_colors_are_fixed_points(self):
        target = palettes.get_palette("db32")
        art = target[np.arange(32) % 32].reshape(4, 8, 3)
        assert np.array_equal(palettes.snap_to_palette(art, target), art)

    def test_alpha_passthrough(self):
        art = make_art()
        alpha = np.full(art.shape[:2], 200, dtype=np.uint8)
        rgba = np.dstack([art, alpha])
        snapped = palettes.snap_to_palette(rgba, palettes.get_palette("gameboy"))
        assert snapped.shape == rgba.shape
        assert np.array_equal(snapped[:, :, 3], alpha)


class TestReduceColors:
    def test_reduces_to_at_most_n(self):
        art = make_art(colors=12)
        out = palettes.reduce_colors(art, 4)
        assert len(colors_of(out)) <= 4

    def test_noop_when_already_under(self):
        art = make_art(colors=3)
        assert np.array_equal(palettes.reduce_colors(art, 8), art)


class TestCliIntegration:
    def test_transparent_bg_flag(self, tmp_path):
        from pixel_perfecter.cli import main

        art = make_art()
        art[0, :] = art[-1, :] = art[:, 0] = art[:, -1] = (40, 42, 60)
        img = np.repeat(np.repeat(art, 10, axis=0), 10, axis=1)
        src = tmp_path / "sprite.png"
        Image.fromarray(img).save(src)
        out_dir = tmp_path / "out"
        main([
            str(src), "-o", str(out_dir), "--transparent-bg",
            "--skip-overlays", "--no-metrics",
        ])
        result = np.array(Image.open(out_dir / "sprite_reconstructed.png"))
        assert result.shape[2] == 4
        border_alpha = result[0, :, 3]
        assert (border_alpha == 0).all()
        # interior non-background pixels stay opaque
        assert (result[:, :, 3] == 255).any()

    def test_palette_flag_constrains_output(self, tmp_path):
        from pixel_perfecter.cli import main

        art = make_art()
        img = np.repeat(np.repeat(art, 10, axis=0), 10, axis=1)
        src = tmp_path / "sprite.png"
        Image.fromarray(img).save(src)
        out_dir = tmp_path / "out"
        main([
            str(src), "-o", str(out_dir), "--palette", "gameboy",
            "--skip-overlays", "--no-metrics",
        ])
        result = np.array(Image.open(out_dir / "sprite_reconstructed.png"))
        target = {tuple(c) for c in palettes.get_palette("gameboy")}
        assert colors_of(result) <= target
        assert result.shape[0] == art.shape[0]
