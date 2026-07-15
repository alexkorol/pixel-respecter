"""Tests for the ComfyUI node wrapper (numpy path; torch path auto-skips)."""

from __future__ import annotations

import numpy as np
import pytest

from comfyui_pixel_perfecter.nodes import (
    NODE_CLASS_MAPPINGS,
    PixelPerfecter,
    reconstruct_array,
)


def make_art(grid: int = 32, colors: int = 12, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    palette = rng.integers(0, 256, size=(colors, 3), dtype=np.uint8)
    return palette[rng.integers(0, colors, size=(grid, grid))]


class TestReconstructArray:
    def test_roundtrip_clean_upscale(self):
        art = make_art()
        img = np.repeat(np.repeat(art, 10, axis=0), 10, axis=1)
        native, upscaled, cell = reconstruct_array(img)
        assert cell == 10
        assert np.array_equal(native, art)
        assert upscaled.shape == img.shape
        # NN re-upscale of an exact reconstruction reproduces the input.
        assert np.array_equal(upscaled, img)

    def test_accepts_alpha_channel(self):
        art = make_art()
        img = np.repeat(np.repeat(art, 8, axis=0), 8, axis=1)
        rgba = np.dstack([img, np.full(img.shape[:2], 255, dtype=np.uint8)])
        native, _, cell = reconstruct_array(rgba)
        assert cell == 8
        assert np.array_equal(native, art)


class TestNodeContract:
    def test_mappings_expose_node(self):
        assert NODE_CLASS_MAPPINGS["PixelPerfecter"] is PixelPerfecter

    def test_input_and_return_types(self):
        inputs = PixelPerfecter.INPUT_TYPES()
        assert "image" in inputs["required"]
        assert PixelPerfecter.RETURN_TYPES == ("IMAGE", "IMAGE", "INT")
        assert len(PixelPerfecter.RETURN_NAMES) == 3
        assert callable(getattr(PixelPerfecter, PixelPerfecter.FUNCTION))

    def test_node_execution_with_torch(self):
        torch = pytest.importorskip("torch", exc_type=ImportError)
        art = make_art()
        img = np.repeat(np.repeat(art, 10, axis=0), 10, axis=1)
        tensor = torch.from_numpy(img[None].astype(np.float32) / 255.0)
        native, full, cell = PixelPerfecter().reconstruct(tensor)
        assert cell == 10
        assert native.shape == (1, 32, 32, 3)
        assert full.shape == tensor.shape
        out = (native[0].numpy() * 255.0).round().astype(np.uint8)
        assert np.array_equal(out, art)
