"""Tests for sprite-sheet frame splitting on 1x art."""

from __future__ import annotations

import numpy as np
from PIL import Image

from pixel_perfecter import sheets


def make_sheet(bg=(40, 42, 60)) -> np.ndarray:
    """64x64 sheet: four 16x16 sprites with 8px gutters, reading order."""
    rng = np.random.default_rng(5)
    art = np.full((64, 64, 3), bg, dtype=np.uint8)
    palette = rng.integers(30, 226, size=(8, 3), dtype=np.uint8)
    positions = [(8, 8), (8, 40), (40, 8), (40, 40)]  # (y, x)
    for y, x in positions:
        art[y : y + 16, x : x + 16] = palette[rng.integers(0, 8, size=(16, 16))]
    return art


class TestSplitFrames:
    def test_finds_all_frames_in_reading_order(self):
        art = make_sheet()
        boxes = sheets.split_frames(art)
        assert len(boxes) == 4
        # reading order: row 1 left, row 1 right, row 2 left, row 2 right
        centers = [((y0 + y1) // 2, (x0 + x1) // 2) for x0, y0, x1, y1 in boxes]
        assert centers == sorted(centers, key=lambda c: (c[0] // 8, c[1]))
        for x0, y0, x1, y1 in boxes:
            assert 16 <= (x1 - x0) <= 18  # 16px sprite + padding
            assert 16 <= (y1 - y0) <= 18

    def test_alpha_sheet_uses_transparency(self):
        art = make_sheet()
        rgba = np.dstack([art, np.zeros(art.shape[:2], dtype=np.uint8)])
        rgba[8:24, 8:24, 3] = 255  # only one frame opaque
        boxes = sheets.split_frames(rgba)
        assert len(boxes) == 1

    def test_crop_frames_shapes(self):
        art = make_sheet()
        boxes = sheets.split_frames(art)
        frames = sheets.crop_frames(art, boxes)
        assert all(f.shape[2] == 3 for f in frames)
        assert all(
            f.shape[:2] == (y1 - y0, x1 - x0)
            for f, (x0, y0, x1, y1) in zip(frames, boxes)
        )

    def test_uniform_image_yields_no_frames(self):
        art = np.full((32, 32, 3), (10, 10, 10), dtype=np.uint8)
        assert sheets.split_frames(art) == []


class TestCliSplitFrames:
    def test_end_to_end(self, tmp_path):
        from pixel_perfecter.cli import main

        art = make_sheet()
        img = np.repeat(np.repeat(art, 10, axis=0), 10, axis=1)
        src = tmp_path / "sheet.png"
        Image.fromarray(img).save(src)
        out_dir = tmp_path / "out"
        main([
            str(src), "-o", str(out_dir), "--split-frames",
            "--skip-overlays", "--no-metrics",
        ])
        frames = sorted((out_dir / "sheet_frames").glob("frame_*.png"))
        assert len(frames) == 4
        sizes = {Image.open(f).size for f in frames}
        # every frame cut from the same shared 1x grid
        assert all(16 <= s <= 18 for size in sizes for s in size)
