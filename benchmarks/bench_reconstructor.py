"""Benchmark the reconstructor on synthetic sprites.

Run before and after optimization to compare:
    python benchmarks/bench_reconstructor.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pixel_perfecter.reconstructor import PixelArtReconstructor


def make_sprite(grid: int = 32, cell: int = 10, colors: int = 12, seed: int = 7,
                noise: float = 0.0, scale_jitter: bool = False) -> np.ndarray:
    """Create a synthetic pixel-art image: `grid`x`grid` cells of size `cell`."""
    rng = np.random.default_rng(seed)
    palette = rng.integers(0, 256, size=(colors, 3), dtype=np.uint8)
    art = palette[rng.integers(0, colors, size=(grid, grid))]
    if scale_jitter:
        # Non-integer upscale: cells alternate between cell and cell+1 px wide
        xs = np.repeat(np.arange(grid), [cell + (i % 2) for i in range(grid)])
        img = art[np.ix_(xs, xs)]
    else:
        img = np.repeat(np.repeat(art, cell, axis=0), cell, axis=1)
    if noise > 0:
        jitter = rng.normal(0, noise * 255, img.shape)
        img = np.clip(img.astype(np.float64) + jitter, 0, 255).astype(np.uint8)
    return img


def bench(label: str, img: np.ndarray, runs: int = 3) -> None:
    times = []
    cell_sizes = []
    for _ in range(runs):
        rec = PixelArtReconstructor(image=img.copy(), debug=False)
        t0 = time.perf_counter()
        rec.run(mode="global")
        times.append(time.perf_counter() - t0)
        cell_sizes.append(rec.cell_size)
    print(f"{label:42s} best={min(times):7.3f}s  mean={np.mean(times):7.3f}s  "
          f"cell_size={cell_sizes[0]}")


if __name__ == "__main__":
    print(f"{'case':42s} {'best':>12s} {'mean':>14s}")
    bench("320px clean 10x upscale", make_sprite())
    bench("320px noisy (sigma=2%) 10x upscale", make_sprite(noise=0.02))
    bench("496px non-integer upscale", make_sprite(scale_jitter=True, cell=15))
    bench("128px clean 4x upscale", make_sprite(grid=32, cell=4))
