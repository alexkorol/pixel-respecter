"""Target-palette snapping for reconstructed pixel art.

Maps reconstructed 1x art onto a fixed hardware/community palette
(PICO-8, DawnBringer 32, Game Boy) or reduces it to at most N colors.
Matching happens in CIELAB via a unique-color lookup, so cost scales with
the palette of the art (dozens of colors), not its pixel count.

NES is deliberately absent: the 2C02 emits analog video and has no
canonical RGB palette — every emulator ships a different approximation
(see nesdev wiki, "PPU palettes"). Snapping to made-up values would be
worse than not offering it.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import cv2
import numpy as np

from . import colors as color_ops


def _hex(values: Iterable[str]) -> np.ndarray:
    return np.array(
        [[int(v[i : i + 2], 16) for i in (0, 2, 4)] for v in values],
        dtype=np.uint8,
    )


# PICO-8 system palette (lexaloffle.com/pico-8.php, colors 0-15).
_PICO8 = _hex([
    "000000", "1d2b53", "7e2553", "008751", "ab5236", "5f574f",
    "c2c3c7", "fff1e8", "ff004d", "ffa300", "ffec27", "00e436",
    "29adff", "83769c", "ff77a8", "ffccaa",
])

# DawnBringer 32 (pixeljoint.com/forum/forum_posts.asp?TID=16247).
_DB32 = _hex([
    "000000", "222034", "45283c", "663931", "8f563b", "df7126",
    "d9a066", "eec39a", "fbf236", "99e550", "6abe30", "37946e",
    "4b692f", "524b24", "323c39", "3f3f74", "306082", "5b6ee1",
    "639bff", "5fcde4", "cbdbfc", "ffffff", "9badb7", "847e87",
    "696a6a", "595652", "76428a", "ac3232", "d95763", "d77bba",
    "8f974a", "8a6f30",
])

# Original Game Boy (DMG) greens, dark to light.
_GAMEBOY = _hex(["0f380f", "306230", "8bac0f", "9bbc0f"])

PALETTES: Dict[str, np.ndarray] = {
    "pico8": _PICO8,
    "db32": _DB32,
    "gameboy": _GAMEBOY,
}


def get_palette(name: str) -> np.ndarray:
    key = name.lower().replace("-", "").replace("_", "")
    if key not in PALETTES:
        raise KeyError(
            f"Unknown palette '{name}'. Available: {', '.join(sorted(PALETTES))}"
        )
    return PALETTES[key].copy()


def _to_lab(rgb: np.ndarray) -> np.ndarray:
    """(N, 3) uint8 RGB -> (N, 3) float32 Lab."""
    lab = cv2.cvtColor(rgb.reshape(1, -1, 3), cv2.COLOR_RGB2Lab)
    return lab.reshape(-1, 3).astype(np.float32)


def snap_to_palette(art: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Map every pixel of ``art`` to its nearest palette color (CIELAB).

    Alpha, when present, is carried through untouched.
    """
    rgb = np.ascontiguousarray(art[:, :, :3])
    unique, inverse = np.unique(rgb.reshape(-1, 3), axis=0, return_inverse=True)
    dists = (
        (_to_lab(unique)[:, None, :] - _to_lab(palette)[None, :, :]) ** 2
    ).sum(axis=-1)
    mapped = palette[dists.argmin(axis=1)]
    out = mapped[inverse].reshape(rgb.shape)
    if art.shape[2] == 4:
        return np.dstack([out, art[:, :, 3]])
    return out


def reduce_colors(art: np.ndarray, max_colors: int) -> np.ndarray:
    """Reduce ``art`` to at most ``max_colors`` colors (no dithering)."""
    rgb = np.ascontiguousarray(art[:, :, :3])
    unique = np.unique(rgb.reshape(-1, 3), axis=0)
    if len(unique) <= max_colors:
        return art.copy()
    labels, palette = color_ops.quantize_image(rgb, num_colors=max_colors)
    out = palette[labels]
    if art.shape[2] == 4:
        return np.dstack([out, art[:, :, 3]])
    return out


def apply_palette_options(
    art: np.ndarray,
    palette_name: Optional[str] = None,
    max_colors: Optional[int] = None,
) -> np.ndarray:
    """CLI/node-facing helper: optional max-color reduction, then snapping."""
    out = art
    if max_colors:
        out = reduce_colors(out, max_colors)
    if palette_name and palette_name.lower() != "none":
        out = snap_to_palette(out, get_palette(palette_name))
    return out
