"""ComfyUI node wrapping the pixel-perfecter reconstructor.

Turns AI-generated "pixel art" (images that merely look like pixel art)
into actual grid-true pixel art at its native 1x resolution.

The heavy lifting is pure numpy (`reconstruct_array`), testable without
ComfyUI; torch is only touched inside the node's execute method, where
ComfyUI guarantees it exists.
"""

import sys
from pathlib import Path
from typing import Tuple

import numpy as np

# When this repo is cloned into ComfyUI/custom_nodes/, the pixel_perfecter
# package lives next to this directory rather than on sys.path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pixel_perfecter import palettes  # noqa: E402
from pixel_perfecter.reconstructor import (  # noqa: E402
    PixelArtReconstructor,
    unsharp_prepass,
)


def reconstruct_array(
    img: np.ndarray,
    mode: str = "global",
    palette: str = "none",
    max_colors: int = 0,
    sharpen: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Reconstruct one uint8 RGB image.

    Returns (native_1x, nn_upscaled_back_to_input_size, detected_cell_size).
    """
    source = np.ascontiguousarray(img[:, :, :3])
    if sharpen:
        source = unsharp_prepass(source)
    rec = PixelArtReconstructor(image=source)
    native = rec.run(mode)
    native = palettes.apply_palette_options(
        native, None if palette == "none" else palette, max_colors or None
    )
    h, w = img.shape[:2]
    # NN-upscale the 1x art back to the source canvas for in-place chaining.
    ys = np.minimum((np.arange(h) * native.shape[0]) // h, native.shape[0] - 1)
    xs = np.minimum((np.arange(w) * native.shape[1]) // w, native.shape[1] - 1)
    upscaled = native[np.ix_(ys, xs)]
    return native, upscaled, int(rec.cell_size or 1)


def _pad_to(img: np.ndarray, height: int, width: int) -> np.ndarray:
    """Bottom/right zero-pad so differing native sizes can share a batch."""
    if img.shape[0] == height and img.shape[1] == width:
        return img
    out = np.zeros((height, width, img.shape[2]), dtype=img.dtype)
    out[: img.shape[0], : img.shape[1]] = img
    return out


class PixelPerfecter:
    """Reconstruct true pixel art from an AI-generated approximation."""

    CATEGORY = "image/pixel art"
    FUNCTION = "reconstruct"
    RETURN_TYPES = ("IMAGE", "IMAGE", "INT")
    RETURN_NAMES = ("pixel_art_1x", "pixel_art_full_size", "cell_size")
    DESCRIPTION = (
        "Detects the pixel grid of AI-generated pixel art (including "
        "non-integer upscales) and reconstructs it grid-true at native 1x "
        "resolution. Also returns the result NN-upscaled back to the input "
        "size, plus the detected cell pitch in source pixels."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["global", "adaptive"], {"default": "global"}),
            },
            "optional": {
                "palette": (
                    ["none"] + sorted(palettes.PALETTES),
                    {"default": "none"},
                ),
                "max_colors": (
                    "INT",
                    {"default": 0, "min": 0, "max": 256,
                     "tooltip": "0 = unlimited; otherwise reduce to N colors"},
                ),
                "sharpen": (
                    "BOOLEAN",
                    {"default": False,
                     "tooltip": "Unsharp-mask soft/bloom-heavy sources first"},
                ),
            },
        }

    def reconstruct(
        self,
        image,
        mode: str = "global",
        palette: str = "none",
        max_colors: int = 0,
        sharpen: bool = False,
    ):
        import torch

        batch = image.detach().cpu().numpy()
        natives, upscaled, cells = [], [], []
        for item in batch:
            img = np.clip(item * 255.0, 0, 255).astype(np.uint8)
            native, up, cell = reconstruct_array(
                img, mode, palette, max_colors, sharpen)
            natives.append(native)
            upscaled.append(up)
            cells.append(cell)

        max_h = max(n.shape[0] for n in natives)
        max_w = max(n.shape[1] for n in natives)
        natives = [_pad_to(n, max_h, max_w) for n in natives]

        def to_tensor(arrays):
            stack = np.stack(arrays).astype(np.float32) / 255.0
            return torch.from_numpy(stack)

        return (to_tensor(natives), to_tensor(upscaled), cells[0])


NODE_CLASS_MAPPINGS = {"PixelPerfecter": PixelPerfecter}
NODE_DISPLAY_NAME_MAPPINGS = {"PixelPerfecter": "Pixel Perfecter (true 1x pixel art)"}
