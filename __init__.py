"""ComfyUI entry point.

When this repository is cloned into ``ComfyUI/custom_nodes/``, ComfyUI
imports the repo root as a node pack. Everything else (CLI, GUI, pipeline)
imports the inner packages directly and never touches this file.
"""

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from comfyui_pixel_perfecter.nodes import (
        NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS,
    )
except Exception:  # pragma: no cover - outside ComfyUI just stay importable
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
