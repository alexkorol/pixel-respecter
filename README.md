# Pixel Perfecter

Pixel Perfecter turns degraded "pixel art" — AI-generated renders, scaled screenshots, anti-aliased exports — into clean, grid-aligned sprites at their true 1× resolution. Around that core sits an ML data pipeline that pairs cleaned sprites with generated high-resolution counterparts and packages them for LoRA training (the model learns to **pixelize**: high-res in, pixel art out).

---

## Components at a Glance

- **Core reconstructor (`pixel_perfecter/`)** – Detects the pixel grid (even non-integer upscales), collapses each cell to its dominant color, and emits halo-aware diagnostics.
- **Interactive GUI (`python -m pixel_perfecter.gui`)** – PySide6 app with drag-and-drop import, live previews, and structured feedback logging.
- **Batch CLI (`python -m pixel_perfecter.cli`)** – Processes files or directories, writes reconstructions plus overlays, and records metrics to CSV.
- **ComfyUI node (`comfyui_pixel_perfecter/`)** – Drop the repo into `ComfyUI/custom_nodes/` and post-process generated images into true 1× pixel art inside any workflow. See its [README](comfyui_pixel_perfecter/README.md).
- **Wild benchmark (`benchmarks/wild_corpus/`)** – Real AI-generated images + one-command eval (`python benchmarks/eval_wild.py`) with a visual verdict gallery.
- **Meta pipeline (`pipeline/`)** – extract → clean → tag → caption → prompts → generate → package → train. Includes DCSS tile datamining and a curation gallery.
- **LoRA training (`training/`)** – Fine-tuning infrastructure for the pixelize model.

---

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
# or: source .venv/bin/activate  # macOS / Linux
pip install -r requirements.txt
pip install python-dotenv openai  # for the OpenRouter pipeline backends
```

API keys for the pipeline go in `.env` at the project root (`OPENROUTER_API_KEY=sk-or-...`). The `input/` and `output/` folders are git-ignored working space.

---

## How Reconstruction Works

1. **Candidate detection** — four independent detectors propose cell sizes:
   - *Exact NN check*: largest scale where every block is perfectly uniform
   - *Hough mesh*: Canny → morphological closing → probabilistic Hough → line clustering → trimmed-median pitch (the proper-pixel-art chain)
   - *Autocorrelation* of boundary-mass projections
   - *Phase-fold sweep*: cheap alignment score for every plausible size
2. **Unified scoring** — every candidate `(cell size, offset)` is ranked by one metric: inter-cell contrast / √(intra-cell variance). It peaks at the true grid and resolves harmonics (5 vs 10 px) toward the native resolution. No detector is trusted blindly; the best explanation of the image wins.
3. **Mesh mode** — when the pitch is fractional (e.g. a 32-px sprite upscaled to fill 496 px gives alternating 15/16-px cells), no rigid grid fits. The pitch is measured to sub-pixel precision from autocorrelation peaks, grid lines snap to boundary-mass peaks, and reconstruction uses the variable mesh directly.
4. **Downsampling** — the image is mapped to palette-label space (exact colors when few, quantized otherwise — never dithered), then each cell takes its modal color via a single vectorized bincount. Cell interiors are margin-trimmed so anti-aliased borders don't vote.
5. **Diagnostics** — reconstructions are compared against the source with halo/core separation, tolerant diffs for AI content, and per-image warnings.

Everything is vectorized; typical sprites reconstruct in well under a second (the previous implementation took 8–28 s and mis-detected harmonics and fractional pitches).

---

## Batch CLI

```bash
# Basic run on every PNG / JPEG / BMP / WebP in ./input
python -m pixel_perfecter.cli input --output-dir output

# Walk directories recursively and split overlays
python -m pixel_perfecter.cli input extra/*.png \
    --mode global \
    --recursive \
    --overlay-dir output/overlays
```

### Key options

| Flag | Description |
| ---- | ----------- |
| `inputs...` | Files or directories to process. Directories are scanned (optionally recursively) for supported image types. |
| `-o, --output-dir PATH` | Destination for reconstructed sprites (default: `./output`). |
| `--overlay-dir PATH` | Optional directory for validation overlays (defaults to the output dir). |
| `--mode {global,adaptive}` | Grid strategy. Adaptive fits per-region grids when a single lattice fails. |
| `--palette {pico8,db32,gameboy,none}` | Snap the reconstructed art to a fixed palette (CIELAB nearest-color). NES is omitted deliberately: the 2C02 has no canonical RGB palette. |
| `--max-colors N` | Reduce the reconstruction to at most N colors (no dithering). |
| `--transparent-bg` | Make the dominant border color transparent (output becomes PNG). |
| `--overlay-mode {combined,raw,halo_suppressed,core_only}` | Diagnostic composite to export. |
| `--skip-overlays` | Do not write overlay images. |
| `--ml` | Enable ML suggestions if `ml/` checkpoints are present (ignored in adaptive mode). |
| `--debug` | Verbose logging from the reconstruction pipeline. |
| `--grid-debug` / `--grid-debug-dir PATH` | Export grid detection plots (defaults to `output/grid_debug/`). |
| `--recursive` | Walk directory inputs recursively. |
| `--metrics-path PATH` / `--no-metrics` | Control where the metrics CSV lands (defaults to `output/metrics.csv`). |

Each run produces `<name>_reconstructed.png` (the 1× sprite), `<name>_validation.png` (halo vs core diff overlay), and `metrics.csv`.

---

## GUI

```bash
python -m pixel_perfecter.gui
```

Drag-and-drop loading, live recomputation when cell size or offsets change, multiple overlay styles, optional adaptive per-region mode, and feedback logging to `notes/feedback_log.csv`.

---

## ML Pipeline

```bash
# Full pipeline run (see --help for stage selection)
python -m pipeline.cli run --help

# DCSS tile curation (datamined game descriptions + curation gallery)
python -m pipeline.crawl_descriptions --fetch -o data/dcss-curated --categories item mon dngn

# Batch caption sprites via OpenRouter
python scripts/batch_caption_remaining.py
```

Conventions that matter:

- Pixel art is the **target** (ground truth); generated high-res images are the **input**. The trained model pixelizes.
- Manifest/JSON paths use forward slashes (cross-platform).
- The OpenRouter generator speaks raw `urllib` because the OpenAI SDK strips the `images` field from responses.
- Never run batch API calls on uncurated data — visually QC first.

---

## Running Tests & Benchmarks

```bash
python -m pytest tests/ -x -q
python benchmarks/bench_reconstructor.py
```

The test suite includes ground-truth round-trips: clean integer upscales must reconstruct **pixel-exact**, fractional upscales (15.5 px, 9.7 px pitches) must trigger mesh mode and still match exactly, and noisy/blurred AI-style art must land within tolerance.

---

## Repository Layout

```
pixel-perfecter/
├─ pixel_perfecter/        # Core package
│  ├─ reconstructor.py     #   detection orchestrator + diagnostics (public API)
│  ├─ grid.py              #   vectorized rigid-grid engine (labels, scoring, modal downsample)
│  ├─ mesh.py              #   variable-mesh detection for non-integer upscales
│  ├─ colors.py            #   quantization, alpha handling, background transparency
│  ├─ cli.py / gui*.py     #   batch CLI and PySide6 GUI
├─ pipeline/               # Meta pipeline: extract → clean → … → train
├─ training/               # LoRA fine-tuning infrastructure
├─ scripts/                # Utility scripts (batch captioning, etc.)
├─ tests/                  # pytest suite (93 tests)
├─ benchmarks/             # Reconstruction speed/accuracy benchmarks
├─ ml/                     # Optional grid-parameter suggestion models
├─ input/ · output/ · data/  # Working dirs (git-ignored)
└─ notes/ · archive/       # Design docs and earlier experiments
```

---

## Acknowledgements

The reconstruction core owes a great deal to **Kenneth J. Allen**'s
[proper-pixel-art](https://github.com/KennethJAllen/proper-pixel-art) (MIT):
the Hough-based mesh detection chain (edge closing → line clustering →
trimmed-median pitch → mesh completion), the variable-mesh representation
that handles non-integer upscales, quantize-before-downsample color
handling, transparency-majority alpha cells, and offset-binned dominant
colors all follow his design. His write-ups at
[kennethallenmath.com](https://www.kennethallenmath.com/) are the clearest
treatment of this problem we've found — go read them.

Additional inspiration from [unfake.js](https://github.com/jenissimo/unfake.js)
(rich controls and UX for the same problem space).

---

## License

MIT License – see `LICENSE` for full text.
