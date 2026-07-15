# Pixel Perfecter — ComfyUI Node

Turns AI-generated "pixel art" (images that merely *look* like pixel art)
into actual grid-true pixel art. Detects the pixel grid — including
non-integer upscales — and reconstructs at the native 1× resolution.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/alexkorol/pixel-perfecter
pip install opencv-python-headless   # the only dependency ComfyUI lacks
```

Restart ComfyUI. The node appears as **Pixel Perfecter (true 1× pixel art)**
under `image/pixel art`.

## Node

**Inputs**
- `image` — any IMAGE (e.g. straight from a KSampler/VAE Decode)
- `mode` — `global` (one grid for the whole image, default) or `adaptive`
  (per-region grids when a single lattice fails)

**Outputs**
- `pixel_art_1x` — the reconstruction at native resolution (one output pixel
  per detected art pixel). In a batch, smaller results are zero-padded to
  the largest.
- `pixel_art_full_size` — the same art NN-upscaled back to the input canvas
  size, for chaining into nodes that expect the original dimensions.
- `cell_size` — detected pitch in source pixels (INT).

## Typical workflow

`Checkpoint -> KSampler ("pixel art sprite of ...") -> VAE Decode ->`
**Pixel Perfecter** `-> Save Image`

The 1× output is the asset; the full-size output is for previewing at a
comfortable zoom.
