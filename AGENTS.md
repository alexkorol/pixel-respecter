# Pixel Perfecter

## What this project does
ML pipeline that takes real pixel art sprites, generates high-resolution counterparts (photorealistic/concept art), and packages everything for LoRA model training. The model learns to **pixelize** (high-res -> pixel art), not depixelize.

## Project structure
- `pixel_perfecter/` - Core reconstructor (grid detection, pixel-perfect reconstruction)
- `pipeline/` - Meta pipeline: extract -> clean -> tag -> caption -> prompts -> generate -> package -> train
- `training/` - LoRA fine-tuning infrastructure
- `scripts/` - Utility scripts (batch captioning, etc.)
- `tests/` - pytest test suite
- `data/` - Pipeline output (gitignored, large)
- `input/artist-ref/` - Copyright artist reference images (gitignored)

## Setup
```bash
pip install -r requirements.txt
pip install python-dotenv openai  # for OpenRouter backend
```

API keys go in `.env` at project root:
```
OPENROUTER_API_KEY=sk-or-...
```

## Running tests
```bash
python -m pytest tests/ -x -q
```

## Key commands
```bash
# Run full pipeline
python -m pipeline.cli run --help

# DCSS tile curation
python -m pipeline.crawl_descriptions --fetch -o data/dcss-curated --categories item mon dngn

# Batch caption sprites
python scripts/batch_caption_remaining.py
```

## Important conventions
- All file paths in manifests/JSON use forward slashes (cross-platform)
- The OpenRouter generator uses raw `urllib` (NOT the OpenAI SDK) because the SDK strips the `images` field from responses
- Generator model: `google/gemini-2.5-flash-image` via OpenRouter
- NEVER run batch API calls on uncurated data -- always visually QC first
- Training direction: pixel art = TARGET (ground truth), generated high-res = INPUT
