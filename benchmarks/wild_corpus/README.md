# Wild-Input Benchmark Corpus

Real AI-generated "pixel art" images (Midjourney / ChatGPT / SD / Gemini output)
used to measure how reliably the reconstructor handles inputs it was built for:
gradient-noised, anti-aliased, inconsistent-pitch images that merely *look*
like pixel art. Synthetic integer upscales live in `tests/`; this corpus is
the messy real world.

## Layout

- `manifest.json` — committed. One entry per corpus image: content hash,
  original filename, dimensions, source tag.
- `verdicts.json` — committed. Human pass/fail judgments keyed by content
  hash. These override the automatic heuristics in the eval report.
- `images/` — **git-ignored** (large binaries). Re-populate on any machine
  with `python benchmarks/ingest_wild.py <source-dir>`.

## Workflow

```bash
# 1. Add images to the corpus (dedupes by content hash)
python benchmarks/ingest_wild.py input/ --source chatgpt-image

# 2. Run the zero-config eval (one command, writes output/wild_eval/)
python benchmarks/eval_wild.py

# 3. Open output/wild_eval/report.html, hand-mark verdicts,
#    click "Export verdicts", save over benchmarks/wild_corpus/verdicts.json

# 4. Re-run eval_wild.py — success rate now uses your verdicts
```

## Success criteria

An image **passes** when the reconstructor, with zero manual configuration,
finds a plausible native-resolution grid and the reconstruction is faithful.
Automatic heuristics (grid found, scale ≥ 3×, tolerant core diff) sort results
into pass / suspect / fail; human verdicts in `verdicts.json` are final.
Goal: **>95% pass rate** on 200+ images.
