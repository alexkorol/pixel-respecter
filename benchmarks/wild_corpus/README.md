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
Verdict precedence: `verdicts.json` (human, final say) >
`verdicts_agent.json` (AI visual QC of the report panels) > automatic
heuristics. The tolerant-diff heuristics are triage only — they demonstrably
do not track visual quality, so every suspect needs eyes on it.
Goal: **>95% pass rate** on 200+ images.

## Current state (2026-07-10)

**92.4% (220/238)** — every image visually graded (agent QC, pending human
review). Corpus: 78 of the user's own ChatGPT generations + 160 curated
SFW images from the Civitai public API (pixel-art LoRA models; provenance
in `civitai_metadata.jsonl`).

The 18 failures cluster into:
- **~10 over-coarse picks** on dense fine-pitched art (no flat background).
  Debugging shows the quality score is *flat* across pitches there — no
  harmonic peak — so selection hinges on which candidate sizes happen to be
  proposed. Next research step: bias selection toward finer pitch when the
  score curve shows weak evidence (peakiness gate), validated against the
  full corpus. A slight oversample preserves detail; a coarse miss destroys
  it.
- **6 softly rendered / bloom-heavy sources** whose "pixels" aren't
  grid-aligned at any pitch. `--sharpen` visibly improves them but is not
  auto-gated (no metric separates soft from crisp yet).
- 2 other (garbled palette art, glitch streaks).
