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

## Current state (2026-07-10, human-reviewed)

**99.1% (214/216) — the >95% goal is met.** Corpus: 78 of the user's own
ChatGPT generations + 138 curated SFW images from the Civitai public API
(pixel-art LoRA models; provenance in `civitai_metadata.jsonl`). 22 total
exclusions (17 from a strict agent re-curation, 5 from the user's bench
review — not pixel-art imitations); see `excluded.json` for every one with
a reason.

Human review also settled the bloom/soft cluster: those reconstructions
are faithful — the bloom belongs to the source — so they pass
(`verdicts.json`). The 2 remaining failures are mixed-scale-detail cases
(side profile, skull helmet) where the picked pitch is ~2x the finest
structure; the "peakiness gate" research session targets exactly these.
`--sharpen` remains available for bloom-heavy sources.
