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

## Current state (2026-07-10, post peakiness gate)

**99.1% (214/216).** Corpus: 78 of the user's own ChatGPT generations +
138 curated SFW images from the Civitai public API (pixel-art LoRA
models; provenance in `civitai_metadata.jsonl`). 22 total exclusions
(17 from a strict agent re-curation, 5 from the user's bench review —
not pixel-art imitations); see `excluded.json` for every one with a
reason.

The over-coarse flat-curve cluster is fixed structurally: `_fit_grid`
now always proposes fine pitches (3-8) and gates the harmonic tie-break
on peakiness (best/median of the gated score curve). Flat evidence takes
the finest candidate within 80% of the best score instead of an
arbitrary argmax; a slight oversample preserves the art. All 19 pass
images whose pick changed under the new selection were re-QC'd on the
report panels: every one is faithful (several visibly better, e.g. the
sign text in `740da09b`). Human review settled the bloom/soft cluster:
those reconstructions are faithful — the bloom belongs to the source —
so they pass (`verdicts.json`).

The 2 remaining failures (`420f84a49bd4bc3a` side profile,
`e94292f0af6ee54a` skull helmet) are **harmonic-double** picks, not
flat-curve cases: their score curves peak *hard* at ~2x the true pitch
(31 scores 2.71 vs 1.96 for the true ~15.7 mesh; 34 scores 4.01 vs 2.46
for true 17) because doubling boosts inter-cell contrast while blocky
art keeps intra-cell variance believable. Measured and rejected
discriminators: sub-cell 2x2 split fraction (textured passes score
higher than these fails) and winner-vs-half round-trip tolerant diff
(backwards on clean giant-block art, where the half explains AA edges
*better*). Fixing them needs a genuinely new measurement — e.g.
sub-pitch edge-phase statistics — without which any threshold flips
legitimate coarse passes.
