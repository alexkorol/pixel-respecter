# Fix Lab

Fix Lab turns hand corrections into machine-decision calibration data. Start
it from the repository root:

```bash
python -m fixlab.server
```

Open <http://127.0.0.1:8000/fixlab/>. Select a wild-corpus entry or drop an
image, choose one of the reconstructor's own pitch/mesh candidates, correct
ambiguous cells, nudge mesh lines, merge palette entries, and mark the sprite
complete. The server writes decisions to
`benchmarks/wild_corpus/fixes/<id>.jsonl` and the corrected sprite to
`benchmarks/wild_corpus/ground_truth/<id>.png`.

## Saving and bulk sessions

- Every calibration decision is appended to the sprite's JSONL immediately.
- The corrected PNG is written only when **Save corrected sprite** or
  **Save & next** is pressed.
- **Export this sprite** and **Export session backup** are optional backups
  when `fixlab.server` is running; they are required when using a static-only
  server that cannot write the repository.
- The Finish card always shows the exact JSONL and PNG destination paths.
- **Save & next** advances the corpus selector for bulk review.

Cell selection works from either the source or enlarged corrected image. The
yellow boxes stay synchronized, and cell-vote decisions capture the 5x5
corrected neighborhood plus a padded source-window reference. Automatic
transparency is opt-in and removes only pixels of the adaptive dominant border
color cluster that are connected to an image edge, preserving disconnected
interior reuse. Step 4 also offers compact palette swatches plus color and hex
inputs for manually choosing the exact border-connected transparency color.
The default-on detached-speckle cleanup also removes small isolated components
that resemble the chosen background while preserving the main sprite and
interior color reuse; turn it off for intentional floating particles/effects.
With transparency enabled, crop-on-save is enabled by default
with a guaranteed one-pixel transparent buffer on all four sides. The export
canvas grows beyond the generated grid when content touches an edge. The
buffer is configurable. The editor and calibration coordinates
stay on the full generated grid; the native preview and saved PNG use the
crop. Fix logs retain the full grid size, tight opaque bounds, padded export
bounds, and final content size for later prompt/model analysis.

The UI also works with `python -m http.server`, but cannot write the repo in
that mode and downloads the data products instead.

`python scripts/fixes_to_tests.py` generates regression cases only for
completed calibrations that have both a decision log and a saved ground-truth
PNG. In-progress JSONL sessions remain available for later work but do not
make the generated test suite fail.
