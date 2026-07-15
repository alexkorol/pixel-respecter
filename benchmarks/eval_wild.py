"""Zero-config eval of the reconstructor over the wild benchmark corpus.

One command in, one report out:

    python benchmarks/eval_wild.py [--out output/wild_eval] [--limit N]

For every corpus image the reconstructor runs with zero manual configuration.
Results are sorted into pass / suspect / fail by automatic heuristics; human
verdicts in benchmarks/wild_corpus/verdicts.json override them. Outputs:

- report.csv       per-image metrics
- panels/<id>.png  original | detected grid | reconstruction side-by-side
- recon/<id>.png   the raw 1x reconstruction
- report.html      visual gallery with verdict buttons + verdict export
"""

import argparse
import csv
import json
import sys
import time
from html import escape
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pixel_perfecter.reconstructor import PixelArtReconstructor

CORPUS_DIR = Path(__file__).resolve().parent / "wild_corpus"
MANIFEST_PATH = CORPUS_DIR / "manifest.json"
VERDICTS_PATH = CORPUS_DIR / "verdicts.json"          # human, final say
AGENT_VERDICTS_PATH = CORPUS_DIR / "verdicts_agent.json"  # AI visual QC
GROUND_TRUTH_DIR = CORPUS_DIR / "ground_truth"

PANEL = 300
# Auto-verdict thresholds (heuristics, human verdicts override):
MIN_CELL = 3          # detected pitch below this = grid not really found
SUSPECT_CORE_DIFF = 12.0   # % tolerant core diff above this = suspect
FAIL_CORE_DIFF = 30.0      # ... above this = fail


def exact_cell_accuracy(result: np.ndarray, ground_truth: np.ndarray) -> float:
    """Exact RGB accuracy over the union of two 1x cell grids.

    Missing or extra rows/columns count as incorrect cells. There is no
    tolerance: this is calibration truth, not a perceptual quality score.
    """
    def _canonical_rgba(image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)
        if image.shape[2] >= 4:
            rgba = image[:, :, :4].copy()
        else:
            alpha = np.full(image.shape[:2] + (1,), 255, dtype=np.uint8)
            rgba = np.concatenate([image[:, :, :3], alpha], axis=2)
        rgba[rgba[:, :, 3] == 0, :3] = 0
        return rgba

    result = _canonical_rgba(result)
    ground_truth = _canonical_rgba(ground_truth)
    height = max(result.shape[0], ground_truth.shape[0])
    width = max(result.shape[1], ground_truth.shape[1])
    if height == 0 or width == 0:
        return 0.0
    overlap_h = min(result.shape[0], ground_truth.shape[0])
    overlap_w = min(result.shape[1], ground_truth.shape[1])
    exact = np.all(
        result[:overlap_h, :overlap_w] == ground_truth[:overlap_h, :overlap_w],
        axis=-1,
    )
    return 100.0 * float(exact.sum()) / float(height * width)


def draw_grid(img: Image.Image, rec: PixelArtReconstructor) -> Image.Image:
    canvas = img.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size
    if rec.mesh_lines:
        lines_x, lines_y = rec.mesh_lines
    else:
        cell = rec.cell_size or 4
        ox, oy = rec.offset or (0, 0)
        lines_x = range(int(ox), w, int(cell))
        lines_y = range(int(oy), h, int(cell))
    for x in lines_x:
        draw.line([(x, 0), (x, h)], fill=(255, 0, 80), width=1)
    for y in lines_y:
        draw.line([(0, y), (w, y)], fill=(255, 0, 80), width=1)
    return canvas


def evaluate_entry(entry: dict, out_dir: Path) -> dict:
    path = CORPUS_DIR / entry["file"]
    row = {
        "id": entry["id"],
        "name": entry["original_name"],
        "source": entry.get("source", "?"),
        "input_size": f"{entry['width']}x{entry['height']}",
    }
    try:
        with Image.open(path) as im:
            original = im.convert("RGB")
        rec = PixelArtReconstructor(image=np.array(original))
        t0 = time.perf_counter()
        result = rec.run("global")
        elapsed = time.perf_counter() - t0
    except Exception as exc:  # pylint: disable=broad-except
        row.update(
            auto_verdict="fail",
            reason=f"exception: {exc}",
            cell_size="", grid_kind="", output_size="",
            core_diff_tol="", warnings="", seconds="",
            ground_truth_accuracy="", ground_truth_shape_match="",
        )
        return row

    metrics = rec.last_metrics or {}
    core_diff = float(metrics.get("percent_diff_core_tolerant", -1))
    grid_kind = metrics.get("grid_kind", "?")
    warnings = list(metrics.get("warnings", []))
    no_grid = bool(metrics.get("no_grid"))
    cell = rec.cell_size or 0

    if no_grid:
        verdict, reason = "fail", "no reliable grid detected"
    elif cell < MIN_CELL and grid_kind != "mesh":
        verdict, reason = "fail", f"implausible pitch ({cell}px)"
    elif core_diff > FAIL_CORE_DIFF:
        verdict, reason = "fail", f"core diff {core_diff:.1f}%"
    elif core_diff > SUSPECT_CORE_DIFF:
        verdict, reason = "suspect", f"core diff {core_diff:.1f}%"
    elif warnings:
        verdict, reason = "suspect", "; ".join(warnings)[:120]
    else:
        verdict, reason = "pass", ""

    row.update(
        auto_verdict=verdict,
        reason=reason,
        cell_size=cell,
        grid_kind=grid_kind,
        output_size=f"{result.shape[1]}x{result.shape[0]}",
        core_diff_tol=round(core_diff, 2),
        warnings="; ".join(warnings),
        seconds=round(elapsed, 2),
        ground_truth_accuracy="",
        ground_truth_shape_match="",
    )

    truth_path = GROUND_TRUTH_DIR / f"{entry['id']}.png"
    if truth_path.exists():
        with Image.open(truth_path) as truth_image:
            truth = np.asarray(truth_image.convert("RGBA"))
        row["ground_truth_accuracy"] = round(
            exact_cell_accuracy(result, truth), 4
        )
        row["ground_truth_shape_match"] = result.shape[:2] == truth.shape[:2]

    # Panel: original | grid overlay | reconstruction (NN back to source size)
    recon_img = Image.fromarray(result)
    recon_img.save(out_dir / "recon" / f"{entry['id']}.png")
    grid_view = draw_grid(original, rec)
    rec_up = recon_img.resize(original.size, Image.Resampling.NEAREST)
    sheet = Image.new("RGB", (PANEL * 3 + 20, PANEL), (24, 24, 28))
    for i, panel in enumerate((original, grid_view, rec_up)):
        thumb = panel.copy()
        thumb.thumbnail((PANEL, PANEL), Image.Resampling.NEAREST)
        sheet.paste(thumb, (i * (PANEL + 10), (PANEL - thumb.height) // 2))
    sheet.save(out_dir / "panels" / f"{entry['id']}.jpg", quality=88)
    return row


def render_html(rows: list, human: dict, out_dir: Path) -> None:
    cards = []
    for r in rows:
        rid = r["id"]
        hv = human.get(rid, "")
        effective = hv or r["auto_verdict"]
        cards.append(f"""
<div class="card v-{escape(effective)}" data-id="{escape(rid)}" data-auto="{escape(r['auto_verdict'])}">
  <img src="panels/{escape(rid)}.jpg" loading="lazy">
  <div class="meta">
    <b>{escape(str(r['name'])[:60])}</b><br>
    cell={escape(str(r['cell_size']))} ({escape(str(r['grid_kind']))})
    &nbsp; 1x={escape(str(r['output_size']))}
    &nbsp; core diff={escape(str(r['core_diff_tol']))}%
    {('&nbsp; exact ground truth=' + escape(str(r.get('ground_truth_accuracy', ''))) + '%') if r.get('ground_truth_accuracy', '') != '' else ''}
    &nbsp; {escape(str(r['seconds']))}s<br>
    auto: <span class="badge b-{escape(r['auto_verdict'])}">{escape(r['auto_verdict'])}</span>
    {('&mdash; ' + escape(str(r['reason']))) if r['reason'] else ''}
  </div>
  <div class="btns">
    <button onclick="setV('{escape(rid)}','pass')">pass</button>
    <button onclick="setV('{escape(rid)}','fail')">fail</button>
    <button onclick="setV('{escape(rid)}','')">clear</button>
    <span class="hv"></span>
  </div>
</div>""")

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Wild corpus eval</title><style>
body {{ background:#141418; color:#ddd; font-family:system-ui,sans-serif; margin:16px; }}
.summary {{ font-size:18px; margin-bottom:16px; }}
.card {{ background:#1e1e24; border-left:6px solid #555; border-radius:6px;
        margin-bottom:14px; padding:10px; }}
.card.v-pass {{ border-left-color:#3fb950; }}
.card.v-suspect {{ border-left-color:#d29922; }}
.card.v-fail {{ border-left-color:#f85149; }}
.card img {{ max-width:100%; display:block; margin-bottom:6px; }}
.badge {{ padding:1px 8px; border-radius:8px; font-size:12px; }}
.b-pass {{ background:#1a4d2a; }} .b-suspect {{ background:#5c4a12; }} .b-fail {{ background:#5c1a1a; }}
.btns {{ margin-top:6px; }} button {{ margin-right:6px; }}
.hv {{ color:#8bf; margin-left:10px; }}
#bar {{ position:sticky; top:0; background:#141418; padding:8px 0; z-index:9; }}
</style></head><body>
<div id="bar">
  <span class="summary" id="summary"></span>
  <button onclick="exportV()">Export verdicts.json</button>
  <label><input type="checkbox" id="onlyBad" onchange="filter()"> show only suspect/fail</label>
</div>
{''.join(cards)}
<script>
const human = {json.dumps(human)};
function refresh() {{
  let counts = {{pass:0, suspect:0, fail:0}};
  document.querySelectorAll('.card').forEach(c => {{
    const id = c.dataset.id, auto = c.dataset.auto;
    const v = human[id] || auto;
    c.className = 'card v-' + v;
    c.querySelector('.hv').textContent = human[id] ? ('human: ' + human[id]) : '';
    counts[v] = (counts[v]||0) + 1;
  }});
  const total = counts.pass + counts.suspect + counts.fail;
  const rate = total ? (100 * counts.pass / total).toFixed(1) : 0;
  document.getElementById('summary').textContent =
    `pass ${{counts.pass}} / suspect ${{counts.suspect}} / fail ${{counts.fail}}` +
    ` of ${{total}} — pass rate ${{rate}}%  `;
}}
function setV(id, v) {{ if (v) human[id] = v; else delete human[id]; refresh(); filter(); }}
function filter() {{
  const only = document.getElementById('onlyBad').checked;
  document.querySelectorAll('.card').forEach(c => {{
    const v = human[c.dataset.id] || c.dataset.auto;
    c.style.display = (only && v === 'pass') ? 'none' : '';
  }});
}}
function exportV() {{
  const blob = new Blob([JSON.stringify(human, null, 2) + '\\n'],
                        {{type: 'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'verdicts.json'; a.click();
}}
refresh();
</script></body></html>"""
    (out_dir / "report.html").write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("output/wild_eval"))
    parser.add_argument("--limit", type=int, default=0, help="Only first N images")
    parser.add_argument("--only", help="Comma-separated corpus ids to re-run")
    args = parser.parse_args()

    if not MANIFEST_PATH.exists():
        sys.exit("No manifest. Run benchmarks/ingest_wild.py first.")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entries = manifest["entries"]
    if args.only:
        wanted = set(args.only.split(","))
        entries = [e for e in entries if e["id"] in wanted]
    if args.limit:
        entries = entries[: args.limit]

    # Verdict precedence: human > agent visual QC > auto heuristics.
    human = {}
    if AGENT_VERDICTS_PATH.exists():
        human.update({
            k: v
            for k, v in json.loads(
                AGENT_VERDICTS_PATH.read_text(encoding="utf-8")
            ).items()
            if not k.startswith("_")
        })
    if VERDICTS_PATH.exists():
        human.update({
            k: v
            for k, v in json.loads(
                VERDICTS_PATH.read_text(encoding="utf-8")
            ).items()
            if not k.startswith("_")
        })

    out_dir = args.out
    (out_dir / "panels").mkdir(parents=True, exist_ok=True)
    (out_dir / "recon").mkdir(parents=True, exist_ok=True)

    rows = []
    for i, entry in enumerate(entries, 1):
        row = evaluate_entry(entry, out_dir)
        rows.append(row)
        effective = human.get(row["id"], row["auto_verdict"])
        print(
            f"[{i:3d}/{len(entries)}] {effective:7s} cell={str(row['cell_size']):>4s} "
            f"{str(row['grid_kind']):5s} diff={str(row['core_diff_tol']):>6s}% "
            f"gt={str(row.get('ground_truth_accuracy', '')):>8s}% "
            f"{row['name'][:52]}"
        )

    fields = ["id", "name", "source", "input_size", "auto_verdict", "reason",
              "cell_size", "grid_kind", "output_size", "core_diff_tol",
              "warnings", "seconds", "ground_truth_accuracy",
              "ground_truth_shape_match"]
    report_path = out_dir / "report.csv"
    if args.only and report_path.exists():
        # Partial re-run: merge into the existing report, don't clobber it.
        fresh = {r["id"]: r for r in rows}
        merged = []
        with open(report_path, newline="", encoding="utf-8") as fh:
            for old in csv.DictReader(fh):
                merged.append(fresh.pop(old["id"], old))
        rows = merged + list(fresh.values())
    with open(report_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    render_html(rows, human, out_dir)

    counts = {"pass": 0, "suspect": 0, "fail": 0}
    for r in rows:
        counts[human.get(r["id"], r["auto_verdict"])] = (
            counts.get(human.get(r["id"], r["auto_verdict"]), 0) + 1
        )
    total = sum(counts.values())
    rate = 100.0 * counts["pass"] / total if total else 0.0
    print(
        f"\npass {counts['pass']} / suspect {counts['suspect']} / "
        f"fail {counts['fail']} of {total}  ->  pass rate {rate:.1f}%"
    )
    print(f"Report: {out_dir / 'report.html'}")


if __name__ == "__main__":
    main()
