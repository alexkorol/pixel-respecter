"""Visual QC of the reconstructor on real AI-generated images.

For each sample: original | original + detected grid | reconstruction
(NN-upscaled back to original size). Rows stack into one QC sheet.

    python benchmarks/qc_real_images.py [output_path]
"""

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pixel_perfecter.reconstructor import PixelArtReconstructor

SAMPLES = [
    "input/20250610_1843_Red Triangle Pixel Art_simple_compose_01jxeafmtye8v8zpqyn290t9nb.png",
    "input/20250610_2037_Pixel Gold Star_simple_compose_01jxegz6xsfk6bc9dng8a92zhs.png",
    "input/20250610_2049_Green Smiley Pixel Art_simple_compose_01jxehphg7fjhrc4gy66858g3k.png",
    "input/20250612_1243_Retro Pixel Character_simple_compose_01jxjtny7ff0svtfqsjft4405d.png",
    "input/20250611_2122_Pixel Art Boots_remix_01jxh5ydenfw0td0pn980yj0cg.png",
]

PANEL = 320


def draw_grid(img: Image.Image, rec: PixelArtReconstructor) -> Image.Image:
    canvas = img.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size
    if rec.mesh_lines:
        lines_x, lines_y = rec.mesh_lines
    else:
        cell = rec.cell_size or 4
        ox, oy = rec.offset or (0, 0)
        lines_x = list(range(ox, w, cell))
        lines_y = list(range(oy, h, cell))
    for x in lines_x:
        draw.line([(x, 0), (x, h)], fill=(255, 0, 80), width=1)
    for y in lines_y:
        draw.line([(0, y), (w, y)], fill=(255, 0, 80), width=1)
    return canvas


def main(out_path: str = "output/qc_new_core.png") -> None:
    rows = []
    for sample in SAMPLES:
        path = Path(sample)
        if not path.exists():
            print(f"skip (missing): {path.name}")
            continue
        with Image.open(path) as im:
            original = im.convert("RGB")

        rec = PixelArtReconstructor(image=np.array(original))
        t0 = time.perf_counter()
        result = rec.run("global")
        elapsed = time.perf_counter() - t0

        kind = (rec.last_metrics or {}).get("grid_kind", "?")
        core = (rec.last_metrics or {}).get("percent_diff_core_tolerant", -1)
        print(
            f"{path.name[:48]:50s} cell={rec.cell_size} kind={kind:5s} "
            f"out={result.shape[1]}x{result.shape[0]} "
            f"tol_core_diff={core:5.1f}% {elapsed:5.2f}s"
        )

        grid_view = draw_grid(original, rec)
        rec_up = Image.fromarray(result).resize(original.size, Image.Resampling.NEAREST)

        row = Image.new("RGB", (PANEL * 3 + 20, PANEL + 26), (24, 24, 28))
        for i, panel in enumerate((original, grid_view, rec_up)):
            thumb = panel.resize((PANEL, PANEL), Image.Resampling.NEAREST)
            row.paste(thumb, (i * (PANEL + 10), 26))
        caption = (
            f"{path.name[:42]}   cell={rec.cell_size} ({kind})   "
            f"1x={result.shape[1]}x{result.shape[0]}   {elapsed:.2f}s"
        )
        ImageDraw.Draw(row).text((4, 6), caption, fill=(230, 230, 230))
        rows.append(row)

    if not rows:
        print("No samples found.")
        return
    sheet = Image.new(
        "RGB",
        (max(r.width for r in rows), sum(r.height for r in rows)),
        (24, 24, 28),
    )
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    print(f"QC sheet saved to {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
