"""
Batch command line interface for the Pixel Perfecter reconstruction pipeline.

This wrapper keeps the core logic in Python so that it plays nicely with the
existing Qt GUI.  The CLI focuses on predictable batch runs and mirrors the
defaults that the legacy ``process_all_images`` helper used, while exposing
more knobs for power users.

Usage examples
--------------

Process every image in ``input/`` and drop the results in ``output/``::

    python -m pixel_perfecter.cli input --output-dir output

Process a single file with ML suggestions and skip overlay exports::

    python -m pixel_perfecter.cli input/blob.png --ml --skip-overlays
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

from PIL import Image

from . import palettes
from .reconstructor import (
    PixelArtReconstructor,
    build_validation_diagnostics,
    _evaluate_with_parameters,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class BatchConfig:
    """Runtime configuration derived from CLI arguments."""

    inputs: Sequence[Path]
    output_dir: Path
    overlay_dir: Optional[Path]
    mode: str
    use_ml: bool
    debug: bool
    use_hough: bool
    initial_upscale: int
    overlay_mode: str
    save_overlays: bool
    grid_debug_dir: Optional[Path]
    metrics_path: Optional[Path]
    palette: Optional[str] = None
    max_colors: Optional[int] = None


def _gather_images(sources: Sequence[Path], recursive: bool) -> List[Path]:
    """Collect candidate image files from the provided locations."""
    seen: set[Path] = set()
    images: List[Path] = []

    for source in sources:
        if source.is_dir():
            iterator: Iterable[Path]
            iterator = source.rglob("*") if recursive else source.iterdir()
            for candidate in iterator:
                if not candidate.is_file():
                    continue
                if candidate.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                resolved = candidate.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                images.append(resolved)
        elif source.is_file():
            if source.suffix.lower() not in IMAGE_EXTENSIONS:
                print(f"[WARN] Skipping unsupported file: {source}")
                continue
            resolved = source.resolve()
            if resolved not in seen:
                seen.add(resolved)
                images.append(resolved)
        else:
            print(f"[WARN] Input path not found: {source}")

    images.sort()
    return images


def _load_ml_suggester(enable: bool) -> Optional[Callable]:
    """Import the ML suggestion helper if requested."""
    if not enable:
        return None
    try:
        from ml.inference import suggest_parameters  # type: ignore

        return suggest_parameters
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARN] ML suggestions unavailable: {exc}")
        return None


def _ensure_dir(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    path.mkdir(parents=True, exist_ok=True)
    return path


def _process_single_image(
    image_path: Path,
    cfg: BatchConfig,
    suggest_fn: Optional[Callable],
) -> Optional[dict]:
    """Run the reconstruction pipeline for one image and persist artefacts."""
    try:
        reconstructor = PixelArtReconstructor(str(image_path), debug=cfg.debug)
        reconstructor.use_hough = cfg.use_hough
        reconstructor.initial_upscale = cfg.initial_upscale
        result = reconstructor.run(mode=cfg.mode)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[ERROR] Failed to process {image_path.name}: {exc}")
        return None

    cell_size = reconstructor.cell_size if getattr(reconstructor, "cell_size", None) else 1
    offset = reconstructor.offset if getattr(reconstructor, "offset", None) else (0, 0)
    diagnostics = build_validation_diagnostics(str(image_path), result, cell_size, offset)
    overlays: dict = diagnostics["overlays"]  # type: ignore[assignment]
    metrics: dict = diagnostics["metrics"]  # type: ignore[assignment]

    selected_overlay = overlays.get(cfg.overlay_mode, overlays.get("combined"))
    best = {
        "result": result,
        "overlay": selected_overlay,
        "overlays": overlays,
        "metrics": metrics,
        "reconstructor": reconstructor,
    }
    ml_applied = False

    if suggest_fn and cfg.mode == "global":
        try:
            suggestions = suggest_fn(str(image_path))
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[WARN] ML suggestions failed for {image_path.name}: {exc}")
            suggestions = []

        for suggestion in suggestions:
            try:
                candidate = _evaluate_with_parameters(
                    image_path=image_path,
                    cell_size=suggestion.cell_size,
                    offset=suggestion.offset,
                    debug=cfg.debug,
                )
            except Exception as exc:  # pylint: disable=broad-except
                print(f"[WARN] ML suggestion error on {image_path.name}: {exc}")
                continue

            candidate_overlay = candidate["overlays"].get(cfg.overlay_mode, candidate["overlay"])
            candidate["overlay"] = candidate_overlay

            if (
                candidate["metrics"]["percent_diff"] + 0.1
                < best["metrics"]["percent_diff"]
            ):
                candidate["metrics"]["warnings"].append(
                    f"Applied ML suggestion (conf={getattr(suggestion, 'confidence', 0.0):.2f})"
                )
                best = candidate
                ml_applied = True

    result = best["result"]
    overlay = best["overlay"]
    metrics = best["metrics"]
    reconstructor = best["reconstructor"]

    # Palette constraints are a stylization of the final asset; diagnostics
    # above keep judging the raw reconstruction against the source.
    result = palettes.apply_palette_options(result, cfg.palette, cfg.max_colors)

    output_path = cfg.output_dir / f"{image_path.stem}_reconstructed{image_path.suffix}"
    Image.fromarray(result).save(output_path)

    overlay_path: Optional[Path] = None
    if cfg.save_overlays and overlay is not None:
        overlay_root = cfg.overlay_dir or cfg.output_dir
        overlay_root.mkdir(parents=True, exist_ok=True)
        overlay_path = overlay_root / f"{image_path.stem}_validation{image_path.suffix}"
        Image.fromarray(overlay).save(overlay_path)

    if cfg.grid_debug_dir:
        reconstructor.debug_grid_detection(save_dir=str(cfg.grid_debug_dir))

    warnings = metrics.get("warnings", []) or []
    if warnings:
        print(f"[WARN] {image_path.name}: {' | '.join(warnings)}")
    else:
        summary = metrics.get(
            "percent_diff_core_tolerant",
            metrics.get("percent_diff", metrics.get("percent_diff_total", 0.0)),
        )
        print(f"[OK] {image_path.name}: tolerant core diff={summary:.2f}%")

    record = {
        "image": image_path.name,
        "cell_size": metrics.get("cell_size", cell_size),
        "offset_x": metrics.get("offset", (0, 0))[0],
        "offset_y": metrics.get("offset", (0, 0))[1],
        "percent_diff": round(metrics.get("percent_diff", 0.0), 4),
        "percent_diff_core": round(metrics.get("percent_diff_core", 0.0), 4),
        "percent_diff_halo": round(metrics.get("percent_diff_halo", 0.0), 4),
        "percent_diff_outside": round(metrics.get("percent_diff_outside", 0.0), 4),
        "grid_ratio": round(metrics.get("grid_ratio", 0.0), 4),
        "warnings": " | ".join(warnings),
        "mode": getattr(reconstructor, "mode", cfg.mode),
        "regions": len(getattr(reconstructor, "region_summaries", []) or [None]),
        "ml_used": "yes" if ml_applied else ("n/a" if cfg.mode == "adaptive" else "no"),
        "output_path": str(output_path),
        "overlay_path": str(overlay_path) if overlay_path else "",
    }

    return record


def _write_metrics_csv(metrics: List[dict], path: Path) -> None:
    fieldnames = [
        "image",
        "cell_size",
        "offset_x",
        "offset_y",
        "percent_diff",
        "percent_diff_core",
        "percent_diff_halo",
        "percent_diff_outside",
        "grid_ratio",
        "warnings",
        "mode",
        "regions",
        "ml_used",
        "output_path",
        "overlay_path",
    ]
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)
    print(f"[INFO] Metrics written to {path}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch pixel art reconstruction CLI.")
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Image files or directories to process.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for reconstructed images (default: ./output).",
    )
    parser.add_argument(
        "--overlay-dir",
        type=Path,
        help="Optional directory for diagnostic overlays (defaults to output dir).",
    )
    parser.add_argument(
        "--mode",
        choices=["global", "adaptive"],
        default="global",
        help="Reconstruction mode. Adaptive refines per-region grids.",
    )
    parser.add_argument(
        "--overlay-mode",
        choices=["combined", "raw", "halo_suppressed", "core_only"],
        default="combined",
        help="Overlay visualization variant to save.",
    )
    parser.add_argument(
        "--skip-overlays",
        action="store_true",
        help="Do not export diagnostic overlay images.",
    )
    parser.add_argument(
        "--ml",
        action="store_true",
        help="Apply ML suggestions when available (global mode only).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging during reconstruction.",
    )
    parser.add_argument(
        "--no-hough",
        action="store_true",
        help="Disable Hough line detection (use autocorrelation only).",
    )
    parser.add_argument(
        "--upscale",
        type=int,
        default=2,
        choices=[1, 2, 3, 4],
        help="Upscale factor for edge detection (1=none, 2=default, higher=better for fine grids).",
    )
    parser.add_argument(
        "--grid-debug",
        action="store_true",
        help="Export grid detection plots to <output>/grid_debug/.",
    )
    parser.add_argument(
        "--grid-debug-dir",
        type=Path,
        help="Custom directory for grid debug plots.",
    )
    parser.add_argument(
        "--palette",
        choices=sorted(palettes.PALETTES) + ["none"],
        default="none",
        help="Snap the reconstructed art to a fixed palette (CIELAB nearest).",
    )
    parser.add_argument(
        "--max-colors",
        type=int,
        help="Reduce the reconstructed art to at most N colors (no dithering).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When inputs include directories, walk them recursively.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        help="Write a CSV summary to the provided path (defaults to <output>/metrics.csv).",
    )
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Do not emit the metrics CSV.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    images = _gather_images(args.inputs, recursive=args.recursive)
    if not images:
        print("[ERROR] No matching images found.")
        return

    output_dir = args.output_dir.resolve()
    _ensure_dir(output_dir)

    overlay_dir: Optional[Path] = None
    if args.overlay_dir:
        overlay_dir = args.overlay_dir.resolve()
        _ensure_dir(overlay_dir)

    grid_debug_dir: Optional[Path]
    if args.grid_debug_dir:
        grid_debug_dir = args.grid_debug_dir.resolve()
        _ensure_dir(grid_debug_dir)
    elif args.grid_debug:
        grid_debug_dir = (output_dir / "grid_debug").resolve()
        _ensure_dir(grid_debug_dir)
    else:
        grid_debug_dir = None

    metrics_path: Optional[Path]
    if args.no_metrics:
        metrics_path = None
    else:
        metrics_path = args.metrics_path.resolve() if args.metrics_path else output_dir / "metrics.csv"

    cfg = BatchConfig(
        inputs=images,
        output_dir=output_dir,
        overlay_dir=overlay_dir,
        mode=args.mode,
        use_ml=args.ml,
        debug=args.debug,
        use_hough=not args.no_hough,
        initial_upscale=args.upscale,
        overlay_mode=args.overlay_mode,
        save_overlays=not args.skip_overlays,
        grid_debug_dir=grid_debug_dir,
        metrics_path=metrics_path,
        palette=None if args.palette == "none" else args.palette,
        max_colors=args.max_colors,
    )

    suggest_fn = _load_ml_suggester(cfg.use_ml)
    metrics_records: List[dict] = []

    print(f"[INFO] Found {len(images)} image(s) to process -> {output_dir}")

    for image_path in images:
        record = _process_single_image(image_path, cfg, suggest_fn)
        if record is not None:
            metrics_records.append(record)

    if metrics_records and cfg.metrics_path:
        _write_metrics_csv(metrics_records, cfg.metrics_path)


if __name__ == "__main__":
    main()
