"""Command-line interface for the video-to-service-data pipeline.

Subcommands
-----------
    extract     sample unique frames from a video
    ocr         run vision OCR on frames, emitting JSON-lines rows
    dedupe      merge overlapping rows and write services.csv
    analyze     compute stats + plots from services.csv / services.jsonl
    run         do all of the above end-to-end

Examples
--------
    # Full pipeline
    python -m service_analytics.cli run path/to/video.mp4 \\
        --out data/service-analytics/run1 \\
        --model google/gemini-2.5-pro

    # Just re-run analytics on an existing deduped file
    python -m service_analytics.cli analyze data/service-analytics/run1/services.jsonl \\
        --out data/service-analytics/run1/reports
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

from service_analytics.schema import ServiceRow, dump_jsonl, load_jsonl

logger = logging.getLogger("service_analytics")


def _load_dotenv_if_present() -> None:
    """Best-effort .env loading so OPENROUTER_API_KEY is available.

    Matches what the rest of the repo does (see CLAUDE.md).
    """
    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:
        return
    # look in cwd then repo root
    for candidate in (Path.cwd() / ".env", Path(__file__).resolve().parents[1] / ".env"):
        if candidate.exists():
            load_dotenv(candidate)
            return


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_extract(args: argparse.Namespace) -> int:
    from service_analytics.extract_frames import extract_frames

    frames = extract_frames(
        Path(args.video),
        Path(args.out),
        mode=args.mode,
        scene_threshold=args.scene_threshold,
        min_gap_s=args.min_gap,
        max_frames=args.max_frames,
        fps_fallback=args.fps_fallback,
    )
    print(f"Extracted {len(frames)} unique frames to {args.out}")
    return 0


def cmd_ocr(args: argparse.Namespace) -> int:
    from service_analytics.extract_frames import load_manifest
    from service_analytics.ocr import ocr_frames

    _load_dotenv_if_present()
    frames = load_manifest(Path(args.frames_dir))
    results = ocr_frames(
        frames,
        Path(args.out),
        model=args.model,
        rate_limit_s=args.rate_limit,
        max_retries=args.max_retries,
        save_raw=args.save_raw,
        resume=args.resume,
    )
    total_rows = sum(len(r.rows) for r in results)
    errors = sum(1 for r in results if r.error)
    print(
        f"OCR complete: {len(results)} frames processed, "
        f"{total_rows} raw rows extracted, {errors} errors"
    )
    return 0 if errors == 0 else 1


def cmd_dedupe(args: argparse.Namespace) -> int:
    from service_analytics.dedupe import dedupe_rows

    rows = load_jsonl(Path(args.input))
    merged = dedupe_rows(rows)
    dump_jsonl(merged, Path(args.out))
    print(f"Deduped {len(rows)} -> {len(merged)} unique rows; wrote {args.out}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    from service_analytics.analyze import analyze

    rows = load_jsonl(Path(args.input))
    summary = analyze(rows, Path(args.out), make_plots=not args.no_plots)
    print(f"Analyzed {len(rows)} rows.")
    print(f"  Total services:  {summary.get('total_services')}")
    print(f"  Unique clients:  {summary.get('unique_clients')}")
    if summary.get("peak_month"):
        print(f"  Peak month:      {summary['peak_month']}")
    if summary.get("peak_day_of_week"):
        print(f"  Peak day:        {summary['peak_day_of_week']}")
    rev = summary.get("revenue", {})
    if "grand_total" in rev:
        print(f"  Revenue total:   ${rev['grand_total']:,.2f}")
    print(f"Outputs written to {args.out}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """End-to-end: extract -> ocr -> dedupe -> analyze."""
    from service_analytics.analyze import analyze
    from service_analytics.dedupe import dedupe_rows
    from service_analytics.extract_frames import extract_frames
    from service_analytics.ocr import ocr_frames

    _load_dotenv_if_present()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    frames_dir = out / "frames"
    raw_rows_path = out / "raw_rows.jsonl"
    deduped_path = out / "services.jsonl"
    reports_dir = out / "reports"

    logger.info("Step 1/4: extracting frames")
    frames = extract_frames(
        Path(args.video),
        frames_dir,
        mode=args.mode,
        scene_threshold=args.scene_threshold,
        min_gap_s=args.min_gap,
        max_frames=args.max_frames,
        fps_fallback=args.fps_fallback,
    )

    logger.info("Step 2/4: OCR via %s", args.model)
    ocr_frames(
        frames,
        raw_rows_path,
        model=args.model,
        rate_limit_s=args.rate_limit,
        max_retries=args.max_retries,
        save_raw=args.save_raw,
        resume=args.resume,
    )

    logger.info("Step 3/4: deduping")
    rows = load_jsonl(raw_rows_path)
    merged = dedupe_rows(rows)
    dump_jsonl(merged, deduped_path)

    logger.info("Step 4/4: analyzing")
    summary = analyze(merged, reports_dir, make_plots=not args.no_plots)

    print()
    print("=" * 60)
    print("  Pipeline complete")
    print("=" * 60)
    print(f"  Frames extracted:  {len(frames)}")
    print(f"  Raw rows:          {len(rows)}")
    print(f"  Unique services:   {len(merged)}")
    print(f"  Unique clients:    {summary.get('unique_clients')}")
    if summary.get("peak_month"):
        print(f"  Peak month:        {summary['peak_month']}")
    if summary.get("peak_day_of_week"):
        print(f"  Peak day-of-week:  {summary['peak_day_of_week']}")
    rev = summary.get("revenue", {})
    if "grand_total" in rev:
        print(f"  Revenue total:     ${rev['grand_total']:,.2f}")
    print(f"  Reports:           {reports_dir}")
    return 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _add_extract_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--mode", choices=["auto", "ffmpeg", "opencv"], default="auto")
    p.add_argument(
        "--scene-threshold",
        type=float,
        default=0.04,
        help="Scene-change sensitivity (lower = more frames kept). Default 0.04.",
    )
    p.add_argument(
        "--min-gap",
        type=float,
        default=0.25,
        help="Minimum seconds between kept frames. Default 0.25.",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Hard cap on extracted frames. Default: unlimited.",
    )
    p.add_argument(
        "--fps-fallback",
        type=float,
        default=2.0,
        help="FPS to use if scene detection finds nothing. Default 2.0.",
    )


def _add_ocr_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--model",
        default=os.environ.get("SERVICE_ANALYTICS_MODEL", "google/gemini-2.5-pro"),
        help="OpenRouter model id. Default: google/gemini-2.5-pro",
    )
    p.add_argument("--rate-limit", type=float, default=0.4)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--save-raw", action="store_true", default=True)
    p.add_argument("--no-save-raw", dest="save_raw", action="store_false")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m service_analytics.cli",
        description=(
            "Turn a video of a scrolling service-report spreadsheet into "
            "a clean CSV + analytics."
        ),
    )
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    # extract
    sp = sub.add_parser("extract", help="Sample unique frames from a video")
    sp.add_argument("video")
    sp.add_argument("--out", required=True)
    _add_extract_args(sp)
    sp.set_defaults(func=cmd_extract)

    # ocr
    sp = sub.add_parser("ocr", help="Vision-OCR extracted frames to JSON-lines rows")
    sp.add_argument("frames_dir")
    sp.add_argument("--out", required=True)
    _add_ocr_args(sp)
    sp.set_defaults(func=cmd_ocr)

    # dedupe
    sp = sub.add_parser("dedupe", help="Merge overlapping rows into unique services")
    sp.add_argument("input")
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_dedupe)

    # analyze
    sp = sub.add_parser("analyze", help="Compute stats and plots from deduped rows")
    sp.add_argument("input")
    sp.add_argument("--out", required=True)
    sp.add_argument("--no-plots", action="store_true")
    sp.set_defaults(func=cmd_analyze)

    # run — end-to-end
    sp = sub.add_parser("run", help="Run the whole pipeline end-to-end")
    sp.add_argument("video")
    sp.add_argument("--out", required=True)
    _add_extract_args(sp)
    _add_ocr_args(sp)
    sp.add_argument("--no-plots", action="store_true")
    sp.set_defaults(func=cmd_run)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
