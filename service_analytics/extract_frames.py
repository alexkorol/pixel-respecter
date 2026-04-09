"""Sample unique frames from a video of a scrolling spreadsheet.

Strategy
--------
A phone recording of a monitor shows a lot of redundant motion: the same
rows visible for many consecutive frames. We only want to OCR frames that
actually added new rows to the viewport.

Two extraction modes are supported:

  1. ``ffmpeg`` (preferred) — uses ffmpeg's ``select='gt(scene,T)'`` filter
     to drop frames that don't differ enough from the previous frame. Fast
     and does not require loading frames in Python.

  2. ``opencv`` (fallback) — iterates the video with OpenCV and keeps a
     frame when the mean absolute difference against the last-kept frame
     exceeds a threshold. Works without an ffmpeg binary.

Both modes produce the same output directory layout:

    <out_dir>/
        frame_000000_t=0.000.jpg
        frame_000001_t=1.350.jpg
        ...
        manifest.json

The manifest stores ``[{"path": "...", "time_s": 1.35, "index": 1}, ...]``
so downstream stages don't need to re-parse the filenames.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ExtractedFrame:
    path: str
    time_s: float
    index: int

    def to_dict(self):
        return {"path": self.path, "time_s": self.time_s, "index": self.index}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def extract_frames(
    video_path: Path,
    out_dir: Path,
    *,
    mode: str = "auto",
    scene_threshold: float = 0.04,
    min_gap_s: float = 0.25,
    max_frames: Optional[int] = None,
    fps_fallback: float = 2.0,
) -> List[ExtractedFrame]:
    """Extract a list of unique frames from ``video_path`` into ``out_dir``.

    Args:
        video_path: Path to the input video.
        out_dir: Directory to write frames + manifest to. Will be created.
        mode: ``auto``, ``ffmpeg``, or ``opencv``. ``auto`` uses ffmpeg if
            available, else opencv.
        scene_threshold: Scene-change sensitivity. ffmpeg's scene score is
            roughly 0..1, where 0.04 catches slow scrolling. Raise for less
            sensitive (fewer frames), lower for more.
        min_gap_s: Minimum seconds between kept frames; prevents near-identical
            frames when the video is very high-framerate.
        max_frames: Cap on number of frames extracted (safety). None = no cap.
        fps_fallback: If scene detection picks zero frames (e.g. the scroll
            is too slow), fall back to this fixed fps.

    Returns:
        List of ``ExtractedFrame`` records, also written to ``out_dir/manifest.json``.
    """
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved = _resolve_mode(mode)
    logger.info("Extracting frames from %s via %s", video_path, resolved)

    if resolved == "ffmpeg":
        frames = _extract_ffmpeg(
            video_path,
            out_dir,
            scene_threshold=scene_threshold,
            min_gap_s=min_gap_s,
        )
        if not frames:
            logger.warning(
                "Scene-change extraction returned 0 frames, falling back to "
                "fixed %.2f fps sampling",
                fps_fallback,
            )
            frames = _extract_ffmpeg_fixed_fps(video_path, out_dir, fps_fallback)
    else:
        frames = _extract_opencv(
            video_path,
            out_dir,
            scene_threshold=scene_threshold,
            min_gap_s=min_gap_s,
            fps_fallback=fps_fallback,
        )

    if max_frames is not None and len(frames) > max_frames:
        logger.info("Capping extracted frames: %d -> %d", len(frames), max_frames)
        # keep evenly-spaced frames up to the cap
        step = len(frames) / max_frames
        frames = [frames[int(i * step)] for i in range(max_frames)]

    manifest = out_dir / "manifest.json"
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump([fr.to_dict() for fr in frames], f, indent=2)
    logger.info("Wrote %d frames to %s", len(frames), out_dir)
    return frames


def load_manifest(frames_dir: Path) -> List[ExtractedFrame]:
    """Load the frame manifest written by ``extract_frames``."""
    p = Path(frames_dir) / "manifest.json"
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        ExtractedFrame(path=d["path"], time_s=float(d["time_s"]), index=int(d["index"]))
        for d in data
    ]


# ---------------------------------------------------------------------------
# Mode resolution
# ---------------------------------------------------------------------------


def _resolve_mode(mode: str) -> str:
    if mode == "auto":
        return "ffmpeg" if shutil.which("ffmpeg") else "opencv"
    if mode == "ffmpeg" and not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg mode requested but 'ffmpeg' is not on PATH")
    if mode not in ("ffmpeg", "opencv"):
        raise ValueError(f"Unknown extraction mode: {mode}")
    return mode


# ---------------------------------------------------------------------------
# ffmpeg backend
# ---------------------------------------------------------------------------


def _extract_ffmpeg(
    video_path: Path,
    out_dir: Path,
    *,
    scene_threshold: float,
    min_gap_s: float,
) -> List[ExtractedFrame]:
    """Extract scene-change frames using ffmpeg's ``select`` filter.

    We use ``showinfo`` to capture each kept frame's timestamp from stderr.
    """
    pattern = out_dir / "frame_%06d.jpg"
    # select frames where scene score > threshold, also filter out frames
    # closer than min_gap_s apart via a separate time-gated rolling state.
    # ffmpeg can't easily do the min-gap natively, so we rely on scene
    # threshold here and do a second-pass filter in Python below.
    vf = f"select='gt(scene,{scene_threshold})',showinfo"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "info",
        "-i",
        str(video_path),
        "-vf",
        vf,
        "-vsync",
        "vfr",
        "-q:v",
        "3",
        str(pattern),
    ]
    logger.debug("Running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {proc.returncode}):\n{proc.stderr[-800:]}"
        )
    # showinfo writes lines like: "... pts_time:1.234567 ..."
    times: List[float] = []
    for line in proc.stderr.splitlines():
        if "pts_time:" not in line or "Parsed_showinfo" not in line:
            continue
        try:
            t = float(line.split("pts_time:")[1].split()[0])
            times.append(t)
        except (IndexError, ValueError):
            continue

    frame_files = sorted(out_dir.glob("frame_*.jpg"))
    if len(frame_files) != len(times):
        # alignment mismatch — best effort: zip shorter length
        logger.warning(
            "ffmpeg frame count (%d) does not match showinfo time count (%d); "
            "aligning by minimum",
            len(frame_files),
            len(times),
        )
    n = min(len(frame_files), len(times))
    results: List[ExtractedFrame] = []
    last_kept_t = -1e9
    kept_idx = 0
    for i in range(n):
        t = times[i]
        if (t - last_kept_t) < min_gap_s:
            frame_files[i].unlink(missing_ok=True)
            continue
        new_name = out_dir / f"frame_{kept_idx:06d}_t={t:.3f}.jpg"
        frame_files[i].rename(new_name)
        results.append(ExtractedFrame(path=str(new_name), time_s=t, index=kept_idx))
        kept_idx += 1
        last_kept_t = t
    return results


def _extract_ffmpeg_fixed_fps(
    video_path: Path,
    out_dir: Path,
    fps: float,
) -> List[ExtractedFrame]:
    """Fallback: sample at a fixed framerate."""
    # clear any stale frames
    for p in out_dir.glob("frame_*.jpg"):
        p.unlink(missing_ok=True)
    pattern = out_dir / "frame_%06d.jpg"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps},showinfo",
        "-q:v",
        "3",
        str(pattern),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg fallback failed (exit {proc.returncode}):\n{proc.stderr[-800:]}"
        )
    results = []
    frame_files = sorted(out_dir.glob("frame_*.jpg"))
    for i, path in enumerate(frame_files):
        t = i / fps
        new_name = out_dir / f"frame_{i:06d}_t={t:.3f}.jpg"
        path.rename(new_name)
        results.append(ExtractedFrame(path=str(new_name), time_s=t, index=i))
    return results


# ---------------------------------------------------------------------------
# OpenCV backend
# ---------------------------------------------------------------------------


def _extract_opencv(
    video_path: Path,
    out_dir: Path,
    *,
    scene_threshold: float,
    min_gap_s: float,
    fps_fallback: float,
) -> List[ExtractedFrame]:
    """Scene-diff frame extraction using OpenCV. No ffmpeg required."""
    import cv2  # local import so the package imports without cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    logger.info("OpenCV video: %d frames @ %.2f fps", total, src_fps)

    # OpenCV's scene score is abstract — we use mean-abs-diff on downsampled
    # grayscale frames, normalized to 0..1. 0.04 is a reasonable default for
    # slow scrolling text.
    kept: List[ExtractedFrame] = []
    last_gray = None
    last_kept_t = -1e9
    kept_idx = 0
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / src_fps
        frame_idx += 1
        small = cv2.cvtColor(
            cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2GRAY,
        )
        if last_gray is None:
            score = 1.0
        else:
            diff = cv2.absdiff(small, last_gray)
            score = float(diff.mean()) / 255.0

        if score > scene_threshold and (t - last_kept_t) >= min_gap_s:
            path = out_dir / f"frame_{kept_idx:06d}_t={t:.3f}.jpg"
            cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            kept.append(ExtractedFrame(path=str(path), time_s=t, index=kept_idx))
            kept_idx += 1
            last_gray = small
            last_kept_t = t

    cap.release()

    if not kept:
        # nothing tripped the scene threshold — sample at fps_fallback
        logger.warning(
            "OpenCV scene extraction got 0 frames, falling back to %.2f fps",
            fps_fallback,
        )
        cap = cv2.VideoCapture(str(video_path))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(src_fps / fps_fallback)))
        frame_idx = 0
        kept_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % step == 0:
                t = frame_idx / src_fps
                path = out_dir / f"frame_{kept_idx:06d}_t={t:.3f}.jpg"
                cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                kept.append(ExtractedFrame(path=str(path), time_s=t, index=kept_idx))
                kept_idx += 1
            frame_idx += 1
        cap.release()

    return kept
