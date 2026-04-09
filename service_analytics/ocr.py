"""Vision-LLM OCR over extracted frames via OpenRouter.

Uses the same OpenRouter pattern as the rest of this repo (see
``pipeline/captioner.py``). We go through raw ``urllib`` here — not the
OpenAI SDK — so we can set ``response_format`` and reliably get back a
JSON string for structured extraction.

Default model is ``google/gemini-2.5-pro`` which handles messy phone-on-
monitor shots of dense tables much better than Flash. Override with
``--model google/gemini-2.5-flash`` for ~10x cheaper runs.

Output layout
-------------
For a run with frames in ``frames_dir`` and output at ``out_path``:

    <out_path>                  # JSON-lines, one ServiceRow per line
    <out_path>.raw/             # optional raw model responses (for debug)
        frame_000000.json
        ...
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from service_analytics.schema import RowSource, ServiceRow, dump_jsonl

logger = logging.getLogger(__name__)


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-2.5-pro"


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """\
You are extracting rows from a screenshot of an interlock service report \
spreadsheet. The user recorded a video of their monitor by phone, so the \
image may have glare, moire, skew, and motion blur — but the row layout is \
consistent.

The spreadsheet has roughly these columns (the exact set may vary):
  - service date              (the date the service was performed)
  - client name               (format: "Last, First")
  - client id                 (a 14–16 digit account / reference number)
  - technician                (person who performed the service)
  - payment status            (e.g. NoPaymentMade, Paid, etc.)
  - several dollar columns    (fees — install, monitoring, calibration, etc.)
  - follow-up / due dates     (one or more additional dates per row)

Return a strict JSON object with this shape:
{
  "rows": [
    {
      "service_date": "MM/DD/YYYY" or null,
      "client_name":  "Last, First" or null,
      "client_id":    "digits as a string" or null,
      "technician":   "name" or null,
      "payment_status": "..." or null,
      "amount_total": number or null,
      "amounts":   { "column_header": number, ... },
      "other_dates": { "column_header": "MM/DD/YYYY", ... },
      "extra":     { "column_header": "value", ... }
    }
  ],
  "header_seen": true | false,
  "notes": "brief notes about what was unreadable, if anything"
}

RULES:
- Use null (not empty string) for fields you cannot read.
- OMIT rows that are cut off at the very top or very bottom of the frame — \
only include rows that are fully visible. Partially-cut rows will be picked \
up from adjacent frames.
- Keep client_id as a STRING to preserve leading zeros. Strip internal spaces.
- Parse all dollar values as numbers without the "$" (e.g. 117.59).
- amount_total should be the sum column if the sheet has one, otherwise null.
- For amounts / other_dates / extra, use normalized snake_case header names \
(e.g. "install_fee", "next_appointment"). Put anything that doesn't fit the \
typed fields into "extra".
- Respond with ONLY the JSON object. No markdown fences, no commentary.
- If the image is entirely unreadable, return {"rows": [], "header_seen": false, "notes": "..."}.
"""


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class OCRFrameResult:
    """Result of OCR'ing a single frame."""

    frame_path: str
    frame_time_s: float
    rows: List[ServiceRow]
    header_seen: bool = False
    notes: str = ""
    raw: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def ocr_frames(
    frames: Iterable,  # list of ExtractedFrame (avoid hard import cycle)
    out_path: Path,
    *,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    rate_limit_s: float = 0.4,
    max_retries: int = 3,
    save_raw: bool = True,
    resume: bool = True,
) -> List[OCRFrameResult]:
    """Run the vision model over each frame and write rows as JSON lines.

    Args:
        frames: Iterable of ExtractedFrame (or any object with .path,
            .time_s, .index attributes).
        out_path: JSON-lines file to append rows to.
        model: OpenRouter model id.
        api_key: OPENROUTER_API_KEY override.
        rate_limit_s: Minimum seconds between API calls.
        max_retries: Retries per frame on transient errors.
        save_raw: Save raw per-frame model responses for debugging.
        resume: Skip frames already present in the existing output file.

    Returns:
        List of ``OCRFrameResult``.
    """
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY not set. Put it in .env or export it."
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = out_path.parent / (out_path.name + ".raw") if save_raw else None
    if raw_dir:
        raw_dir.mkdir(parents=True, exist_ok=True)

    done_frames = _already_done_frames(out_path) if resume else set()

    results: List[OCRFrameResult] = []
    last_call = 0.0
    # open in append so we don't stomp prior progress on resume
    mode = "a" if (resume and out_path.exists()) else "w"
    with open(out_path, mode, encoding="utf-8") as out_f:
        for fr in frames:
            fpath = Path(fr.path)
            if str(fpath) in done_frames:
                logger.info("Skipping already-OCR'd frame: %s", fpath.name)
                continue

            # rate-limit between calls
            elapsed = time.time() - last_call
            if elapsed < rate_limit_s:
                time.sleep(rate_limit_s - elapsed)

            result = _ocr_single_frame(
                fpath,
                float(fr.time_s),
                model=model,
                api_key=api_key,
                max_retries=max_retries,
            )
            last_call = time.time()
            results.append(result)

            if save_raw and result.raw is not None:
                raw_path = raw_dir / f"{fpath.stem}.json"
                with open(raw_path, "w", encoding="utf-8") as rf:
                    json.dump(result.raw, rf, indent=2, default=str)

            # stream rows to disk immediately for crash safety
            for row in result.rows:
                row.sources.append(
                    RowSource(
                        frame_path=str(fpath),
                        frame_time_s=float(fr.time_s),
                        model=model,
                    )
                )
                out_f.write(json.dumps(row.to_dict(), default=str))
                out_f.write("\n")
            out_f.flush()

            if result.error:
                logger.warning("Frame %s: %s", fpath.name, result.error)
            else:
                logger.info(
                    "Frame %s: %d rows (header_seen=%s)",
                    fpath.name,
                    len(result.rows),
                    result.header_seen,
                )

    return results


# ---------------------------------------------------------------------------
# Single-frame call
# ---------------------------------------------------------------------------


def _ocr_single_frame(
    frame_path: Path,
    frame_time_s: float,
    *,
    model: str,
    api_key: str,
    max_retries: int,
) -> OCRFrameResult:
    """Call OpenRouter once for a single frame, parse the result."""
    try:
        with open(frame_path, "rb") as f:
            img_bytes = f.read()
    except OSError as e:
        return OCRFrameResult(
            frame_path=str(frame_path),
            frame_time_s=frame_time_s,
            rows=[],
            error=f"could not read frame: {e}",
        )

    mime = "image/jpeg" if frame_path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    b64 = base64.b64encode(img_bytes).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"

    payload = json.dumps(
        {
            "model": model,
            "max_tokens": 8000,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": EXTRACTION_PROMPT},
                    ],
                }
            ],
        }
    ).encode("utf-8")

    last_err: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                OPENROUTER_URL,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": "https://github.com/alexkorol/pixel-perfecter",
                    "X-Title": "pixel-perfecter service_analytics",
                },
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                raw_data = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:400] if hasattr(e, "read") else ""
            last_err = f"HTTP {e.code}: {body}"
            if e.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                logger.warning(
                    "OpenRouter %s on %s, retrying in %ds",
                    e.code,
                    frame_path.name,
                    wait,
                )
                time.sleep(wait)
                continue
            break
        except urllib.error.URLError as e:
            last_err = f"URLError: {e}"
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            break
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            break

    if raw_data is None:
        return OCRFrameResult(
            frame_path=str(frame_path),
            frame_time_s=frame_time_s,
            rows=[],
            error=last_err or "unknown error",
        )

    content = _extract_content_text(raw_data)
    parsed = _parse_extraction_json(content)

    row_objs: List[ServiceRow] = []
    if isinstance(parsed, dict):
        for rd in parsed.get("rows") or []:
            if isinstance(rd, dict):
                row_objs.append(ServiceRow.from_dict(rd))

    return OCRFrameResult(
        frame_path=str(frame_path),
        frame_time_s=frame_time_s,
        rows=row_objs,
        header_seen=bool(parsed.get("header_seen")) if isinstance(parsed, dict) else False,
        notes=str(parsed.get("notes", "")) if isinstance(parsed, dict) else "",
        raw=raw_data,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_content_text(raw: Dict[str, Any]) -> str:
    """Pull the text content out of an OpenRouter chat-completions response."""
    try:
        msg = raw["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return ""
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    # Some providers return a list of content parts
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                texts.append(part.get("text", ""))
        return "\n".join(texts)
    return ""


_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _parse_extraction_json(text: str) -> Any:
    """Parse a JSON response, stripping fences and finding the outer object."""
    if not text:
        return {}
    cleaned = _JSON_FENCE_RE.sub("", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Try to find the first {...} block
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return {}


def _already_done_frames(out_path: Path) -> set:
    """Return the set of frame_paths that already have rows in the output.

    Used for resume — we skip any frame that's already represented in the
    existing JSON-lines file.
    """
    done = set()
    if not out_path.exists():
        return done
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for src in d.get("sources") or []:
                    if isinstance(src, dict) and src.get("frame_path"):
                        done.add(src["frame_path"])
    except OSError:
        pass
    return done
