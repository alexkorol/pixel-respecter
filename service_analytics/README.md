# service_analytics

Turn a video of a scrolling service-report spreadsheet into a clean CSV of
service records plus analytics (volume, unique clients, peak months/days,
service-type breakdown, revenue).

Built for the interlock shop's 6-month "ECO by Payment Type and Date Range"
report when all you have is a phone video of the monitor — no original
spreadsheet file.

## How it works

```
  video.mp4
     │
     ├─► 1. extract_frames    (ffmpeg scene-change detection)
     │       │
     │       └─► frames/frame_NNNNNN_t=S.SSS.jpg
     │
     ├─► 2. ocr              (OpenRouter → Gemini 2.5 Pro, structured JSON)
     │       │
     │       └─► raw_rows.jsonl  (one row per sighting, may be duplicates)
     │
     ├─► 3. dedupe           (group by (client_id, date), majority-vote merge)
     │       │
     │       └─► services.jsonl  (one row per unique service)
     │
     └─► 4. analyze          (pandas + matplotlib)
             │
             └─► reports/
                     services.csv
                     summary.json
                     services_by_month.png
                     services_by_dow.png
                     revenue_by_month.png
                     payment_status.png
```

## Setup

You need ffmpeg on your PATH (`brew install ffmpeg` / `apt install ffmpeg`)
and these Python packages:

```bash
pip install pandas matplotlib python-dotenv
# opencv is already in the repo's requirements.txt
```

Add your OpenRouter key to `.env` at the repo root (same file used by the
rest of the repo):

```
OPENROUTER_API_KEY=sk-or-...
```

## Usage

### End-to-end

```bash
python -m service_analytics.cli run path/to/video.mp4 \
    --out data/service-analytics/run1
```

This runs all four stages and writes everything under `data/service-analytics/run1/`.

### Piecewise (when iterating)

```bash
# 1. Extract frames only
python -m service_analytics.cli extract video.mp4 --out data/run1/frames

# 2. OCR a frames directory
python -m service_analytics.cli ocr data/run1/frames --out data/run1/raw_rows.jsonl

# 3. Dedupe
python -m service_analytics.cli dedupe data/run1/raw_rows.jsonl \
    --out data/run1/services.jsonl

# 4. Analyze
python -m service_analytics.cli analyze data/run1/services.jsonl \
    --out data/run1/reports
```

## Tuning knobs

| Flag | Default | What to change it for |
|---|---|---|
| `--scene-threshold` | `0.04` | Lower (e.g. `0.02`) if the scroll is very slow and frames are being skipped. Raise if you're getting too many near-identical frames. |
| `--min-gap` | `0.25` s | Minimum seconds between kept frames. Raise if the video is smooth-scrolling and you're getting 5 frames of the same viewport. |
| `--fps-fallback` | `2.0` | Used if scene detection finds nothing (video too uniform). Bumps to fixed-rate sampling. |
| `--max-frames` | unlimited | Safety cap. Useful for dry runs on long videos. |
| `--model` | `google/gemini-2.5-pro` | Use `google/gemini-2.5-flash` for ~10x cheaper / faster runs at somewhat lower accuracy. |
| `--rate-limit` | `0.4` s | Minimum seconds between API calls. |
| `--no-resume` | | By default, rerunning `ocr` skips frames already in the output file. Pass this to force re-OCR. |

## What the model is asked for

The vision model is told the spreadsheet is an interlock service report
and asked to return strict JSON with one object per row:

```json
{
  "rows": [
    {
      "service_date": "10/29/2025",
      "client_name": "Blaugerberr, Elijah",
      "client_id": "0059022300190034",
      "technician": "Adams, Michael",
      "payment_status": "NoPaymentMade",
      "amount_total": 82.00,
      "amounts": { "install_fee": 50.00, "monitoring": 32.00 },
      "other_dates": { "next_appointment": "11/29/2025" },
      "extra": {}
    }
  ],
  "header_seen": false,
  "notes": ""
}
```

The prompt explicitly tells the model to **omit rows cut off at the top/bottom
of the frame** — those partial rows are picked up from adjacent frames and
reassembled during dedupe. See the full prompt in `service_analytics/ocr.py`.

## Dedupe logic

Rows are grouped by a stable key:

1. `(client_id, service_date)` if client_id is present (the best key — a
   14–16 digit number is far more unique than a name).
2. `(client_name, service_date)` as a fallback.

Within each group, fields are merged with **majority voting**: if three
frames all say `"Adams, Michael"` and one says `"Adarns, Michael"`, the
majority wins. This is robust to individual-frame OCR errors.

Sources (frame paths + timestamps) are concatenated so every merged row
retains provenance.

## PII note

This video contains real customer data. The pipeline sends each frame to
OpenRouter → Google. Make sure your shop is OK with that under their data
agreements. Everything else stays on disk locally.

If you need a fully-local alternative later, the ffmpeg frame extraction,
dedupe, and analytics stages are all local-only — only `ocr.py` talks to
the network. You could swap it for PaddleOCR or a local LLaVA model.

## Testing

```bash
python -m pytest tests/test_service_analytics.py -v
```

Tests cover schema parsing, dedupe logic, and the analysis summary. They
do not hit the network or require ffmpeg.
