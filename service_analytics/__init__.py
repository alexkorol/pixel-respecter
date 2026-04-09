"""Video-to-service-data analytics for interlock service spreadsheets.

Turns a video of a scrolling spreadsheet (e.g. a phone recording of a
monitor) into a clean CSV of service records, then runs analytics on
volume, clients, peak months/days, and service types.

Pipeline stages:
  1. extract_frames — sample unique frames from the video
  2. ocr            — run a vision model over each frame to get structured rows
  3. dedupe         — collapse overlapping frames into one row per service
  4. analyze        — pandas + matplotlib reports

See the package README.md for setup and CLI usage.
"""

from service_analytics.schema import ServiceRow, RowSource

__all__ = ["ServiceRow", "RowSource"]
