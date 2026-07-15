"""Serve Fix Lab and persist its calibration data into the repository.

Run from the repository root::

    python -m fixlab.server

The static UI still works under ``python -m http.server``; in that mode it
downloads JSONL/PNG files because the persistence endpoints are unavailable.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "benchmarks" / "wild_corpus"
ID_RE = re.compile(r"^[a-f0-9]{16}$")


class FixLabHandler(SimpleHTTPRequestHandler):
    """Static-file handler with narrow, repository-local write endpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_POST(self) -> None:  # noqa: N802 - stdlib hook name
        match = re.fullmatch(
            r"/api/(fixes|ground_truth)/([a-f0-9]{16})", urlparse(self.path).path
        )
        if not match or not ID_RE.fullmatch(match.group(2)):
            self._json(404, {"error": "unknown endpoint"})
            return
        product, image_id = match.groups()
        try:
            payload = self._read_json()
            if product == "fixes":
                decision = payload.get("decision")
                if not isinstance(decision, dict) or decision.get("id") != image_id:
                    raise ValueError("decision id does not match URL")
                target = CORPUS / "fixes" / f"{image_id}.jsonl"
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(json.dumps(decision, separators=(",", ":")) + "\n")
            else:
                raw = base64.b64decode(payload.get("png", ""), validate=True)
                if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
                    raise ValueError("payload is not a PNG")
                target = CORPUS / "ground_truth" / f"{image_id}.png"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._json(400, {"error": str(exc)})
            return
        self._json(200, {"ok": True, "path": target.relative_to(ROOT).as_posix()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.bind, args.port), FixLabHandler)
    print(f"Fix Lab: http://{args.bind}:{args.port}/fixlab/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
