"""Ingest real AI-generated images into the wild benchmark corpus.

Copies images into benchmarks/wild_corpus/images/ under content-hash names
(deduplicating re-runs) and updates manifest.json.

    python benchmarks/ingest_wild.py input/ --source chatgpt-image
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

from PIL import Image

CORPUS_DIR = Path(__file__).resolve().parent / "wild_corpus"
MANIFEST_PATH = CORPUS_DIR / "manifest.json"
IMAGES_DIR = CORPUS_DIR / "images"
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {"entries": []}


def save_manifest(manifest: dict) -> None:
    manifest["entries"].sort(key=lambda e: e["id"])
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def ingest(source_dir: Path, source_tag: str, recursive: bool) -> None:
    manifest = load_manifest()
    known = {e["id"] for e in manifest["entries"]}
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    pattern = "**/*" if recursive else "*"
    added = skipped = 0
    for path in sorted(source_dir.glob(pattern)):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()[:16]
        if digest in known:
            skipped += 1
            continue
        try:
            with Image.open(path) as im:
                width, height = im.size
        except Exception as exc:  # pylint: disable=broad-except
            print(f"skip (unreadable): {path.name} ({exc})")
            continue
        dest = IMAGES_DIR / f"{digest}{path.suffix.lower()}"
        shutil.copy2(path, dest)
        manifest["entries"].append(
            {
                "id": digest,
                "file": f"images/{dest.name}",
                "original_name": path.name,
                "width": width,
                "height": height,
                "source": source_tag,
            }
        )
        known.add(digest)
        added += 1

    save_manifest(manifest)
    print(
        f"Added {added}, skipped {skipped} duplicates. "
        f"Corpus size: {len(manifest['entries'])} images."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Directory of images to ingest")
    parser.add_argument(
        "--source",
        dest="source_tag",
        default="unknown",
        help="Provenance tag stored in the manifest (e.g. chatgpt-image, midjourney)",
    )
    parser.add_argument("--recursive", action="store_true", help="Walk subdirectories")
    args = parser.parse_args()

    if not args.source.is_dir():
        sys.exit(f"Not a directory: {args.source}")
    ingest(args.source, args.source_tag, args.recursive)


if __name__ == "__main__":
    main()
