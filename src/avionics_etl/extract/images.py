"""Image ingest: a content-addressed blob store plus a Parquet manifest.

The spec requires the platform to support image processing — visual inspection
of recovered hardware for scorching, corrosion and connector damage. C-MAPSS has
no images, so this builds the *path* rather than the content: when real
inspection photographs arrive they slot in without redesign.

Two rules shape the design.

**Images never go inside Parquet.**  A columnar format is built for scanning
many small typed values, not for multi-megabyte opaque blobs. Storing them
inline would bloat every row group and make any query about a temperature
reading drag image bytes along with it. Instead the pixels live on disk and
Parquet holds a *manifest* — small, typed, and joinable to telemetry on
``(source_id, mission_id, asset_id)``.

**Blobs are content-addressed.**  A blob's filename is the SHA-256 of its bytes.
Re-ingesting the same photograph is a no-op, the same image arriving through two
routes is stored once, and any later corruption is detectable by rehashing. This
is how git stores objects.

::

    data/images/
      blobs/
        3f/3fa9c1...e2.png          <- filename IS the content hash
        a0/a0b47d...91.jpg
      manifest/
        source_id=TV-D1/batch_id=TV-D1-recovery-03/part-0.parquet

Usage::

    python -m avionics_etl.extract.images
    python -m avionics_etl.extract.images --list
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from avionics_etl import config, lineage
from avionics_etl.canonical import model
from avionics_etl.load import parquet_writer

BLOBS = "blobs"
MANIFEST = "manifest"

MANIFEST_COLUMNS = [
    "image_id",          # SHA-256 of the file's bytes — the primary key
    model.SOURCE_ID,
    model.MISSION_ID,
    model.ASSET_ID,
    model.BATCH_ID,
    "sequence_no",       # nullable: an image may not map to a specific cycle
    "captured_at",
    "media_type",
    "width",
    "height",
    "bytes",
    "blob_path",
    "original_name",
]


@dataclass(frozen=True)
class ImageRecord:
    """One discovered image, before it is stored."""

    path: Path
    mission_id: str
    asset_id: str


def _digest(path: Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def _dimensions(path: Path) -> tuple[int | None, int | None]:
    """Pixel dimensions, if an image library is available.

    Deliberately optional: a missing decoder degrades the manifest, it does not
    stop ingestion. The bytes are what matter.
    """
    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError:
        return None, None
    try:
        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except Exception:
        return None, None


def discover(cfg: config.Config) -> list[ImageRecord]:
    """Walk the configured image source, parsing entity ids from the path.

    Expected layout is ``<source>/<mission_id>/<asset_id>/<file>``. Files that do
    not sit at that depth are skipped with a warning rather than guessed at —
    mislabelling which asset a photograph belongs to is worse than ignoring it.
    """
    img_cfg = cfg.get("images", {}) or {}
    root = Path(str(img_cfg.get("source", ""))).expanduser()
    if not str(root) or not root.exists():
        return []

    media_types = {e.lower() for e in img_cfg.get("media_types", [])}
    found: list[ImageRecord] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in media_types:
            continue
        rel = path.relative_to(root)
        if len(rel.parts) < 3:
            print(f"  skipping {rel}: expected <mission_id>/<asset_id>/<file>", file=sys.stderr)
            continue
        found.append(ImageRecord(path=path, mission_id=rel.parts[0], asset_id=rel.parts[1]))
    return found


def store_blob(path: Path, blob_root: Path, *, depth: int = 2) -> tuple[str, Path, bool]:
    """Copy an image into the content-addressed store.

    Returns ``(image_id, blob_path, was_new)``. A blob that already exists is
    left untouched — identical content cannot be stored twice.
    """
    image_id = _digest(path)
    target_dir = blob_root / image_id[:depth]
    target = target_dir / f"{image_id}{path.suffix.lower()}"
    if target.exists():
        return image_id, target, False
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
    return image_id, target, True


def ingest(cfg: config.Config, *, source_id: str | None = None, batch_id: str | None = None) -> lineage.Lineage:
    """Store discovered images and write their manifest."""
    rec = lineage.Lineage.start("images", cfg.hash)
    images_root = cfg.ensure("images")
    blob_root = images_root / BLOBS
    img_cfg = cfg.get("images", {}) or {}
    depth = int(img_cfg.get("blob_depth", 2))

    records = discover(cfg)
    if not records:
        src = img_cfg.get("source", "<unset>")
        print(
            f"images : no image source at {src} — nothing to ingest.\n"
            f"         The blob store and manifest schema are ready; real "
            f"inspection images can be dropped in at any time."
        )
        rec.notes["images_found"] = 0
        rec.finish(cfg.ensure("artifacts") / "manifests")
        return rec

    source_id = source_id or "inspection"
    batch_id = batch_id or f"{source_id}-{datetime.now(timezone.utc):%Y%m%d}"
    stamp = rec.stamp()

    rows: list[dict[str, object]] = []
    new_blobs = 0
    for item in records:
        image_id, blob_path, was_new = store_blob(item.path, blob_root, depth=depth)
        new_blobs += was_new
        width, height = _dimensions(blob_path)
        stat = item.path.stat()
        rows.append(
            {
                "image_id": image_id,
                model.SOURCE_ID: source_id,
                model.MISSION_ID: item.mission_id,
                model.ASSET_ID: item.asset_id,
                model.BATCH_ID: batch_id,
                "sequence_no": pd.NA,
                "captured_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds"),
                "media_type": item.path.suffix.lower().lstrip("."),
                "width": width,
                "height": height,
                "bytes": stat.st_size,
                "blob_path": str(blob_path.relative_to(images_root)),
                "original_name": item.path.name,
                **stamp,
            }
        )

    manifest = pd.DataFrame(rows, columns=[*MANIFEST_COLUMNS, *stamp])
    manifest["sequence_no"] = manifest["sequence_no"].astype("Int32")
    manifest = manifest.drop_duplicates(subset=["image_id"])

    parquet_writer.write_partitioned(
        manifest,
        images_root / MANIFEST,
        partition_by=[model.SOURCE_ID, model.BATCH_ID],
        sort_by=[model.MISSION_ID, model.ASSET_ID, "image_id"],
        compression=cfg.get("parquet.compression.bronze", "zstd"),
        row_group_bytes=int(cfg.get("parquet.row_group_bytes", 1 << 27)),
    )

    rec.notes.update(
        {"images_found": len(records), "unique_images": len(manifest), "new_blobs": new_blobs}
    )
    rec.add_output(f"{MANIFEST}/{batch_id}", len(manifest))
    rec.finish(cfg.ensure("artifacts") / "manifests")

    dupes = len(records) - len(manifest)
    print(
        f"images : {len(manifest):,} unique image(s), {new_blobs} new blob(s)"
        + (f", {dupes} duplicate(s) deduplicated by content" if dupes else "")
    )
    return rec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true", help="show what would be ingested")
    parser.add_argument("--source-id", default=None)
    parser.add_argument("--batch-id", default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    cfg = config.load(args.config)
    if args.list:
        records = discover(cfg)
        print(f"images : {len(records)} file(s) under {cfg.get('images.source')}")
        for r in records[:20]:
            print(f"  {r.mission_id}/{r.asset_id}/{r.path.name}")
        return 0

    ingest(cfg, source_id=args.source_id, batch_id=args.batch_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
