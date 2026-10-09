"""Source-agnostic ingest driver: any adapter -> the bronze layer.

This module knows nothing about C-MAPSS. It takes a
:class:`~avionics_etl.adapters.base.SourceAdapter`, asks what batches are
available, and streams each one into partitioned Parquet.

Three properties matter, and all three come from the spec's operating model —
"of the order of 64GB", "terabytes", "on a continuous operational basis, data
shall be fed forward from previous simulations, flights etc.":

**Bounded memory.**  Batches are read as chunks and written as row groups, so
peak memory is proportional to the chunk size, not the batch size.

**Append-only.**  A batch is written once. Previously ingested batches are never
rewritten, so history accumulates across missions instead of being rebuilt.

**Idempotent.**  An ingest run skips batches already recorded in the watermark
log. Re-running after a crash is safe and cheap; ``--force`` re-ingests.

See docs/PRIMER.md section 11.

Usage::

    python -m avionics_etl.extract.ingest                 # ingest what is new
    python -m avionics_etl.extract.ingest --force         # re-ingest everything
    python -m avionics_etl.extract.ingest --batch FD001-train
    python -m avionics_etl.extract.ingest --list
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from avionics_etl import config, lineage
from avionics_etl.adapters.base import Batch, SourceAdapter
from avionics_etl.adapters.cmapss import CmapssAdapter
from avionics_etl.canonical import model
from avionics_etl.load import parquet_writer

MEASUREMENTS = "measurements"
EXTERNAL_LABELS = "external_labels"
WATERMARK = "ingest_log.json"


# ---------------------------------------------------------------------------
# watermark
# ---------------------------------------------------------------------------

def watermark_path(cfg: config.Config) -> Path:
    return cfg.ensure("artifacts") / WATERMARK


def load_watermark(cfg: config.Config) -> dict[str, dict]:
    """Which batches have already been ingested, keyed by batch id."""
    path = watermark_path(cfg)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_watermark(cfg: config.Config, log: dict[str, dict]) -> None:
    watermark_path(cfg).write_text(json.dumps(log, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

def ingest_batch(
    adapter: SourceAdapter,
    batch: Batch,
    cfg: config.Config,
    rec: lineage.Lineage,
    *,
    chunk_rows: int,
) -> dict:
    """Stream one batch into bronze. Returns its watermark record."""
    bronze = cfg.ensure("bronze")
    pq_cfg = cfg["parquet"]
    stamp = rec.stamp()

    keys = {model.SOURCE_ID: batch.source_id, model.BATCH_ID: batch.batch_id}
    writer = parquet_writer.StreamingPartitionWriter(
        bronze / MEASUREMENTS,
        keys,
        compression=pq_cfg["compression"]["bronze"],
        row_group_bytes=pq_cfg["row_group_bytes"],
    )

    with writer:
        for chunk in adapter.read_chunks(batch, chunk_rows=chunk_rows):
            for col, value in stamp.items():
                chunk[col] = value
            writer.write(chunk)

    sorted_ok = parquet_writer.verify_sorted(writer.path, model.SORT_COLS)
    if not sorted_ok:
        # Not fatal — but it means row-group statistics are useless for this
        # partition, so predicate pushdown degrades to a full scan.
        print(
            f"  WARNING {batch.batch_id}: rows are not ordered by "
            f"{model.SORT_COLS}; the adapter should emit sorted chunks",
            file=sys.stderr,
        )

    # External labels, where the source supplies them separately.
    label_rows = 0
    reader = getattr(adapter, "read_external_labels", None)
    if reader is not None and (labels := reader(batch)) is not None:
        for col, value in stamp.items():
            labels[col] = value
        parquet_writer.write_partitioned(
            labels,
            bronze / EXTERNAL_LABELS,
            partition_by=[model.SOURCE_ID, model.BATCH_ID],
            sort_by=[model.ASSET_ID],
            compression=pq_cfg["compression"]["bronze"],
            row_group_bytes=pq_cfg["row_group_bytes"],
        )
        label_rows = len(labels)

    for f in batch.files:
        rec.add_source(f)
    rec.add_output(f"{MEASUREMENTS}/{batch.batch_id}", writer.rows)

    return {
        "batch_id": batch.batch_id,
        "source_id": batch.source_id,
        "mission_id": batch.mission_id,
        "adapter": adapter.name,
        "trajectory_complete": batch.trajectory_complete,
        "rows": writer.rows,
        "label_rows": label_rows,
        "bytes_in": batch.total_bytes,
        "sorted": sorted_ok,
        "files": {f.name: lineage.file_digest(f) for f in batch.files},
        "run_id": rec.run_id,
        "config_hash": cfg.hash,
        "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def ingest(
    adapter: SourceAdapter,
    cfg: config.Config,
    *,
    force: bool = False,
    only: list[str] | None = None,
    chunk_rows: int | None = None,
) -> lineage.Lineage:
    """Ingest every new batch the adapter offers."""
    rec = lineage.Lineage.start("bronze", cfg.hash)
    log = load_watermark(cfg)
    chunk_rows = chunk_rows or int(cfg.get("ingest.chunk_rows", 500_000))

    batches = adapter.discover()
    if only:
        wanted = set(only)
        batches = [b for b in batches if b.batch_id in wanted]
        if missing := wanted - {b.batch_id for b in batches}:
            raise KeyError(f"unknown batch ids: {sorted(missing)}")

    ingested, skipped = 0, 0
    total_rows = 0
    for batch in batches:
        prior = log.get(batch.batch_id)
        if prior and not force:
            current = {f.name: lineage.file_digest(f) for f in batch.files}
            if current == prior.get("files"):
                skipped += 1
                continue
            print(f"  {batch.batch_id}: source files changed, re-ingesting")

        record = ingest_batch(adapter, batch, cfg, rec, chunk_rows=chunk_rows)
        log[batch.batch_id] = record
        ingested += 1
        total_rows += record["rows"]
        print(f"  {batch.batch_id:<14} {record['rows']:>7,} rows")

    save_watermark(cfg, log)
    rec.notes.update(
        {"batches_ingested": ingested, "batches_skipped": skipped, "rows": total_rows}
    )
    rec.finish(cfg.ensure("artifacts") / "manifests")

    print(
        f"bronze : {ingested} batch(es) ingested, {skipped} already current, "
        f"{total_rows:,} rows written"
    )
    return rec


# ---------------------------------------------------------------------------

def build_adapter(cfg: config.Config, name: str) -> SourceAdapter:
    """Resolve an adapter by name.

    The one place a source name maps to an implementation. Adding the ISRO
    source means adding one entry here.
    """
    registry_path = config.REPO_ROOT / "configs" / "channels.yaml"
    if name == "cmapss":
        return CmapssAdapter(cfg.path("raw"), registry_path, cfg.subsets)
    raise KeyError(f"unknown adapter {name!r}; known adapters: ['cmapss']")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--adapter", default="cmapss", help="source adapter (default: cmapss)")
    parser.add_argument("--batch", action="append", dest="batches", help="ingest only this batch id (repeatable)")
    parser.add_argument("--force", action="store_true", help="re-ingest batches already recorded")
    parser.add_argument("--chunk-rows", type=int, default=None, help="rows per read chunk")
    parser.add_argument("--list", action="store_true", help="list available batches and exit")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    cfg = config.load(args.config)
    adapter = build_adapter(cfg, args.adapter)

    if args.list:
        log = load_watermark(cfg)
        print(adapter.describe())
        for b in adapter.discover():
            state = "ingested" if b.batch_id in log else "new"
            print(
                f"  {b.batch_id:<14} {b.total_bytes / 1e6:>6.1f} MB  "
                f"complete={str(b.trajectory_complete):<5} {state}"
            )
        return 0

    ingest(adapter, cfg, force=args.force, only=args.batches, chunk_rows=args.chunk_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
