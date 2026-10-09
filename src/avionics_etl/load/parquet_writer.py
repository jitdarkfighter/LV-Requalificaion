"""Partitioned Parquet writing.

Centralises the storage-layout decisions argued in docs/PRIMER.md section 7:
Hive-style partition directories, sorting within each partition so row-group
statistics stay tight, and row groups sized so predicate pushdown has useful
granularity without memory spikes.

Writes are atomic per partition: a partition directory is replaced wholesale
rather than appended to, which is what makes stages idempotent (PRIMER §9).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_ROW_GROUP_BYTES = 128 * 1024 * 1024
MIN_ROW_GROUP_ROWS = 4_096


def _row_group_rows(table: pa.Table, target_bytes: int) -> int:
    """Rows per row group that approximates ``target_bytes`` uncompressed."""
    if table.num_rows == 0:
        return MIN_ROW_GROUP_ROWS
    bytes_per_row = max(1, table.nbytes // table.num_rows)
    return max(MIN_ROW_GROUP_ROWS, target_bytes // bytes_per_row)


def _partition_dir(root: Path, keys: dict[str, object]) -> Path:
    """Hive-style path: ``root/col=value/col2=value2``."""
    path = root
    for col, value in keys.items():
        path = path / f"{col}={value}"
    return path


def write_partitioned(
    df: pd.DataFrame,
    root: Path,
    *,
    partition_by: Iterable[str],
    sort_by: Iterable[str] = (),
    compression: str = "zstd",
    row_group_bytes: int = DEFAULT_ROW_GROUP_BYTES,
    schema: pa.Schema | None = None,
) -> dict[str, int]:
    """Write ``df`` as Hive-partitioned Parquet, replacing existing partitions.

    Partition columns are encoded in the directory path and dropped from the
    file body — that is what lets a query skip whole directories without opening
    a single file.

    Returns a mapping of partition path (relative to ``root``) to row count.
    """
    partition_by = list(partition_by)
    sort_by = list(sort_by)
    root.mkdir(parents=True, exist_ok=True)

    missing = [c for c in partition_by + sort_by if c not in df.columns]
    if missing:
        raise KeyError(f"columns not present in frame: {missing}")

    written: dict[str, int] = {}

    groups = [((), df)] if not partition_by else df.groupby(partition_by, sort=True)
    for keys, part in groups:
        if partition_by:
            keys = keys if isinstance(keys, tuple) else (keys,)
            key_map = dict(zip(partition_by, keys))
            body = part.drop(columns=partition_by)
        else:
            key_map = {}
            body = part

        if sort_by:
            body = body.sort_values(sort_by, kind="stable")
        body = body.reset_index(drop=True)

        out_dir = _partition_dir(root, key_map)
        if out_dir.exists():
            shutil.rmtree(out_dir)  # replace wholesale => idempotent
        out_dir.mkdir(parents=True, exist_ok=True)

        part_schema = None
        if schema is not None:
            keep = [f for f in schema if f.name in body.columns]
            part_schema = pa.schema(keep)

        table = pa.Table.from_pandas(body, schema=part_schema, preserve_index=False)
        pq.write_table(
            table,
            out_dir / "part-0.parquet",
            compression=compression,
            row_group_size=_row_group_rows(table, row_group_bytes),
            use_dictionary=True,
            write_statistics=True,
        )
        written[str(out_dir.relative_to(root))] = len(body)

    return written


class StreamingPartitionWriter:
    """Write one partition incrementally, as a sequence of row groups.

    :func:`write_partitioned` needs the whole frame in memory. This does not:
    chunks are appended as row groups to a single open file, so peak memory
    stays proportional to one chunk however large the batch is. That is what
    makes a 64 GB arrival ingestible on one machine (docs/PRIMER.md section 11).

    The trade-off: row groups are written in arrival order, so **the adapter must
    emit chunks already ordered by the sort key**. :func:`verify_sorted` checks
    this after the fact rather than trusting it.

    Used as a context manager::

        with StreamingPartitionWriter(root, {"source_id": "FD001"}) as w:
            for chunk in adapter.read_chunks(batch):
                w.write(chunk)
    """

    def __init__(
        self,
        root: Path,
        partition_keys: dict[str, object],
        *,
        compression: str = "zstd",
        row_group_bytes: int = DEFAULT_ROW_GROUP_BYTES,
    ) -> None:
        self.root = root
        self.partition_keys = partition_keys
        self.compression = compression
        self.row_group_bytes = row_group_bytes
        self.rows = 0
        self._writer: pq.ParquetWriter | None = None
        self._dir = _partition_dir(root, partition_keys)
        self._path = self._dir / "part-0.parquet"
        self._buffer: list[pd.DataFrame] = []
        self._buffered_rows = 0
        self._target_rows: int | None = None

    def __enter__(self) -> "StreamingPartitionWriter":
        if self._dir.exists():
            shutil.rmtree(self._dir)  # replace wholesale => idempotent
        self._dir.mkdir(parents=True, exist_ok=True)
        return self

    def write(self, df: pd.DataFrame) -> None:
        """Append a chunk. Partition-key columns are dropped from the body."""
        body = df.drop(columns=[c for c in self.partition_keys if c in df.columns])
        if body.empty:
            return
        table = pa.Table.from_pandas(body, preserve_index=False)

        if self._writer is None:
            self._writer = pq.ParquetWriter(
                self._path,
                table.schema,
                compression=self.compression,
                use_dictionary=True,
                write_statistics=True,
            )
            self._target_rows = _row_group_rows(table, self.row_group_bytes)

        # Accumulate small chunks so row groups reach a useful size rather than
        # inheriting the reader's chunk size.
        self._buffer.append(body)
        self._buffered_rows += len(body)
        if self._buffered_rows >= (self._target_rows or MIN_ROW_GROUP_ROWS):
            self._flush()
        self.rows += len(body)

    def _flush(self) -> None:
        if not self._buffer or self._writer is None:
            return
        merged = pd.concat(self._buffer, ignore_index=True) if len(self._buffer) > 1 else self._buffer[0]
        self._writer.write_table(pa.Table.from_pandas(merged, preserve_index=False))
        self._buffer.clear()
        self._buffered_rows = 0

    def __exit__(self, *exc: object) -> None:
        self._flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    @property
    def path(self) -> Path:
        return self._path


def verify_sorted(path: Path, sort_by: list[str]) -> bool:
    """Check that a written Parquet file is ordered by ``sort_by``.

    Sorting matters because row-group min/max statistics are only useful when
    ranges are tight and non-overlapping (docs/PRIMER.md section 7). We verify
    rather than assume, reading only the key columns.
    """
    table = pq.read_table(path, columns=sort_by)
    df = table.to_pandas()
    if df.empty:
        return True
    return bool(df.equals(df.sort_values(sort_by, kind="stable").reset_index(drop=True)))


def read_partitioned(root: Path, **filters: object) -> pd.DataFrame:
    """Read a partitioned dataset back, optionally filtering on partition keys.

    Partition columns are restored from the directory names by pyarrow.
    """
    import pyarrow.dataset as ds

    if not root.exists():
        raise FileNotFoundError(f"no dataset at {root}")

    dataset = ds.dataset(root, format="parquet", partitioning="hive")
    expr = None
    for col, value in filters.items():
        clause = ds.field(col) == value
        expr = clause if expr is None else (expr & clause)
    return dataset.to_table(filter=expr).to_pandas()
