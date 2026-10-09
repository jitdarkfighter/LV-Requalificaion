"""The source adapter interface.

An adapter is the *only* place that knows how one data source is laid out. It
answers three questions:

1. **What is available to ingest?**  :meth:`SourceAdapter.discover` returns
   :class:`Batch` records — one per arrival unit.
2. **How do I read one batch without running out of memory?**
   :meth:`SourceAdapter.read_chunks` yields bounded-size frames already in
   canonical form.
3. **What channels does this source provide?**  :meth:`SourceAdapter.registry`.

Everything downstream is source-agnostic. When the real ISRO schema arrives,
adding it means writing one new subclass here plus one channel registry — no
transform, test or warehouse code changes.

See ``adapters/README.md`` for a step-by-step guide, and docs/PRIMER.md
sections 10 and 11.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import pandas as pd

from avionics_etl.canonical.model import ChannelRegistry

DEFAULT_CHUNK_ROWS = 500_000


@dataclass(frozen=True)
class Batch:
    """One unit of arrival.

    Ingestion is append-only and tracked per batch, so a batch is written once
    and never rewritten. Re-running ingest skips batches already recorded in the
    watermark log unless explicitly forced.

    For C-MAPSS a batch is one source file. For real mission data a batch would
    be one recovery event, or one telemetry dump.
    """

    batch_id: str
    source_id: str
    mission_id: str
    files: tuple[Path, ...]
    trajectory_complete: bool
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def total_bytes(self) -> int:
        return sum(f.stat().st_size for f in self.files if f.exists())


class SourceAdapter(abc.ABC):
    """Base class for all data sources."""

    #: Short identifier, used in logs and the watermark record.
    name: str = "unnamed"

    @abc.abstractmethod
    def discover(self) -> list[Batch]:
        """List everything this source currently offers for ingestion."""

    @abc.abstractmethod
    def read_chunks(
        self, batch: Batch, *, chunk_rows: int = DEFAULT_CHUNK_ROWS
    ) -> Iterator[pd.DataFrame]:
        """Yield the batch as canonical frames of at most ``chunk_rows`` rows.

        This is a generator, not a return of one frame, and that is the whole
        point: peak memory stays proportional to ``chunk_rows`` regardless of
        whether the batch is 40 MB or 64 GB.

        Each yielded frame must carry the canonical entity columns
        (``source_id``, ``mission_id``, ``asset_id``, ``sequence_no``) plus the
        channels declared by :meth:`registry`.
        """

    @abc.abstractmethod
    def registry(self) -> ChannelRegistry:
        """The channels this source provides."""

    # -- helpers ------------------------------------------------------------

    def describe(self) -> str:
        batches = self.discover()
        total = sum(b.total_bytes for b in batches)
        return f"{self.name}: {len(batches)} batches, {total / 1e6:,.1f} MB"
