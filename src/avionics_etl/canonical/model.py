"""The canonical data model — the shape every source normalises into.

Why this exists
---------------
C-MAPSS is a *surrogate*. The real telemetry schema from ISRO has not been shown
to us and will be different: different identifiers, different channel names,
different units, almost certainly a different notion of "cycle".

If the pipeline were written against C-MAPSS column names, swapping sources
later would mean rewriting every transform. Instead:

* **Source adapters** (``avionics_etl.adapters``) know about one source's file
  format and column names, and emit frames in the canonical shape below.
* **Everything downstream** — validation, regime assignment, normalisation,
  features, labels, warehouse — reads the canonical entity columns and the
  **channel registry**, never a hardcoded channel list.

Adding a new source is then one adapter file plus one registry block.

See docs/PRIMER.md section 10.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Iterable, Literal

import yaml

# ---------------------------------------------------------------------------
# Entity columns — present on every canonical table, whatever the source.
# ---------------------------------------------------------------------------

SOURCE_ID = "source_id"
"""Which data source this row came from. ``FD002`` today; ``TV-D1`` tomorrow."""

MISSION_ID = "mission_id"
"""Which mission or campaign. Required by the spec's "fed forward from previous
flights" clause — assessments must accumulate across missions."""

ASSET_ID = "asset_id"
"""The source's own identifier for the asset — its NATURAL key.

Natural keys collide. C-MAPSS numbers units from 1 in every file, so
``FD001-train`` unit 1 and ``FD001-test`` unit 1 are *different engines* that
share a number. Never group, split or join on this alone."""

ASSET_UID = "asset_uid"
"""Globally unique, stable asset identity. Group, split and join on THIS.

Adapters are responsible for constructing it. For C-MAPSS it is
``{source}-{split}-{unit:04d}``, because unit numbers are only unique within a
file. For real hardware it would be the serial number, which is genuinely unique
and — importantly — persists across missions, so the same flight computer
recovered twice is correctly recognised as one asset with a longer history."""

SEQUENCE_NO = "sequence_no"
"""Monotonic usage counter, starting at 1. A flight cycle today; a flight count
or power-on count tomorrow. This is the spine of every table.

WARNING: this column is algebraically part of the RUL label and must never be
used as a model feature. See docs/PRIMER.md section 8b."""

BATCH_ID = "batch_id"
"""Which ingest arrival produced this row. Ingestion is append-only, so a batch
is written once and never rewritten (docs/PRIMER.md section 11)."""

TRAJECTORY_COMPLETE = "trajectory_complete"
"""True when this asset's history runs all the way to failure.

This is the canonical form of C-MAPSS's train/test distinction, and it
generalises: a *recovered* avionics box has by definition NOT failed, so its
trajectory is incomplete. The flag drives which labelling path applies —
derived (complete) or externally supplied (incomplete). See PRIMER section 5."""

ENTITY_COLS = [SOURCE_ID, MISSION_ID, ASSET_UID, ASSET_ID, SEQUENCE_NO]

#: Columns that identify a row. None of these may ever enter a feature matrix.
IDENTIFIER_COLS = [*ENTITY_COLS, BATCH_ID, TRAJECTORY_COMPLETE]

#: The key that uniquely identifies one asset's trajectory.
ASSET_KEY = [SOURCE_ID, ASSET_UID]

#: The key that uniquely identifies one row.
ROW_KEY = [*ASSET_KEY, SEQUENCE_NO]

#: Columns appended by the lineage stamp.
LINEAGE_COLS = ["_run_id", "_config_hash", "_stage"]

#: Label columns produced downstream. Also never features.
LABEL_COLS = ["rul", "rul_raw", "max_sequence_no", "rul_true"]

#: Derived bookkeeping that must not leak into features either.
DERIVED_NON_FEATURE_COLS = ["regime_id", "fold", "n_outliers"]

PARTITION_COLS = [SOURCE_ID, BATCH_ID]
"""Bronze partitioning. NOT by asset: C-MAPSS has ~1,400 assets totalling 265k
rows, so per-asset directories would average ~190 rows each — the small-files
problem (PRIMER section 7). Sorting by (asset_id, sequence_no) within each
partition keeps row-group statistics tight, so per-asset predicate pushdown
still works. The waveform and image stores DO partition further, because there
each asset is large."""

SORT_COLS = [ASSET_UID, SEQUENCE_NO]

ChannelRole = Literal["sensor", "setting", "discrete", "derived"]


# ---------------------------------------------------------------------------
# Channel registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Channel:
    """One measured channel, as declared in ``configs/channels.yaml``."""

    name: str
    role: ChannelRole
    dtype: str = "float32"
    symbol: str | None = None
    quantity: str | None = None
    unit: str | None = None
    bounds: tuple[float, float] | None = None
    max_rate: float | None = None
    observed: tuple[float, float] | None = None

    @property
    def is_measurement(self) -> bool:
        """Sensors and discrete channels measure the asset's state.

        Settings describe the *operating condition*, not the asset, so they are
        handled separately — they define the regime rather than being normalised
        within it (PRIMER section 6).
        """
        return self.role in ("sensor", "discrete")


class ChannelRegistry:
    """The set of channels a source provides, loaded from configuration.

    Transforms iterate this instead of hardcoding ``s1..s21``. Swapping in a new
    source means swapping in a new registry, not editing transform code.
    """

    def __init__(self, channels: dict[str, Channel], *, source: str, version: int):
        self._channels = channels
        self.source = source
        self.version = version

    # -- construction -------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ChannelRegistry":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"channel registry not found: {path}")
        raw = yaml.safe_load(path.read_text())
        if not isinstance(raw, dict) or "channels" not in raw:
            raise ValueError(f"{path} has no 'channels:' block")

        channels: dict[str, Channel] = {}
        for name, spec in raw["channels"].items():
            spec = dict(spec or {})
            bounds = spec.pop("bounds", None)
            observed = spec.pop("observed", None)
            channels[name] = Channel(
                name=name,
                bounds=tuple(bounds) if bounds else None,
                observed=tuple(observed) if observed else None,
                **{k: v for k, v in spec.items() if k in Channel.__annotations__},
            )
        return cls(channels, source=raw.get("source", "unknown"), version=raw.get("version", 1))

    # -- access -------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._channels)

    def __contains__(self, name: object) -> bool:
        return name in self._channels

    def __getitem__(self, name: str) -> Channel:
        try:
            return self._channels[name]
        except KeyError:
            raise KeyError(
                f"channel {name!r} is not in the registry. Known channels: "
                f"{sorted(self._channels)}"
            ) from None

    def __iter__(self):
        return iter(self._channels.values())

    def names(self, role: ChannelRole | None = None) -> list[str]:
        """Channel names, optionally filtered by role. Registry order is stable."""
        return [c.name for c in self._channels.values() if role is None or c.role == role]

    @cached_property
    def sensors(self) -> list[str]:
        return self.names("sensor")

    @cached_property
    def settings(self) -> list[str]:
        """Operating-condition channels. These define the regime (PRIMER §6)."""
        return self.names("setting")

    @cached_property
    def measurements(self) -> list[str]:
        """Channels that measure the asset's state, i.e. candidate features."""
        return [c.name for c in self._channels.values() if c.is_measurement]

    @property
    def dtypes(self) -> dict[str, str]:
        return {c.name: c.dtype for c in self._channels.values()}


# ---------------------------------------------------------------------------
# Feature-column policy — enforced in one place, used by the handoff and tests
# ---------------------------------------------------------------------------

NON_FEATURE_COLS = frozenset(
    IDENTIFIER_COLS + LINEAGE_COLS + LABEL_COLS + DERIVED_NON_FEATURE_COLS
)


def feature_columns(columns: Iterable[str]) -> list[str]:
    """Filter a column list down to those legally usable as model features.

    Removes identifiers, lineage stamps, labels and derived bookkeeping. Anything
    prefixed with ``_`` is also excluded, so new internal columns are safe by
    default rather than dangerous by default.

    This function is the single definition of "is this a feature?" — the ETL→ML
    handoff and the leakage test suite both use it, so they cannot disagree.
    """
    return [
        c for c in columns
        if c not in NON_FEATURE_COLS and not c.startswith("_")
    ]


def assert_canonical(columns: Iterable[str]) -> None:
    """Raise if a frame is missing required canonical entity columns."""
    cols = set(columns)
    missing = [c for c in ENTITY_COLS if c not in cols]
    if missing:
        raise ValueError(
            f"frame is not in canonical form; missing entity columns: {missing}. "
            f"Source adapters must emit {ENTITY_COLS}."
        )
