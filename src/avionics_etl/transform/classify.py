"""Classify parameters as static, dynamic or redundant.

This is the step that makes high-dimensional telemetry tractable. A launch
vehicle test touches 3,500 parameters, but any individual test only *affects*
10–200 of them; the rest are supposed to sit still at their ambient values.
Running a deep model over all 3,500 wastes compute on channels that, by design,
contain nothing to learn.

So each channel is routed by what it actually does during the segment:

``static``
    Barely moves. Verified by a limit check — does it stay near its ambient
    value? A static channel that drifts is itself an anomaly (a failed sensor,
    or an isolation failure where operating one subsystem disturbs another).

``dynamic``
    Genuinely varies. These go to the forecasting models.

``redundant``
    A declared backup for another channel. Verified by comparing against its
    prime: if the two disagree by more than a tolerance, one of them is wrong.
    This catches faults no single-channel check can see, because each reading
    is individually plausible.

**Scale matters when deciding what "barely moves" means.** Raw variance is not
comparable across channels — a pressure in psia and a temperature in Rankine
have variances in different units entirely. Variation is therefore measured
relative to the channel's declared operating range from the registry, falling
back to a coefficient of variation when no bounds are declared.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from avionics_etl.canonical.model import ChannelRegistry

Kind = Literal["static", "dynamic", "redundant"]

#: Default: a channel moving less than 1% of its operating range is static.
DEFAULT_VARIANCE_THRESHOLD = 0.01

CLASSIFICATION_COLUMNS = [
    "channel",
    "kind",
    "std",
    "relative_std",
    "mean",
    "ambient",
    "n",
    "scale_basis",
]


@dataclass(frozen=True)
class RedundancyPair:
    """A prime channel and its declared backup.

    Redundancy is a property of the hardware, declared in the measurement plan.
    It cannot be inferred from the data — two channels correlating highly might
    be redundant, or might simply both respond to the same physical event.
    """

    prime: str
    backup: str
    tolerance: float
    unit: str | None = None


def _scale_of(channel: str, registry: ChannelRegistry | None, values: pd.Series) -> tuple[float, str]:
    """Reference scale for judging how much a channel moved.

    Prefers the declared operating range, because that is a statement about the
    hardware rather than about this particular sample.
    """
    if registry is not None and channel in registry:
        bounds = registry[channel].bounds
        if bounds is not None and bounds[1] > bounds[0]:
            return float(bounds[1] - bounds[0]), "declared_bounds"
    mean = float(np.abs(values.mean()))
    if mean > 1e-12:
        return mean, "coefficient_of_variation"
    return 1.0, "absolute"


def classify(
    df: pd.DataFrame,
    channels: Sequence[str],
    *,
    registry: ChannelRegistry | None = None,
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
    redundancy: Sequence[RedundancyPair] = (),
) -> pd.DataFrame:
    """Classify each channel over one segment of data.

    Returns one row per channel. Redundancy is declared, so a channel named as
    a backup is labelled ``redundant`` regardless of how much it moves — it is
    checked against its prime rather than modelled.
    """
    backups = {p.backup for p in redundancy}
    rows = []
    for name in channels:
        if name not in df.columns:
            continue
        v = df[name].astype("float64")
        scale, basis = _scale_of(name, registry, v)
        std = float(v.std(ddof=0))
        rel = std / scale if scale > 0 else 0.0

        if name in backups:
            kind: Kind = "redundant"
        elif rel < variance_threshold:
            kind = "static"
        else:
            kind = "dynamic"

        rows.append(
            {
                "channel": name,
                "kind": kind,
                "std": std,
                "relative_std": rel,
                "mean": float(v.mean()),
                # Ambient is the value a static channel is supposed to hold.
                # The median is used rather than the mean so a brief excursion
                # does not drag the reference toward itself.
                "ambient": float(v.median()),
                "n": int(v.notna().sum()),
                "scale_basis": basis,
            }
        )
    return pd.DataFrame(rows, columns=CLASSIFICATION_COLUMNS)


def check_static(
    df: pd.DataFrame,
    classification: pd.DataFrame,
    *,
    registry: ChannelRegistry | None = None,
    tolerance: float = 0.05,
) -> pd.DataFrame:
    """Limit check for static channels: did any stray from its ambient value?

    ``tolerance`` is a fraction of the channel's reference scale, so one setting
    applies across channels in different units.
    """
    static = classification[classification.kind == "static"]
    hits = []
    for _, row in static.iterrows():
        name = row.channel
        if name not in df.columns:
            continue
        v = df[name].astype("float64")
        scale, _ = _scale_of(name, registry, v)
        deviation = (v - row.ambient).abs()
        limit = tolerance * scale
        bad = deviation > limit
        if not bad.any():
            continue
        idx = df.index[bad]
        hits.append(
            pd.DataFrame(
                {
                    "channel": name,
                    "rule": "static_drift",
                    "value": v.loc[idx],
                    "ambient": row.ambient,
                    "deviation": deviation.loc[idx],
                    "limit": limit,
                    "detail": (
                        f"{name} is a static parameter but moved more than "
                        f"{tolerance:.0%} of its operating range from ambient"
                    ),
                },
                index=idx,
            )
        )
    if not hits:
        return pd.DataFrame(
            columns=["channel", "rule", "value", "ambient", "deviation", "limit", "detail"]
        )
    return pd.concat(hits).sort_index()


def check_redundancy(df: pd.DataFrame, pairs: Sequence[RedundancyPair]) -> pd.DataFrame:
    """Compare each redundant channel against its prime.

    Catches faults invisible to any single-channel check: both readings can sit
    comfortably inside their own limits while disagreeing with each other, which
    means at least one of them is lying.
    """
    hits = []
    for pair in pairs:
        if pair.prime not in df.columns or pair.backup not in df.columns:
            continue
        d = (df[pair.prime].astype("float64") - df[pair.backup].astype("float64")).abs()
        bad = d > pair.tolerance
        if not bad.any():
            continue
        idx = df.index[bad]
        hits.append(
            pd.DataFrame(
                {
                    "channel": pair.backup,
                    "prime": pair.prime,
                    "rule": "redundancy_disagreement",
                    "divergence": d.loc[idx],
                    "tolerance": pair.tolerance,
                    "detail": (
                        f"{pair.prime} and {pair.backup} disagree by more than "
                        f"{pair.tolerance:g}{' ' + pair.unit if pair.unit else ''}"
                    ),
                },
                index=idx,
            )
        )
    if not hits:
        return pd.DataFrame(
            columns=["channel", "prime", "rule", "divergence", "tolerance", "detail"]
        )
    return pd.concat(hits).sort_index()


def summarise(classification: pd.DataFrame) -> pd.Series:
    """Counts per kind — how much the classification actually saved."""
    return classification.kind.value_counts().reindex(
        ["static", "dynamic", "redundant"], fill_value=0
    )
