"""Feature extraction.

Turns per-row sensor readings into something a model can learn from, using only
information available at prediction time.

Three families, all built from the channel registry rather than a hardcoded
column list, so a new data source gets features automatically:

**Level** — the regime-normalised value itself. How far from normal-for-this-
operating-condition is this reading, right now.

**Trend** — causal rolling statistics over recent history: mean, standard
deviation, min, max, and slope. Degradation is smooth and sensor noise is not,
so a rolling window suppresses the noise and exposes the trend. This is where
most of the signal lives.

**Change** — the step since the previous observation, and since ``w`` steps back.

Every window is backward-looking and grouped by asset, via
:func:`avionics_etl.transform.causal.causal_rolling`. Nothing here calls
``.rolling()`` directly, and nothing may.

Feature count grows fast: ``len(channels) x len(windows) x len(stats)``. With 21
sensors, three windows and five statistics that is 315 columns before the level
features. That is fine for tree models and a lot for a small dataset, which is
why :mod:`avionics_etl.transform.reduce` offers dimensionality reduction on top.

See docs/PRIMER.md sections 3 and 8d.
"""

from __future__ import annotations

import pandas as pd

from avionics_etl.canonical import model
from avionics_etl.canonical.model import ChannelRegistry
from avionics_etl.transform import normalize
from avionics_etl.transform.causal import causal_rolling


def feature_source_columns(
    df: pd.DataFrame, registry: ChannelRegistry, *, use_normalised: bool = True
) -> list[str]:
    """Which columns features are built from.

    Defaults to the regime-normalised columns. Building trend features on raw
    values would carry the operating-condition effect straight into every
    window, which is the problem normalisation exists to remove.

    Channels that are constant everywhere are skipped — a rolling mean of a
    constant is a constant, and 21 useless columns cost real time and space.
    """
    out: list[str] = []
    for channel in registry.measurements:
        col = f"{channel}{normalize.SUFFIX}" if use_normalised else channel
        if col not in df.columns:
            continue
        if df[col].nunique(dropna=True) <= 1:
            continue
        out.append(col)
    return out


def build(
    df: pd.DataFrame,
    registry: ChannelRegistry,
    *,
    windows: list[int],
    stats: list[str],
    group_cols: list[str] | None = None,
    use_normalised: bool = True,
) -> pd.DataFrame:
    """Build the full feature block. Returns new columns only."""
    group_cols = group_cols or list(model.ASSET_KEY)
    sources = feature_source_columns(df, registry, use_normalised=use_normalised)
    if not sources:
        raise ValueError("no usable channels to build features from")

    blocks: list[pd.DataFrame] = [df[sources].copy()]
    for window in windows:
        for stat in stats:
            blocks.append(
                causal_rolling(
                    df, sources, window=window, stat=stat, group_cols=group_cols
                )
            )
    return pd.concat(blocks, axis=1)


def describe(features: pd.DataFrame) -> pd.DataFrame:
    """Per-feature summary, for the manifest and for spotting dead columns."""
    desc = features.describe().T
    desc["n_unique"] = features.nunique()
    desc["constant"] = desc.n_unique <= 1
    desc["null_fraction"] = features.isna().mean()
    return desc.reset_index(names="feature")
