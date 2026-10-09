"""Regime-wise normalisation — the highest-leverage transform in the pipeline.

Having assigned a regime to each row (:mod:`avionics_etl.transform.regime`), we
z-score each channel **within its own regime**:

    z = (value - mean_of_this_regime) / std_of_this_regime

What remains is deviation from *normal for this operating condition*, which is
degradation. Measured on C-MAPSS FD002, this takes sensor/label correlations
from 0.001-0.056 (indistinguishable from noise) to 0.32-0.77.

Two disciplines make it correct rather than merely effective:

**Fitted on training data only.** Statistics computed over the whole dataset
encode the evaluation set's distribution — that is leakage (docs/PRIMER.md §8c).
Parameters are fitted on the training fold, persisted, and applied read-only
everywhere else.

**Persisted, never refitted.** The fitted table is written to
``data/artifacts/`` and is the single source of truth. If the ML side refits on
its own data, the two disagree and nobody finds out for a week.

Settings channels are deliberately NOT normalised: they *define* the regime
rather than being measured within it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from avionics_etl.canonical import model
from avionics_etl.canonical.model import ChannelRegistry
from avionics_etl.transform.regime import REGIME_COL

PARAMS_FILE = "normalization_params.parquet"
MINMAX_PARAMS_FILE = "minmax_params.parquet"
SUFFIX = "_z"
MINMAX_SUFFIX = "_mm"

#: Scaling methods. They answer different questions and are not interchangeable.
#:
#: ``zscore``  deviation in standard deviations. Unbounded, so how *unusual* a
#:             value is survives the transform — which is what RUL regression
#:             and outlier detection want.
#: ``minmax``  rescaled to [0, 1]. Bounded, so relative position within the
#:             operating range is preserved and absolute magnitudes stay
#:             comparable across channels. The launch-vehicle anomaly-detection
#:             literature uses this, because spatial relationships between
#:             parameters depend on absolute level, and neural networks train
#:             more stably on bounded inputs.
METHODS = ("zscore", "minmax")

#: Below this many observations a regime's statistics are not trustworthy.
MIN_GROUP_SIZE = 30

GROUP_COLS = [model.SOURCE_ID, REGIME_COL]


def fit(
    df: pd.DataFrame,
    registry: ChannelRegistry,
    *,
    group_cols: list[str] | None = None,
    min_group_size: int = MIN_GROUP_SIZE,
) -> pd.DataFrame:
    """Compute per-(source, regime, channel) mean and standard deviation.

    ``df`` must already be restricted to training rows — this function does not
    filter, so that the caller's intent is explicit and visible.

    A zero or near-zero standard deviation (a channel that is constant within a
    regime) is replaced by 1.0, which maps every value to 0.0 rather than to
    infinity. Constant channels carry no information; they should not be able to
    produce one.
    """
    group_cols = group_cols or GROUP_COLS
    channels = [c for c in registry.measurements if c in df.columns]
    if not channels:
        raise ValueError("no measurement channels present to normalise")

    grouped = df.groupby(group_cols, sort=True)[channels]
    means = grouped.mean().stack().rename("mean")
    stds = grouped.std(ddof=0).stack().rename("std")
    counts = grouped.count().stack().rename("n")

    params = pd.concat([means, stds, counts], axis=1).reset_index()
    params = params.rename(columns={params.columns[len(group_cols)]: "channel"})

    params["degenerate"] = params["std"].isna() | (params["std"] < 1e-9)
    params["std"] = params["std"].where(~params.degenerate, 1.0)
    params["undersampled"] = params.n < min_group_size

    return params[[*group_cols, "channel", "mean", "std", "n", "degenerate", "undersampled"]]


def apply(
    df: pd.DataFrame,
    params: pd.DataFrame,
    *,
    group_cols: list[str] | None = None,
    suffix: str = SUFFIX,
) -> pd.DataFrame:
    """Apply fitted parameters, returning the normalised columns only.

    Rows whose (source, regime) combination was never seen during fitting are
    left as NaN rather than silently normalised against the wrong baseline. That
    is a real condition worth surfacing — it means an operating regime appeared
    that training never covered.
    """
    group_cols = group_cols or GROUP_COLS
    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        raise KeyError(f"cannot normalise: missing grouping columns {missing}")

    out: dict[str, pd.Series] = {}
    key = df[group_cols].astype(str).agg("|".join, axis=1)

    for channel, spec in params.groupby("channel", sort=False):
        if channel not in df.columns:
            continue
        lookup = spec.assign(_key=spec[group_cols].astype(str).agg("|".join, axis=1))
        mean = key.map(dict(zip(lookup._key, lookup["mean"])))
        std = key.map(dict(zip(lookup._key, lookup["std"])))
        out[f"{channel}{suffix}"] = ((df[channel] - mean) / std).astype("float32")

    return pd.DataFrame(out, index=df.index)


def unseen_groups(df: pd.DataFrame, params: pd.DataFrame, *, group_cols: list[str] | None = None) -> pd.DataFrame:
    """Regimes present in the data but absent from the fitted parameters."""
    group_cols = group_cols or GROUP_COLS
    have = set(map(tuple, params[group_cols].drop_duplicates().to_numpy()))
    seen = df[group_cols].drop_duplicates()
    mask = [tuple(row) not in have for row in seen.to_numpy()]
    return seen[mask]


def save(params: pd.DataFrame, artifacts_dir: Path) -> Path:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = artifacts_dir / PARAMS_FILE
    params.to_parquet(path, index=False)
    return path


def load(artifacts_dir: Path) -> pd.DataFrame:
    path = artifacts_dir / PARAMS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"normalisation parameters not found at {path}. Run `make silver` first — "
            "they are fitted once and applied read-only thereafter."
        )
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# min-max scaling
# ---------------------------------------------------------------------------

def fit_minmax(
    df: pd.DataFrame,
    registry: ChannelRegistry,
    *,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Per-(group, channel) minimum and maximum.

    As with the z-score path, ``df`` must already be restricted to training
    rows; this function does not filter on the caller's behalf.

    A channel that is constant within a group has no range. Its span is set to
    1.0, mapping every value to 0.0 — a no-information channel should not be
    able to produce infinity.
    """
    group_cols = group_cols or GROUP_COLS
    channels = [c for c in registry.measurements if c in df.columns]
    if not channels:
        raise ValueError("no measurement channels present to normalise")

    grouped = df.groupby(group_cols, sort=True)[channels]
    lo = grouped.min().stack().rename("lo")
    hi = grouped.max().stack().rename("hi")
    n = grouped.count().stack().rename("n")

    params = pd.concat([lo, hi, n], axis=1).reset_index()
    params = params.rename(columns={params.columns[len(group_cols)]: "channel"})

    params["span"] = params.hi - params.lo
    params["degenerate"] = params.span.isna() | (params.span < 1e-9)
    params["span"] = params["span"].where(~params.degenerate, 1.0)
    return params[[*group_cols, "channel", "lo", "hi", "span", "n", "degenerate"]]


def apply_minmax(
    df: pd.DataFrame,
    params: pd.DataFrame,
    *,
    group_cols: list[str] | None = None,
    suffix: str = MINMAX_SUFFIX,
    clip: bool = True,
) -> pd.DataFrame:
    """Apply fitted min/max, returning the scaled columns only.

    ``clip`` bounds the output to [0, 1]. Held-out data legitimately contains
    values outside the training range, and for a bounded scaling the honest
    options are to clip or to let the output leave its nominal range. Clipping
    is the default because the downstream consumers of this scaling — neural
    networks — assume bounded inputs. The z-score path is the one to use when
    how far outside the range a value fell is the thing you care about.
    """
    group_cols = group_cols or GROUP_COLS
    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        raise KeyError(f"cannot normalise: missing grouping columns {missing}")

    out: dict[str, pd.Series] = {}
    key = df[group_cols].astype(str).agg("|".join, axis=1)

    for channel, spec in params.groupby("channel", sort=False):
        if channel not in df.columns:
            continue
        lookup = spec.assign(_key=spec[group_cols].astype(str).agg("|".join, axis=1))
        lo = key.map(dict(zip(lookup._key, lookup["lo"])))
        span = key.map(dict(zip(lookup._key, lookup["span"])))
        scaled = (df[channel] - lo) / span
        if clip:
            scaled = scaled.clip(lower=0.0, upper=1.0)
        out[f"{channel}{suffix}"] = scaled.astype("float32")

    return pd.DataFrame(out, index=df.index)


def save_minmax(params: pd.DataFrame, artifacts_dir: Path) -> Path:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = artifacts_dir / MINMAX_PARAMS_FILE
    params.to_parquet(path, index=False)
    return path


def load_minmax(artifacts_dir: Path) -> pd.DataFrame:
    path = artifacts_dir / MINMAX_PARAMS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"min-max parameters not found at {path}. Fit them first — they are "
            "fitted once on training data and applied read-only thereafter."
        )
    return pd.read_parquet(path)
