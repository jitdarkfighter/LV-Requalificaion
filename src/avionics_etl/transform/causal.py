"""Causal feature helpers — the only sanctioned way to compute a window.

A **causal** feature uses only past and present values, never future ones. The
term comes from signal processing: a causal filter cannot see ahead.

This matters because the model will be asked to predict from a *truncated*
history — a recovered box has not failed, so its trajectory simply stops. Any
feature that needs the end of the series, or peeks forward, exists at training
time and not at prediction time. That is data leakage, and it produces excellent
validation numbers and a worthless model.

Two rules, both enforced here rather than remembered:

1. **Backward-looking only.** Never ``center=True``.
2. **Grouped by asset.** An ungrouped window slides off the end of one
   trajectory and onto the start of an unrelated one.

Nothing in this repository calls ``.rolling()`` directly. It calls
:func:`causal_rolling`.

See docs/PRIMER.md section 8d.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

STATS = ("mean", "std", "min", "max", "slope", "delta")


def _slope_weights(window: int) -> tuple[np.ndarray, float]:
    """Least-squares slope over evenly spaced points, as a fixed convolution.

    For x = 0..w-1 the ordinary least squares slope reduces to a weighted sum of
    the y values, with weights (i - mean(x)) and a constant denominator. That
    turns a per-window regression into one dot product.
    """
    w = np.arange(window, dtype="float64")
    w -= w.mean()
    return w, float((w**2).sum())


def causal_rolling(
    df: pd.DataFrame,
    columns: Iterable[str],
    *,
    window: int,
    stat: str,
    group_cols: Sequence[str],
    min_periods: int | None = None,
    suffix: str | None = None,
) -> pd.DataFrame:
    """Backward-looking rolling statistic, computed within each asset.

    Returns a frame of new columns named ``{column}_{stat}{window}``, aligned to
    ``df``'s index. The caller concatenates them; this function never mutates the
    input.

    ``min_periods`` defaults to 1 so early rows get a partial-window value rather
    than a null — except for ``slope``, which needs a full window to mean
    anything and yields 0.0 (no trend established) before that.
    """
    if stat not in STATS:
        raise ValueError(f"unknown stat {stat!r}; expected one of {STATS}")
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    columns = list(columns)
    missing = [c for c in [*group_cols, *columns] if c not in df.columns]
    if missing:
        raise KeyError(f"columns not present: {missing}")

    suffix = suffix if suffix is not None else f"_{stat}{window}"
    grouped = df.groupby(list(group_cols), sort=False)
    out: dict[str, pd.Series] = {}

    if stat == "delta":
        # Change since the previous step. Undefined at the start of a
        # trajectory, where 0.0 (no change observed) is the honest value.
        for col in columns:
            out[f"{col}{suffix}"] = grouped[col].diff(window).fillna(0.0)
        return pd.DataFrame(out, index=df.index)

    if stat == "slope":
        # A rolling least-squares slope is a fixed-weight sum over the window, so
        # it is also a fixed-weight sum over W lagged copies of the series. That
        # turns ~15M per-window Python callbacks into ~W vectorised shifts.
        weights, denom = _slope_weights(window)
        position = grouped.cumcount()
        full_window = position >= (window - 1)

        for col in columns:
            lagged = grouped[col]
            total = None
            for k in range(window):
                term = lagged.shift(k) * weights[window - 1 - k]
                total = term if total is None else total + term
            slope = (total / denom).where(full_window, 0.0)
            out[f"{col}{suffix}"] = slope.fillna(0.0)
        return pd.DataFrame(out, index=df.index)

    mp = min_periods if min_periods is not None else 1
    for col in columns:
        out[f"{col}{suffix}"] = grouped[col].transform(
            lambda s: getattr(s.rolling(window, min_periods=mp), stat)()
        )
    frame = pd.DataFrame(out, index=df.index)
    if stat == "std":
        # A single observation has no spread; 0.0 is correct, NaN is not.
        frame = frame.fillna(0.0)
    return frame


def is_causal_safe(name: str) -> bool:
    """Whether a column name looks like it was produced causally.

    A crude guard, used by the leakage tests to catch obviously forward-looking
    names that slipped in by hand.
    """
    forbidden = ("future", "lead", "centered", "center", "_max_cycle", "total_", "final_")
    lowered = name.lower()
    return not any(token in lowered for token in forbidden)
