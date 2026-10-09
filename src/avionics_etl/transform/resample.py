"""Time normalisation and cross-mission synchronisation.

Launch vehicle parameters are sampled at different rates — the paper notes
telemetry formats change mid-flight to use the available bandwidth, so the same
parameter can arrive at several frequencies across a test. Anything that reasons
over multiple channels at once needs them on a common time base first.

Two operations, and the distinction between them matters:

**Resampling to a common rate.** How you interpolate depends on what the signal
*is*:

* *Analog* channels (temperatures, pressures, currents) vary continuously, so a
  value between two samples is well approximated by a straight line. Linear
  interpolation.
* *Discrete* channels (status words, relay states, health bytes) do not vary
  between samples — they hold their last value until something changes them.
  Interpolating those linearly would invent states that never existed, e.g. a
  relay "half open". Zero-order hold instead.

Getting this backwards is a silent corruption: the numbers look plausible and
are wrong.

**Mission synchronisation.** Each mission's data is stamped with its own
absolute wall-clock time. Comparing missions on absolute time would align
nothing. Re-basing each mission so it starts at zero puts them on a common
elapsed-time axis.

See docs/PRIMER.md section 11 for where this sits in the pipeline.
"""

from __future__ import annotations

from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd

Interp = Literal["linear", "zoh"]

#: Channel roles that hold their value between samples rather than varying
#: continuously. Everything else is treated as analog.
DISCRETE_ROLES = ("discrete",)


def interpolation_for(role: str) -> Interp:
    """Which interpolation a channel role requires."""
    return "zoh" if role in DISCRETE_ROLES else "linear"


def synchronise_missions(
    df: pd.DataFrame,
    *,
    time_col: str,
    group_cols: Sequence[str],
) -> pd.Series:
    """Re-base each group's time axis so it starts at zero.

    ``t' = T - T_0`` within each group. Returns the elapsed-time column; the
    caller decides whether to replace the original or keep both, because the
    absolute timestamp is still wanted for traceability.
    """
    missing = [c for c in [time_col, *group_cols] if c not in df.columns]
    if missing:
        raise KeyError(f"columns not present: {missing}")
    origin = df.groupby(list(group_cols), sort=False)[time_col].transform("min")
    return (df[time_col] - origin).rename("elapsed")


def _interp_channel(
    t_src: np.ndarray, v_src: np.ndarray, t_dst: np.ndarray, how: Interp
) -> np.ndarray:
    if len(t_src) == 0:
        return np.full(len(t_dst), np.nan)
    if len(t_src) == 1:
        return np.full(len(t_dst), v_src[0])

    if how == "linear":
        return np.interp(t_dst, t_src, v_src)

    # Zero-order hold: take the most recent sample at or before each target
    # time. searchsorted gives the insertion point, so index-1 is that sample.
    idx = np.searchsorted(t_src, t_dst, side="right") - 1
    idx = np.clip(idx, 0, len(v_src) - 1)
    out = v_src[idx]
    # Target times before the first sample have no predecessor to hold.
    out = np.where(t_dst < t_src[0], v_src[0], out)
    return out


def resample_group(
    df: pd.DataFrame,
    *,
    time_col: str,
    channels: dict[str, Interp],
    rate_hz: float,
    start: float | None = None,
    end: float | None = None,
) -> pd.DataFrame:
    """Resample one contiguous block onto a uniform grid at ``rate_hz``.

    The grid runs from ``start`` to ``end`` inclusive of the start. Channels are
    interpolated according to their declared kind.
    """
    if rate_hz <= 0:
        raise ValueError(f"rate_hz must be positive, got {rate_hz}")
    if df.empty:
        return pd.DataFrame(columns=[time_col, *channels])

    t = df[time_col].to_numpy(dtype="float64")
    order = np.argsort(t, kind="stable")
    t = t[order]

    lo = float(t[0]) if start is None else float(start)
    hi = float(t[-1]) if end is None else float(end)
    step = 1.0 / rate_hz
    n = int(np.floor((hi - lo) / step)) + 1
    grid = lo + np.arange(max(n, 1)) * step

    out = {time_col: grid}
    for name, how in channels.items():
        if name not in df.columns:
            continue
        v = df[name].to_numpy(dtype="float64")[order]
        out[name] = _interp_channel(t, v, grid, how)
    return pd.DataFrame(out)


def resample(
    df: pd.DataFrame,
    *,
    time_col: str,
    group_cols: Sequence[str],
    channels: dict[str, Interp],
    rate_hz: float,
) -> pd.DataFrame:
    """Resample every group onto its own uniform grid at a common rate.

    Groups are resampled independently — a grid spanning two assets would
    interpolate across the boundary between them, inventing samples that bridge
    two unrelated signals.
    """
    group_cols = list(group_cols)
    pieces: list[pd.DataFrame] = []
    for keys, part in df.groupby(group_cols, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        block = resample_group(part, time_col=time_col, channels=channels, rate_hz=rate_hz)
        for col, value in zip(group_cols, keys):
            block[col] = value
        pieces.append(block)
    if not pieces:
        return pd.DataFrame(columns=[*group_cols, time_col, *channels])
    out = pd.concat(pieces, ignore_index=True)
    return out[[*group_cols, time_col, *[c for c in channels if c in out.columns]]]


def observed_rate(df: pd.DataFrame, *, time_col: str, group_cols: Sequence[str]) -> pd.DataFrame:
    """Median sample rate per group — used to spot mixed-rate sources."""
    rows = []
    for keys, part in df.groupby(list(group_cols), sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        t = np.sort(part[time_col].to_numpy(dtype="float64"))
        dt = np.diff(t)
        dt = dt[dt > 0]
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "n": len(part),
                "median_dt": float(np.median(dt)) if len(dt) else np.nan,
                "rate_hz": float(1.0 / np.median(dt)) if len(dt) else np.nan,
            }
        )
    return pd.DataFrame(rows)
