"""Flag outliers. Never delete them.

An **outlier** is a value far from the rest of the distribution. In most data
work the instinct is to remove them. Here that instinct is actively dangerous:
**in a fault-detection system, "unusual" is exactly what we are hunting.**
Deleting outliers can mean deleting the thing the platform was built to find.

So the rule is a clean split of responsibility:

* A **contract violation** (:mod:`avionics_etl.contracts`) is *impossible* — a
  temperature of 9999 degrees is a broken sensor. Those rows are quarantined.
* An **outlier** is merely *unusual*. It stays in the data with a flag beside it,
  and a human or a model decides what it means.

Three detectors, each catching something the others miss:

``zscore``   distance from the mean in standard deviations. Simple, but the mean
             and standard deviation are themselves distorted by the outliers.
``iqr``      outside ``[Q1 - k*IQR, Q3 + k*IQR]``. **Robust** — extreme values
             barely move the quartiles, so it does not hide what it is looking
             for.
``hampel``   a rolling, robust detector built for time series, comparing each
             point to the median of its recent neighbourhood via the **MAD**
             (median absolute deviation). Catches a spike that is unremarkable
             against the whole trajectory but wrong for where it sits.

All three run on **regime-normalised** values, so an outlier means "unusual for
this operating condition" rather than "unusual across all conditions", which
would just re-find the regimes.

The flags are safe to use as model features, and that safety is deliberate
rather than incidental: the Hampel window is backward-looking, the z-score uses
train-fitted normalisation, and the IQR fences are fitted on the training fold
and applied read-only. Fitting the fences over every row would let the
evaluation set decide what counts as unusual.

See docs/PRIMER.md section 12.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METHODS = ("zscore", "iqr", "hampel")

OUTLIER_SUFFIX = "_outlier"
COUNT_COL = "n_outliers"

#: Scale factor making the MAD a consistent estimator of the standard deviation
#: for normally distributed data.
MAD_TO_SIGMA = 1.4826

#: Absolute floor on the robust scale estimate. Inputs are regime-normalised, so
#: this is in units of standard deviations. Without any floor, a locally flat
#: channel divides by ~0 and flags the first real movement as an anomaly.
MIN_SIGMA = 0.1


def flag_zscore(values: pd.DataFrame, *, threshold: float) -> pd.DataFrame:
    """Values further than ``threshold`` standard deviations from their mean.

    Inputs are already regime-normalised, so the z-score *is* the value.
    """
    return values.abs() > threshold


def fit_iqr(values: pd.DataFrame, groups: pd.Series, *, multiplier: float) -> pd.DataFrame:
    """Interquartile fences per (group, channel).

    Fitted separately from application so the quantiles can be computed on the
    training fold alone. Computing them over every row would let the evaluation
    set's distribution decide what counts as unusual — leakage, and enough to
    make the flags unusable as model features (docs/PRIMER.md section 8c).
    """
    rows = []
    for key, part in values.groupby(groups, sort=False):
        q1 = part.quantile(0.25)
        q3 = part.quantile(0.75)
        iqr = q3 - q1
        rows.append(
            pd.DataFrame(
                {
                    "group": key,
                    "channel": values.columns,
                    "low": (q1 - multiplier * iqr).to_numpy(),
                    "high": (q3 + multiplier * iqr).to_numpy(),
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["group", "channel", "low", "high"]
    )


def flag_iqr(values: pd.DataFrame, groups: pd.Series, *, fences: pd.DataFrame) -> pd.DataFrame:
    """Apply fitted fences. Groups absent from the fences are never flagged."""
    out = {}
    for col in values.columns:
        spec = fences[fences.channel == col]
        low = groups.map(dict(zip(spec.group, spec.low)))
        high = groups.map(dict(zip(spec.group, spec.high)))
        out[col] = ((values[col] < low) | (values[col] > high)).fillna(False)
    return pd.DataFrame(out, index=values.index)


def quantisation_step(values: pd.DataFrame, groups: pd.Series) -> pd.Series:
    """The smallest change each channel can actually express, per channel.

    Several C-MAPSS sensors are coarsely quantised — s16 takes two distinct
    values across the entire dataset, s10 takes 21. For those, a rolling MAD is
    zero most of the time, and any scale floor smaller than one quantum makes
    every single tick look like a 5-sigma event.

    Estimated as the median non-zero step within a trajectory, which is exactly
    "one quantum" for a quantised channel and negligibly small for a continuous
    one.
    """
    steps = values.groupby(groups, sort=False).diff().abs()
    return steps.where(steps > 0).median().fillna(0.0)


def flag_hampel(
    values: pd.DataFrame,
    groups: pd.Series,
    *,
    window: int,
    n_sigma: float,
    min_sigma: float = MIN_SIGMA,
) -> pd.DataFrame:
    """Rolling robust detector, backward-looking, within each trajectory.

    Implemented as two rolling medians — of the value, and of the absolute
    deviation from that median — which keeps it vectorised rather than a
    per-window Python callback.

    Two details matter a great deal in practice, and getting them wrong makes
    this detector useless rather than merely imprecise:

    **A full window is required.** With partial windows the deviation from a
    two-point median is near zero, so the MAD collapses, so everything looks
    like an outlier. Rows without enough history are simply not judged — that
    is an honest absence of evidence, not a pass. Measured on C-MAPSS, allowing
    partial windows roughly doubles the flag rate.

    **The scale floor must scale with the channel's own resolution.** Where a
    channel is locally flat the MAD is legitimately zero. A fixed floor then
    makes coarsely quantised channels pathological: measured on C-MAPSS, a fixed
    floor of 0.1 flagged s16 (two distinct values in the whole dataset) on 4.2%
    of rows, because one quantum is 0.44 in normalised units. The floor is
    therefore the larger of an absolute minimum and one quantisation step, so a
    change the sensor cannot express more finely is never an anomaly.
    """
    floors = quantisation_step(values, groups).clip(lower=min_sigma)

    out = {}
    for col in values.columns:
        grouped = values[col].groupby(groups, sort=False)
        median = grouped.transform(
            lambda s: s.rolling(window, min_periods=window).median()
        )
        deviation = (values[col] - median).abs()
        mad = deviation.groupby(groups, sort=False).transform(
            lambda s: s.rolling(window, min_periods=window).median()
        )
        sigma = (MAD_TO_SIGMA * mad).clip(lower=float(floors[col]))
        out[col] = (deviation > n_sigma * sigma).fillna(False)
    return pd.DataFrame(out, index=values.index)


def detect(
    normalised: pd.DataFrame,
    *,
    regime_groups: pd.Series,
    asset_groups: pd.Series,
    zscore_threshold: float,
    iqr_multiplier: float,
    hampel_window: int,
    hampel_n_sigma: float,
    iqr_fences: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every detector and combine.

    Returns ``(flags, summary)``:

    * ``flags`` — one boolean column per channel (``{channel}_outlier``, true if
      *any* detector fired) plus ``n_outliers``, how many channels were flagged
      on that row. A row with many simultaneously flagged channels is a much
      stronger signal than one with a single odd reading.
    * ``summary`` — counts per channel per method, for the run manifest.
    """
    if iqr_fences is None:
        # No fitted fences supplied — fit on the data at hand. Callers that care
        # about leakage (the silver stage does) pass train-fitted fences instead.
        iqr_fences = fit_iqr(normalised, regime_groups, multiplier=iqr_multiplier)

    per_method = {
        "zscore": flag_zscore(normalised, threshold=zscore_threshold),
        "iqr": flag_iqr(normalised, regime_groups, fences=iqr_fences),
        "hampel": flag_hampel(
            normalised, asset_groups, window=hampel_window, n_sigma=hampel_n_sigma
        ),
    }

    combined = None
    for frame in per_method.values():
        combined = frame if combined is None else (combined | frame)
    assert combined is not None

    flags = combined.rename(columns=lambda c: f"{_base(c)}{OUTLIER_SUFFIX}")
    flags[COUNT_COL] = combined.sum(axis=1).astype("int16")

    rows = []
    for method, frame in per_method.items():
        for col in frame.columns:
            n = int(frame[col].sum())
            if n:
                rows.append({"method": method, "channel": _base(col), "flagged": n})
    summary = pd.DataFrame(rows, columns=["method", "channel", "flagged"])

    return flags, summary


def _base(column: str) -> str:
    """Strip the normalisation suffix so flags are named after the channel."""
    return column[:-2] if column.endswith("_z") else column
