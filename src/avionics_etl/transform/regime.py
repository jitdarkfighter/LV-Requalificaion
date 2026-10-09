"""Assign an operating regime to every row.

Sensor readings change for two entirely different reasons: the asset is
degrading, or the operating condition changed. The second is a **confounder** —
an unwanted variable that moves the measurement and hides the effect we care
about. On C-MAPSS FD002 it is so dominant that the degradation signal is not
merely weak, it is statistically invisible (correlations of 0.001–0.056 with the
label).

Fixing that starts with knowing which regime each row was recorded in.

Much of the published literature uses k-means clustering on the operational
settings. That works, but it is unnecessary: on C-MAPSS the regimes separate
**exactly** by rounding the settings to integers. Rounding is deterministic, has
no random seed, needs no fitted model to persist, and cannot drift between runs.
Verified against the data: FD002 and FD004 both yield exactly six regimes,
identical in both; FD001 and FD003 yield one.

See docs/PRIMER.md section 6.
"""

from __future__ import annotations

import pandas as pd

from avionics_etl.canonical.model import ChannelRegistry

REGIME_COL = "regime_id"

#: Separator in the composite regime key. Chosen so the key stays readable in a
#: partition path and a dashboard filter.
SEP = "|"


def assign_regime(df: pd.DataFrame, registry: ChannelRegistry) -> pd.Series:
    """Composite regime key from the rounded operating-condition channels.

    The key is the rounded settings joined by ``|`` — e.g. ``"25|1|60"``. It is
    self-describing (you can read the flight condition straight off it) and
    fully deterministic, so the same input always produces the same key on any
    machine, in any run order.
    """
    settings = [c for c in registry.settings if c in df.columns]
    if not settings:
        raise ValueError(
            "no operating-condition channels found. Declare at least one channel "
            "with role 'setting' in the registry, or the regime is undefined."
        )
    rounded = df[settings].round(0).astype("int64").astype(str)
    return rounded.agg(SEP.join, axis=1).rename(REGIME_COL)


def regime_dimension(df: pd.DataFrame, registry: ChannelRegistry) -> pd.DataFrame:
    """One row per distinct regime — the dimension table for the warehouse.

    Carries the physical settings alongside the key, so a dashboard can label a
    regime "35 kft, Mach 0.84" rather than showing an opaque id.
    """
    settings = [c for c in registry.settings if c in df.columns]
    out = (
        df[[REGIME_COL, *settings]]
        .assign(**{c: df[c].round(0).astype("int64") for c in settings})
        .drop_duplicates(subset=[REGIME_COL])
        .sort_values(REGIME_COL)
        .reset_index(drop=True)
    )
    renames = {c: f"{registry[c].symbol or c}" for c in settings}
    return out.rename(columns=renames)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Rows per regime per source — a quick check that assignment behaved."""
    from avionics_etl.canonical import model

    return (
        df.groupby([model.SOURCE_ID, REGIME_COL], sort=True)
        .size()
        .rename("rows")
        .reset_index()
    )
