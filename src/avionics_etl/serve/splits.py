"""Train/validation splitting — always by asset, never by row.

Rows in a trajectory are not independent: consecutive readings from the same
asset are nearly identical. A random row split puts sequence 40 of asset 7 in
train and sequence 41 in validation, so the model recognises rather than
predicts, and the score is fiction.

We must generalise to *new assets*, so we split by asset. This module is the only
place a split is produced; nothing else constructs one.

See docs/PRIMER.md section 8a.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from avionics_etl.canonical import model

TRAIN = "train"
VALIDATION = "validation"
HOLDOUT = "holdout"


def split_assets(
    assets: pd.DataFrame,
    *,
    fraction: float,
    seed: int,
    stratify_by: Sequence[str] = (model.SOURCE_ID,),
) -> pd.DataFrame:
    """Assign each asset to train or validation.

    ``assets`` must have one row per asset with the entity and stratification
    columns. Splitting is **stratified** — performed independently within each
    stratum, so every source is represented on both sides rather than one source
    landing entirely in validation by chance.

    Deterministic for a given seed, and stable under reordering of the input:
    assets are sorted before assignment, so a different input order cannot
    silently produce a different split.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be strictly between 0 and 1, got {fraction}")

    strata = list(stratify_by)
    key = [*strata, model.ASSET_UID]
    ordered = assets.sort_values(key).reset_index(drop=True)
    out = ordered.copy()
    out["fold"] = TRAIN

    rng = np.random.default_rng(seed)
    for _, group in ordered.groupby(strata, sort=True):
        n_val = max(1, int(round(len(group) * fraction)))
        n_val = min(n_val, len(group) - 1)  # never leave a stratum without training data
        chosen = rng.choice(group.index.to_numpy(), size=n_val, replace=False)
        out.loc[chosen, "fold"] = VALIDATION

    return out[[*key, "fold"]]


def assign_split(
    df: pd.DataFrame,
    *,
    fraction: float,
    seed: int,
    stratify_by: Sequence[str] = (model.SOURCE_ID,),
) -> pd.Series:
    """Per-row fold label, derived from an asset-level split.

    Assets whose trajectory is **incomplete** are placed in ``holdout``: their
    remaining life is not derivable from their own history, so they are the
    evaluation set rather than training material. In C-MAPSS these are the test
    trajectories; in the real system they are every recovered box.
    """
    strata = list(stratify_by)
    key = [*strata, model.ASSET_UID]

    complete = df[df[model.TRAJECTORY_COMPLETE]] if model.TRAJECTORY_COMPLETE in df else df
    assets = complete[key].drop_duplicates()
    mapping = split_assets(assets, fraction=fraction, seed=seed, stratify_by=strata)

    folds = df.merge(mapping, on=key, how="left")["fold"]
    folds.index = df.index
    if model.TRAJECTORY_COMPLETE in df.columns:
        folds = folds.where(df[model.TRAJECTORY_COMPLETE], HOLDOUT)
    return folds.fillna(HOLDOUT).rename("fold")


def assert_no_asset_crosses_folds(df: pd.DataFrame, fold_col: str = "fold") -> None:
    """Raise if any asset appears in more than one fold."""
    counts = df.groupby(model.ASSET_KEY, sort=False)[fold_col].nunique()
    offenders = counts[counts > 1]
    if len(offenders):
        raise AssertionError(
            f"{len(offenders)} asset(s) appear in more than one fold — this is a "
            f"row-level split and invalidates every metric: "
            f"{offenders.head(5).to_dict()}"
        )
