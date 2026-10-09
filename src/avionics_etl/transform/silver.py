"""Build the silver layer: regime, normalisation, outlier flags.

Order matters here, and each step depends on the one before it:

1. **Assign a regime** to every row, by rounding the operating-condition
   channels. Deterministic, no fitted model.
2. **Assign a fold** (train / validation / holdout), by asset and stratified by
   source. Needed *before* normalisation, because the statistics must be fitted
   on training rows only.
3. **Fit normalisation** on the training fold, persist it, apply it everywhere.
4. **Flag outliers** on the normalised values — unusual *for this operating
   condition*, not merely unusual overall.

The output keeps **both raw and normalised** columns. The ML side should be able
to measure what the transform bought rather than take our word for it.

Usage::

    python -m avionics_etl.transform.silver
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from avionics_etl import config, lineage
from avionics_etl.canonical import model
from avionics_etl.canonical.model import ChannelRegistry
from avionics_etl.load import parquet_writer
from avionics_etl.serve import splits
from avionics_etl.transform import normalize, outliers, regime
from avionics_etl.transform.validate import VALIDATED

MEASUREMENTS = "measurements"
REGIME_DIM = "dim_regime.parquet"


def run(cfg: config.Config) -> lineage.Lineage:
    rec = lineage.Lineage.start("silver", cfg.hash)
    registry = ChannelRegistry.from_yaml(config.REPO_ROOT / "configs" / "channels.yaml")

    validated = cfg.path("silver") / VALIDATED
    if not validated.exists():
        raise FileNotFoundError(f"{validated} not found. Run `make validate` first.")
    df = parquet_writer.read_partitioned(validated).reset_index(drop=True)

    # 1 -- regime -----------------------------------------------------------
    df[regime.REGIME_COL] = regime.assign_regime(df, registry)
    n_regimes = df.groupby(model.SOURCE_ID)[regime.REGIME_COL].nunique()

    # 2 -- fold -------------------------------------------------------------
    df["fold"] = splits.assign_split(
        df,
        fraction=float(cfg.get("split.validation_fraction", 0.2)),
        seed=int(cfg.get("split.seed", 0)),
    )
    splits.assert_no_asset_crosses_folds(df)

    # 3 -- normalisation, fitted on the training fold ONLY ------------------
    train = df[df.fold == splits.TRAIN]
    if train.empty:
        raise ValueError("the training fold is empty; cannot fit normalisation")
    params = normalize.fit(train, registry)

    unseen = normalize.unseen_groups(df, params)
    if len(unseen):
        # A regime that training never covered. Its rows normalise to NaN rather
        # than being scaled against the wrong baseline.
        print(f"{'':8}  WARNING {len(unseen)} (source, regime) pair(s) unseen in training:")
        print(unseen.to_string(index=False))

    normalised = normalize.apply(df, params)

    # 4 -- outliers, on the normalised values -------------------------------
    o_cfg = cfg["outliers"]
    regime_groups = df[[model.SOURCE_ID, regime.REGIME_COL]].astype(str).agg("|".join, axis=1)
    # Fences fitted on the training fold only, so the flags stay usable as
    # model features rather than quietly encoding the holdout distribution.
    iqr_fences = outliers.fit_iqr(
        normalised[df.fold == splits.TRAIN],
        regime_groups[df.fold == splits.TRAIN],
        multiplier=float(o_cfg["iqr_multiplier"]),
    )
    flags, outlier_summary = outliers.detect(
        normalised,
        regime_groups=regime_groups,
        asset_groups=df[model.ASSET_KEY].astype(str).agg("|".join, axis=1),
        zscore_threshold=float(o_cfg["zscore_threshold"]),
        iqr_multiplier=float(o_cfg["iqr_multiplier"]),
        hampel_window=int(o_cfg["hampel_window"]),
        hampel_n_sigma=float(o_cfg["hampel_n_sigma"]),
        iqr_fences=iqr_fences,
    )

    silver = pd.concat([df, normalised, flags], axis=1)
    for col, value in rec.stamp().items():
        silver[col] = value

    # -- write --------------------------------------------------------------
    pq_cfg = cfg["parquet"]
    parquet_writer.write_partitioned(
        silver,
        cfg.ensure("silver") / MEASUREMENTS,
        partition_by=pq_cfg["partition_by"],
        sort_by=pq_cfg["sort_within_partition"],
        compression=pq_cfg["compression"]["silver"],
        row_group_bytes=pq_cfg["row_group_bytes"],
    )

    artifacts = cfg.ensure("artifacts")
    params_path = normalize.save(params, artifacts)
    iqr_fences.to_parquet(artifacts / "iqr_fences.parquet", index=False)
    regime.regime_dimension(df, registry).to_parquet(artifacts / REGIME_DIM, index=False)

    # -- report -------------------------------------------------------------
    folds = df.fold.value_counts()
    flagged_rows = int((flags[outliers.COUNT_COL] > 0).sum())
    rec.notes.update(
        {
            "rows": int(len(silver)),
            "regimes_per_source": {k: int(v) for k, v in n_regimes.items()},
            "fold_rows": {k: int(v) for k, v in folds.items()},
            "normalisation_groups": int(len(params.drop_duplicates(normalize.GROUP_COLS))),
            "degenerate_params": int(params.degenerate.sum()),
            "undersampled_params": int(params.undersampled.sum()),
            "rows_with_outliers": flagged_rows,
            "outliers_by_method": outlier_summary.groupby("method").flagged.sum().to_dict(),
        }
    )
    rec.add_output(MEASUREMENTS, int(len(silver)))
    rec.finish(artifacts / "manifests")

    print(f"silver : {len(silver):,} rows, {len(silver.columns)} columns")
    print(f"{'':8}  regimes: " + ", ".join(f"{k}={v}" for k, v in n_regimes.items()))
    print(f"{'':8}  folds:   " + ", ".join(f"{k}={v:,}" for k, v in folds.items()))
    print(
        f"{'':8}  normalisation fitted on {len(train):,} training rows -> "
        f"{len(params):,} (source, regime, channel) params -> {params_path.name}"
    )
    if int(params.degenerate.sum()):
        print(
            f"{'':8}  {int(params.degenerate.sum())} degenerate param(s) "
            f"(channel constant within a regime; mapped to 0.0, not infinity)"
        )
    print(
        f"{'':8}  outliers: {flagged_rows:,} row(s) flagged "
        f"({flagged_rows / len(silver):.2%}), nothing removed"
    )
    for method, n in outlier_summary.groupby("method").flagged.sum().items():
        print(f"{'':10}{method:<10} {int(n):>9,} channel-flags")
    return rec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    run(config.load(args.config))
    return 0


if __name__ == "__main__":
    sys.exit(main())
