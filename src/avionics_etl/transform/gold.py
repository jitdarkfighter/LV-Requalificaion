"""Build the gold layer: analysis-ready features plus labels.

Joins silver measurements to labels, builds causal trend features from the
channel registry, optionally prunes near-duplicate features, and writes the
single table the ML side consumes through
:mod:`avionics_etl.serve.contract`.

Everything fitted here — the prune list, any PCA — is fitted on the **training
fold only** and persisted alongside the normalisation parameters, for the same
reason (docs/PRIMER.md section 8c).

Usage::

    python -m avionics_etl.transform.gold
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from avionics_etl import config, lineage
from avionics_etl.canonical import model
from avionics_etl.canonical.model import ChannelRegistry
from avionics_etl.load import parquet_writer
from avionics_etl.serve import splits
from avionics_etl.serve.contract import FEATURES
from avionics_etl.transform import features as F
from avionics_etl.transform import reduce as R
from avionics_etl.transform.labels import LABELS
from avionics_etl.transform.silver import MEASUREMENTS


def run(cfg: config.Config) -> lineage.Lineage:
    rec = lineage.Lineage.start("gold", cfg.hash)
    registry = ChannelRegistry.from_yaml(config.REPO_ROOT / "configs" / "channels.yaml")

    silver = cfg.path("silver") / MEASUREMENTS
    labels = cfg.path("silver") / LABELS
    for path, target in ((silver, "silver"), (labels, "labels")):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run `make {target}` first.")

    df = parquet_writer.read_partitioned(silver)
    lab = parquet_writer.read_partitioned(labels)[[*model.ROW_KEY, "rul", "rul_raw"]]
    df = df.merge(lab, on=model.ROW_KEY, how="inner").reset_index(drop=True)

    # Feature construction needs rows in trajectory order, or every window is
    # built from the wrong neighbours.
    df = df.sort_values([model.SOURCE_ID, model.ASSET_UID, model.SEQUENCE_NO]).reset_index(drop=True)

    f_cfg = cfg["features"]
    feats = F.build(
        df,
        registry,
        windows=list(f_cfg["rolling_windows"]),
        stats=list(f_cfg["rolling_stats"]),
        use_normalised=bool(f_cfg.get("use_normalised", True)),
    )
    built = len(feats.columns)

    # -- reduction, fitted on the training fold only ------------------------
    artifacts = cfg.ensure("artifacts")
    r_cfg = cfg.get("reduce", {}) or {}
    train_mask = df.fold == splits.TRAIN
    dropped: list[str] = []

    if r_cfg.get("prune_correlated", False):
        dropped = R.fit_prune(feats[train_mask], threshold=float(r_cfg.get("prune_threshold", 0.98)))
        feats = feats.drop(columns=dropped)
        (artifacts / R.PRUNE_FILE).write_text(
            json.dumps({"threshold": r_cfg.get("prune_threshold"), "dropped": dropped}, indent=2)
        )

    pca_components = 0
    if r_cfg.get("pca", False):
        pca = R.fit_pca(feats[train_mask], variance=float(r_cfg.get("pca_variance", 0.95)))
        pca.save(artifacts)
        pca_components = pca.n_components

    # -- assemble ------------------------------------------------------------
    carry = [c for c in model.IDENTIFIER_COLS if c in df.columns] + ["fold", "regime_id", "rul", "rul_raw"]
    carry += [c for c in df.columns if c.endswith("_outlier") or c == "n_outliers"]
    gold = pd.concat([df[carry].reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
    for col, value in rec.stamp().items():
        gold[col] = value

    pq_cfg = cfg["parquet"]
    parquet_writer.write_partitioned(
        gold,
        cfg.ensure("gold") / FEATURES,
        partition_by=pq_cfg["partition_by"],
        sort_by=pq_cfg["sort_within_partition"],
        compression=pq_cfg["compression"]["gold"],
        row_group_bytes=pq_cfg["row_group_bytes"],
    )

    legal = model.feature_columns(gold.columns)
    (artifacts / "feature_names.json").write_text(json.dumps(legal, indent=2))

    rec.notes.update(
        {
            "rows": int(len(gold)),
            "features_built": built,
            "features_pruned": len(dropped),
            "features_final": len(legal),
            "pca_components": pca_components,
            "fold_rows": {k: int(v) for k, v in df.fold.value_counts().items()},
        }
    )
    rec.add_output(FEATURES, int(len(gold)))
    rec.finish(artifacts / "manifests")

    print(f"gold   : {len(gold):,} rows, {len(legal)} features")
    print(f"{'':8}  built {built} -> pruned {len(dropped)} near-duplicates -> {len(legal)} usable")
    if pca_components:
        print(f"{'':8}  PCA: {pca_components} components retained")
    print(f"{'':8}  folds: " + ", ".join(f"{k}={v:,}" for k, v in df.fold.value_counts().items()))
    return rec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    run(config.load(args.config))
    return 0


if __name__ == "__main__":
    sys.exit(main())
