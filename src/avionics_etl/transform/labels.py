"""Build RUL labels.

Two code paths, because trajectories come in two kinds and they are labelled
completely differently. Getting this wrong is easy and silent.

**Complete trajectory** — the asset ran to failure, so the last row *is* the
failure point and the label is derived from the trajectory itself::

    rul_raw = max(sequence_no) - sequence_no

**Incomplete trajectory** — the asset was still working when observation
stopped. Its remaining life cannot come from its own history; it has to be
supplied externally. C-MAPSS ships these in ``RUL_FD00x.txt``: one value per
asset, meaning the RUL *at its final observed row*. Earlier rows are offset
backwards from there::

    rul_raw = rul_true + (max(sequence_no) - sequence_no)

This is not a C-MAPSS quirk. It is exactly the real situation: a recovered
avionics box has not failed, so its remaining life is an external judgement, not
a measurement.

Both paths are then clipped::

    rul = min(rul_raw, rul_clip)

Clipping exists because early in life there is no degradation to detect. An
asset with 300 cycles left and one with 250 produce statistically identical
readings, so an unclipped label asks the model to fit noise. See
docs/PRIMER.md section 3.

Usage::

    python -m avionics_etl.transform.labels
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from avionics_etl import config, lineage
from avionics_etl.canonical import model
from avionics_etl.extract.ingest import EXTERNAL_LABELS
from avionics_etl.load import parquet_writer
from avionics_etl.transform.validate import VALIDATED

LABELS = "labels"

ASSET_KEY = list(model.ASSET_KEY)
ROW_KEY = list(model.ROW_KEY)

#: Columns the label table carries. Deliberately narrow — it joins onto features
#: rather than duplicating them.
LABEL_COLUMNS = [
    *ROW_KEY,
    model.BATCH_ID,
    model.MISSION_ID,
    model.TRAJECTORY_COMPLETE,
    "rul_raw",   # cycles to failure, unclipped
    "rul",       # the training target, clipped
]


def build_labels(
    measurements: pd.DataFrame,
    external: pd.DataFrame | None,
    *,
    rul_clip: int,
) -> pd.DataFrame:
    """Derive per-row RUL for both kinds of trajectory."""
    df = measurements[[*ROW_KEY, model.BATCH_ID, model.MISSION_ID, model.TRAJECTORY_COMPLETE]].copy()

    last = df.groupby(ASSET_KEY, sort=False)[model.SEQUENCE_NO].transform("max")
    remaining = last - df[model.SEQUENCE_NO]

    incomplete = ~df[model.TRAJECTORY_COMPLETE]
    offset = pd.Series(0, index=df.index, dtype="int64")

    if incomplete.any():
        if external is None or external.empty:
            raise ValueError(
                "there are incomplete trajectories but no external labels. "
                "An asset that did not run to failure cannot be labelled from "
                "its own history."
            )
        truth = external[[*ASSET_KEY, "rul_true"]].drop_duplicates(subset=ASSET_KEY)
        merged = df.merge(truth, on=ASSET_KEY, how="left")
        merged.index = df.index

        missing = incomplete & merged.rul_true.isna()
        if missing.any():
            gaps = df.loc[missing, ASSET_KEY].drop_duplicates()
            raise ValueError(
                f"{len(gaps)} incomplete asset(s) have no external label:\n"
                f"{gaps.head(10).to_string(index=False)}"
            )
        offset = merged.rul_true.fillna(0).astype("int64")

    df["rul_raw"] = (remaining + offset).astype("int32")
    df["rul"] = df.rul_raw.clip(upper=rul_clip).astype("int32")
    return df[LABEL_COLUMNS]


def run(cfg: config.Config) -> lineage.Lineage:
    rec = lineage.Lineage.start("labels", cfg.hash)

    validated = cfg.path("silver") / VALIDATED
    if not validated.exists():
        raise FileNotFoundError(f"{validated} not found. Run `make validate` first.")

    measurements = parquet_writer.read_partitioned(validated)
    external_dir = cfg.path("bronze") / EXTERNAL_LABELS
    external = parquet_writer.read_partitioned(external_dir) if external_dir.exists() else None

    labels = build_labels(measurements, external, rul_clip=cfg.rul_clip)
    for col, value in rec.stamp().items():
        labels[col] = value

    pq_cfg = cfg["parquet"]
    parquet_writer.write_partitioned(
        labels,
        cfg.ensure("silver") / LABELS,
        partition_by=pq_cfg["partition_by"],
        sort_by=pq_cfg["sort_within_partition"],
        compression=pq_cfg["compression"]["silver"],
        row_group_bytes=pq_cfg["row_group_bytes"],
    )

    clipped = int((labels.rul_raw > cfg.rul_clip).sum())
    rec.notes.update(
        {
            "rows": int(len(labels)),
            "rul_clip": cfg.rul_clip,
            "rows_at_clip": clipped,
            "clip_fraction": round(clipped / max(1, len(labels)), 4),
        }
    )
    rec.add_output(LABELS, int(len(labels)))
    rec.finish(cfg.ensure("artifacts") / "manifests")

    print(
        f"labels : {len(labels):,} rows, clip={cfg.rul_clip}, "
        f"{clipped:,} ({clipped / len(labels):.1%}) sit at the clip"
    )
    print(
        f"{'':8}  complete trajectories: "
        f"{int(labels[model.TRAJECTORY_COMPLETE].sum()):,} rows (label derived)"
    )
    print(
        f"{'':8}  incomplete trajectories: "
        f"{int((~labels[model.TRAJECTORY_COMPLETE]).sum()):,} rows (label supplied externally)"
    )
    return rec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    run(config.load(args.config))
    return 0


if __name__ == "__main__":
    sys.exit(main())
