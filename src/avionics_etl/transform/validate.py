"""Validate bronze against the channel contracts; quarantine what fails.

This is the deterministic half of the deliverable. It produces two outputs:

``data/silver/validated/``
    Rows that satisfy every contract. The input to everything downstream.

``data/quarantine/violations/``
    One row per (data row, channel, rule) that failed, carrying the entity keys,
    the offending value, the limit it broke, and a readable reason.

**Nothing is ever silently dropped.** A silent drop is a lie: the pipeline
reports success while data disappears. The quarantine table is a first-class
output — it is what a human reviews, and in a qualification context it is the
evidence that a deterministic check was actually performed.

Two kinds of check run here:

*Integrity* — is the data structurally sound? Duplicate keys, trajectories that
do not start at 1, sequences with gaps. These are properties of the dataset, not
of a value.

*Contract* — null, bounds, rate-of-change, per channel. See
:mod:`avionics_etl.contracts`.

Usage::

    python -m avionics_etl.transform.validate
    python -m avionics_etl.transform.validate --demo      # inject faults, write nothing
    python -m avionics_etl.transform.validate --describe  # print the contract table
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from avionics_etl import config, lineage
from avionics_etl.canonical import model
from avionics_etl.canonical.model import ChannelRegistry
from avionics_etl.contracts import VIOLATION_COLUMNS, ContractSuite
from avionics_etl.extract.ingest import MEASUREMENTS
from avionics_etl.load import parquet_writer

VALIDATED = "validated"
VIOLATIONS = "violations"

KEY_COLS = list(model.ROW_KEY)


# ---------------------------------------------------------------------------
# integrity
# ---------------------------------------------------------------------------

def check_integrity(df: pd.DataFrame) -> pd.DataFrame:
    """Structural checks. Returns violations in the same long form as contracts."""
    found: list[pd.DataFrame] = []

    dupes = df.duplicated(subset=KEY_COLS, keep=False)
    if dupes.any():
        idx = df.index[dupes]
        found.append(
            pd.DataFrame(
                {
                    "rule": "duplicate_key",
                    "channel": "",
                    "value": np.nan,
                    "limit_low": np.nan,
                    "limit_high": np.nan,
                    "detail": "more than one row shares this (source, asset, sequence)",
                },
                index=idx,
            )
        )

    starts = df.groupby(KEY_COLS[:-1], sort=False)[model.SEQUENCE_NO].transform("min")
    bad_start = starts != 1
    if bad_start.any():
        idx = df.index[bad_start]
        found.append(
            pd.DataFrame(
                {
                    "rule": "sequence_start",
                    "channel": model.SEQUENCE_NO,
                    "value": df.loc[idx, model.SEQUENCE_NO].astype("float64"),
                    "limit_low": 1.0,
                    "limit_high": np.nan,
                    "detail": "trajectory does not start at sequence 1",
                },
                index=idx,
            )
        )

    if not found:
        return pd.DataFrame(columns=VIOLATION_COLUMNS)
    return pd.concat(found).sort_index()


def find_sequence_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Trajectories whose sequence numbers are not contiguous.

    Reported rather than quarantined: a gap is usually the *consequence* of
    quarantining a row, and knowing where history is now incomplete matters more
    than rejecting the surrounding rows too.
    """
    g = df.groupby(KEY_COLS[:-1], sort=False)[model.SEQUENCE_NO]
    summary = g.agg(n="count", lo="min", hi="max").reset_index()
    summary["expected"] = summary.hi - summary.lo + 1
    gaps = summary[summary.n != summary.expected].copy()
    gaps["missing"] = gaps.expected - gaps.n
    return gaps


# ---------------------------------------------------------------------------
# fault injection (demonstration only)
# ---------------------------------------------------------------------------

def inject_faults(df: pd.DataFrame, registry: ChannelRegistry, *, n: int, seed: int) -> pd.DataFrame:
    """Corrupt a sample of rows so the checks have something to catch.

    C-MAPSS is simulator output with bounds derived from its own observed ranges,
    so in normal operation nothing violates anything — which would leave the most
    important mechanism in the pipeline untested and undemonstrated. This makes
    it visible. It is never used on the real path.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    sensors = [c for c in registry.sensors if c in out.columns]
    picks = rng.choice(len(out), size=min(n, len(out)), replace=False)

    for i, row in enumerate(picks):
        channel = sensors[i % len(sensors)]
        ch = registry[channel]
        kind = i % 3
        if kind == 0 and ch.bounds:                       # impossible value
            out.iloc[row, out.columns.get_loc(channel)] = ch.bounds[1] * 10
        elif kind == 1:                                   # missing value
            out.iloc[row, out.columns.get_loc(channel)] = np.nan
        elif ch.max_rate:                                 # impossible jump
            base = float(out.iloc[row][channel])
            out.iloc[row, out.columns.get_loc(channel)] = base + ch.max_rate * 5
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def validate(cfg: config.Config, *, demo_faults: int = 0, seed: int = 7) -> lineage.Lineage:
    rec = lineage.Lineage.start("validate" if not demo_faults else "validate-demo", cfg.hash)
    registry = ChannelRegistry.from_yaml(config.REPO_ROOT / "configs" / "channels.yaml")
    suite = ContractSuite.from_registry(registry)

    bronze = cfg.path("bronze") / MEASUREMENTS
    if not bronze.exists():
        raise FileNotFoundError(f"bronze layer not found at {bronze}. Run `make bronze` first.")

    df = parquet_writer.read_partitioned(bronze).reset_index(drop=True)
    if demo_faults:
        df = inject_faults(df, registry, n=demo_faults, seed=seed)

    violations = pd.concat(
        [
            check_integrity(df),
            suite.evaluate(df, group_cols=KEY_COLS[:-1]),
        ]
    ).sort_index()

    # A row is rejected if it broke any rule. Capture that before reshaping the
    # violation table, which discards the index that points back at df.
    bad_index = pd.Index(sorted(set(violations.index)))
    clean = df.drop(index=bad_index)

    # Attach entity keys so the quarantine table stands on its own — a reviewer
    # should not need to join back to bronze to know what they are looking at.
    if len(violations):
        keys = df.loc[violations.index, [*KEY_COLS, model.BATCH_ID, model.MISSION_ID]].reset_index(drop=True)
        violations = pd.concat([keys, violations.reset_index(drop=True)], axis=1)

    stamp = rec.stamp()
    pq_cfg = cfg["parquet"]

    if demo_faults:
        out = cfg.ensure("quarantine") / "_demo"
        if len(violations):
            for col, value in stamp.items():
                violations[col] = value
            parquet_writer.write_partitioned(
                violations, out / VIOLATIONS,
                partition_by=[model.SOURCE_ID], sort_by=[model.ASSET_ID, model.SEQUENCE_NO],
                compression=pq_cfg["compression"]["silver"],
                row_group_bytes=pq_cfg["row_group_bytes"],
            )
        _report(df, clean, violations, demo=True)
        rec.notes.update({"injected": demo_faults, "violations": int(len(violations))})
        rec.finish(cfg.ensure("artifacts") / "manifests")
        return rec

    for col, value in stamp.items():
        clean = clean.assign(**{col: value})
    parquet_writer.write_partitioned(
        clean, cfg.ensure("silver") / VALIDATED,
        partition_by=pq_cfg["partition_by"], sort_by=pq_cfg["sort_within_partition"],
        compression=pq_cfg["compression"]["silver"],
        row_group_bytes=pq_cfg["row_group_bytes"],
    )

    quarantine = cfg.ensure("quarantine")
    if len(violations):
        for col, value in stamp.items():
            violations[col] = value
        parquet_writer.write_partitioned(
            violations, quarantine / VIOLATIONS,
            partition_by=[model.SOURCE_ID], sort_by=[model.ASSET_ID, model.SEQUENCE_NO],
            compression=pq_cfg["compression"]["silver"],
            row_group_bytes=pq_cfg["row_group_bytes"],
        )

    gaps = find_sequence_gaps(clean)
    rec.notes.update(
        {
            "rows_in": int(len(df)),
            "rows_validated": int(len(clean)),
            "rows_quarantined": int(len(df) - len(clean)),
            "violations": int(len(violations)),
            "trajectories_with_gaps": int(len(gaps)),
        }
    )
    rec.add_output(VALIDATED, int(len(clean)))
    rec.finish(cfg.ensure("artifacts") / "manifests")

    _report(df, clean, violations, gaps=gaps)
    return rec


def _report(df, clean, violations, *, gaps=None, demo: bool = False) -> None:
    prefix = "demo   " if demo else "validate"
    rejected = len(df) - len(clean)
    print(f"{prefix}: {len(df):,} rows in, {len(clean):,} validated, {rejected:,} quarantined")
    if len(violations):
        counts = violations.groupby(["rule", "channel"]).size().sort_values(ascending=False)
        print(f"{'':8}  {len(violations):,} violation(s):")
        for (rule, channel), n in counts.head(12).items():
            label = f"{rule}/{channel}" if channel else rule
            print(f"{'':10}{label:<28} {n:>6,}")
    else:
        print(f"{'':8}  no contract violations")
    if gaps is not None and len(gaps):
        print(f"{'':8}  {len(gaps)} trajectory(ies) now have sequence gaps (reported, not rejected)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--demo", type=int, nargs="?", const=200, default=0,
                        metavar="N", help="inject N synthetic faults to demonstrate the checks")
    parser.add_argument("--describe", action="store_true", help="print the contract table and exit")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    cfg = config.load(args.config)
    if args.describe:
        registry = ChannelRegistry.from_yaml(config.REPO_ROOT / "configs" / "channels.yaml")
        print(ContractSuite.from_registry(registry).describe().to_string(index=False))
        return 0

    validate(cfg, demo_faults=args.demo, seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
