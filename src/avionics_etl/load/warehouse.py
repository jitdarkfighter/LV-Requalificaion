"""The analytical warehouse: a DuckDB star schema over the Parquet layers.

Why DuckDB and not a server
---------------------------
This workload is entirely **OLAP** — few queries, each scanning and aggregating
large numbers of rows ("mean degradation rate by fault mode and operating
regime"). It is never OLTP: nothing looks up a single row by key. DuckDB is
built for exactly that shape, runs in-process with nothing to administer, and
reads Parquet directly.

Crucially the tables here are **views, not copies**. Parquet remains the single
source of truth; the warehouse is a query surface over it. Rebuilding is
instant, nothing can drift out of sync, and the whole thing ports to another
engine by pointing that engine at the same files.

The schema
----------
::

              dim_source        dim_regime
                    \\              /
                     \\            /
      dim_asset ---- fact_measurement ---- dim_life_phase
            |
            |
       fact_decision ---- dim_cost_policy

**Facts** hold what we measure, one row per (asset, sequence). **Dimensions**
hold what we slice by. Splitting them this way is what makes slice, dice,
drill-down and roll-up plain ``GROUP BY`` queries.

``fact_decision`` and ``dim_cost_policy`` are real tables rather than views,
because they hold something no upstream file contains: the verdicts. The spec
requires "conditional clearance based on decisions made during past missions ...
with human confirmation", which is a *data* requirement hiding in a sentence
about process — to offer a precedent, the system has to store past decisions and
be able to find comparable ones. A verdict is also meaningless without the
economics that produced it, so the cost policy in force is recorded alongside
it and never overwritten.

Usage::

    python -m avionics_etl.load.warehouse            # build
    python -m avionics_etl.load.warehouse --schema   # show tables and columns
    python -m avionics_etl.load.warehouse --demo     # run the example queries
    python -m avionics_etl.load.warehouse --sql "select ..."
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import pandas as pd
import yaml

from avionics_etl import config, lineage
from avionics_etl.serve.contract import FEATURES
from avionics_etl.transform.labels import LABELS
from avionics_etl.transform.silver import MEASUREMENTS
from avionics_etl.transform.validate import VIOLATIONS

DB_FILE = "warehouse.duckdb"


# ---------------------------------------------------------------------------
# example queries — the "slicing and dicing" the spec asks for
# ---------------------------------------------------------------------------

DEMO_QUERIES: dict[str, tuple[str, str]] = {
    "fleet_overview": (
        "Executive view: one row per source",
        """
        SELECT s.source_id,
               s.description,
               s.operating_conditions                AS regimes,
               COUNT(DISTINCT a.asset_uid)           AS assets,
               SUM(a.lifetime)                       AS total_cycles,
               ROUND(AVG(a.lifetime), 1)             AS mean_lifetime,
               SUM(CASE WHEN NOT a.trajectory_complete THEN 1 ELSE 0 END)
                                                     AS awaiting_assessment
        FROM dim_asset a
        JOIN dim_source s USING (source_id)
        GROUP BY ALL
        ORDER BY s.source_id
        """,
    ),
    "degradation_by_regime": (
        "Dice: mean normalised HPC-outlet temperature by regime and life phase",
        """
        SELECT m.source_id,
               r.alt, r.mach, r.tra,
               p.phase,
               p.sort_order,
               COUNT(*)                        AS rows,
               ROUND(AVG(m.s4_z), 3)           AS mean_T50_z,
               ROUND(AVG(m.n_outliers), 3)     AS mean_outlier_flags
        FROM fact_measurement m
        JOIN dim_regime r USING (regime_id)
        JOIN dim_life_phase p ON m.rul_raw BETWEEN p.rul_low AND p.rul_high
        WHERE m.source_id = 'FD004'
        GROUP BY ALL
        ORDER BY r.alt, p.sort_order
        """,
    ),
    "outliers_vs_life": (
        "Roll up: do the outlier detectors track degradation?",
        """
        SELECT p.phase,
               p.sort_order,
               COUNT(*)                                                    AS rows,
               ROUND(100.0 * AVG(CASE WHEN m.n_outliers > 0 THEN 1 ELSE 0 END), 1)
                                                                           AS pct_rows_flagged,
               ROUND(AVG(m.n_outliers), 2)                                 AS mean_flags
        FROM fact_measurement m
        JOIN dim_life_phase p ON m.rul_raw BETWEEN p.rul_low AND p.rul_high
        WHERE m.trajectory_complete
        GROUP BY ALL
        ORDER BY p.sort_order
        """,
    ),
    "asset_detail": (
        "Drill down: the final cycles of one asset",
        """
        SELECT sequence_no, regime_id, rul_raw, rul, n_outliers,
               ROUND(s4_z, 3) AS T50_z, ROUND(s11_z, 3) AS Ps30_z
        FROM fact_measurement
        WHERE asset_uid = (
            SELECT asset_uid FROM dim_asset
            WHERE source_id = 'FD004' AND trajectory_complete
            ORDER BY lifetime DESC LIMIT 1
        )
        ORDER BY sequence_no DESC
        LIMIT 10
        """,
    ),
    "data_quality": (
        "Quality: what the deterministic checks rejected",
        """
        SELECT COALESCE(rule, 'none')        AS rule,
               COALESCE(channel, '-')        AS channel,
               COUNT(*)                      AS violations
        FROM quarantine
        GROUP BY ALL
        ORDER BY violations DESC
        """,
    ),
}


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def _glob(path: Path) -> str:
    return str(path / "**" / "*.parquet").replace("'", "''")


def build(cfg: config.Config) -> lineage.Lineage:
    rec = lineage.Lineage.start("warehouse", cfg.hash)
    artifacts = cfg.ensure("artifacts")
    db_path = artifacts / DB_FILE
    db_path.unlink(missing_ok=True)

    silver = cfg.path("silver")
    gold = cfg.path("gold")
    for path, target in ((silver / MEASUREMENTS, "silver"), (gold / FEATURES, "gold")):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run `make {target}` first.")

    src_meta = yaml.safe_load((config.REPO_ROOT / "configs" / "sources.yaml").read_text())
    sources = pd.DataFrame(
        [
            {
                "source_id": sid,
                "description": spec.get("description"),
                "operating_conditions": spec.get("operating_conditions"),
                "fault_modes": ", ".join(spec.get("fault_modes", [])),
                "n_fault_modes": len(spec.get("fault_modes", [])),
                "simulator": spec.get("simulator"),
            }
            for sid, spec in (src_meta.get("sources") or {}).items()
        ]
    )

    con = duckdb.connect(str(db_path))
    con.register("sources_seed", sources)

    quarantine = cfg.path("quarantine") / VIOLATIONS
    con.execute(f"""
        -- Raw layer views. Parquet stays the single source of truth; these are
        -- a query surface over it, never a copy.
        CREATE VIEW v_silver AS SELECT * FROM read_parquet('{_glob(silver / MEASUREMENTS)}', hive_partitioning=1);
        CREATE VIEW v_labels AS SELECT * FROM read_parquet('{_glob(silver / LABELS)}', hive_partitioning=1);
        CREATE VIEW v_gold   AS SELECT * FROM read_parquet('{_glob(gold / FEATURES)}', hive_partitioning=1);

        -- Dimensions ------------------------------------------------------
        CREATE TABLE dim_source AS SELECT * FROM sources_seed;

        CREATE TABLE dim_asset AS
        SELECT source_id,
               asset_uid,
               any_value(asset_id)             AS asset_id,
               any_value(batch_id)             AS batch_id,
               any_value(mission_id)           AS mission_id,
               any_value(trajectory_complete)  AS trajectory_complete,
               any_value(fold)                 AS fold,
               MAX(sequence_no)                AS lifetime,
               COUNT(*)                        AS observations
        FROM v_silver
        GROUP BY source_id, asset_uid;

        -- One row per distinct flight condition, with the physical settings so
        -- a dashboard can label it rather than showing an opaque key.
        CREATE TABLE dim_regime AS
        SELECT DISTINCT regime_id,
               CAST(round(op_setting_1) AS INTEGER) AS alt,
               CAST(round(op_setting_2) AS INTEGER) AS mach,
               CAST(round(op_setting_3) AS INTEGER) AS tra
        FROM v_silver;

        -- Degradation phase, as a dimension so it can be joined rather than
        -- re-derived with a CASE expression in every query.
        CREATE TABLE dim_life_phase (phase VARCHAR, rul_low INTEGER, rul_high INTEGER, sort_order INTEGER);
        INSERT INTO dim_life_phase VALUES
            ('imminent (0-10)',    0,  10, 1),
            ('critical (11-25)',  11,  25, 2),
            ('degraded (26-50)',  26,  50, 3),
            ('wearing (51-100)',  51, 100, 4),
            ('healthy (101-200)',101, 200, 5),
            ('nominal (200+)',   201, 999999, 6);

        -- Facts -----------------------------------------------------------
        CREATE VIEW fact_measurement AS
        SELECT m.*, l.rul, l.rul_raw
        FROM v_silver m
        JOIN v_labels l USING (source_id, asset_uid, sequence_no);

        CREATE VIEW fact_feature AS SELECT * FROM v_gold;
    """)

    if quarantine.exists():
        con.execute(f"CREATE VIEW quarantine AS SELECT * FROM read_parquet('{_glob(quarantine)}', hive_partitioning=1);")
    else:
        # An empty quarantine is the normal state for clean data, and the
        # dashboard should still be able to query it.
        con.execute("""
            CREATE VIEW quarantine AS
            SELECT NULL::VARCHAR AS source_id, NULL::VARCHAR AS asset_uid,
                   NULL::INTEGER AS sequence_no, NULL::VARCHAR AS rule,
                   NULL::VARCHAR AS channel, NULL::DOUBLE AS value,
                   NULL::VARCHAR AS detail
            WHERE FALSE;
        """)

    con.execute("""
        -- Decision tables. Real tables, not views: they hold the one thing no
        -- upstream file contains. A verdict without the economics that produced
        -- it cannot be explained later, so the policy is recorded with it.
        CREATE TABLE dim_cost_policy (
            policy_id        VARCHAR PRIMARY KEY,
            mission_class    VARCHAR,      -- test flight, commercial, crewed
            cost_scrap_good  DOUBLE,       -- scrapping a serviceable unit
            cost_fly_bad     DOUBLE,       -- flying a unit that then fails
            cost_inspect     DOUBLE,
            rul_threshold    INTEGER,      -- cycles below which a unit is rejected
            valid_from       DATE,
            valid_to         DATE,         -- NULL = currently in force
            notes            VARCHAR
        );

        CREATE TABLE fact_decision (
            decision_id      VARCHAR PRIMARY KEY,
            source_id        VARCHAR,
            asset_uid        VARCHAR,
            mission_id       VARCHAR,
            assessed_at      TIMESTAMP,
            verdict          VARCHAR,      -- accept | inspect | scrap
            predicted_rul    DOUBLE,
            rul_p10          DOUBLE,       -- the distribution, not just a point
            rul_p50          DOUBLE,
            rul_p90          DOUBLE,
            model_version    VARCHAR,      -- so a later model cannot rewrite history
            policy_id        VARCHAR,
            confirmed_by     VARCHAR,      -- the human in the loop
            confirmed_at     TIMESTAMP,
            rationale        VARCHAR
        );

        -- Precedent lookup: the "conditional clearance based on decisions made
        -- during past missions" the spec asks for.
        CREATE VIEW v_decision_precedent AS
        SELECT d.*, a.lifetime, a.trajectory_complete, p.mission_class, p.rul_threshold
        FROM fact_decision d
        LEFT JOIN dim_asset a USING (source_id, asset_uid)
        LEFT JOIN dim_cost_policy p USING (policy_id);
    """)

    counts = {}
    for table in ("dim_source", "dim_asset", "dim_regime", "dim_life_phase"):
        counts[table] = int(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    for view in ("fact_measurement", "fact_feature", "quarantine"):
        counts[view] = int(con.execute(f"SELECT COUNT(*) FROM {view}").fetchone()[0])
    con.close()

    rec.notes.update({"database": str(db_path), "counts": counts})
    rec.finish(artifacts / "manifests")

    print(f"warehouse: {db_path}")
    for name, n in counts.items():
        print(f"{'':10}{name:<20} {n:>10,}")
    return rec


def connect(cfg: config.Config | None = None, *, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    cfg = cfg or config.load()
    path = cfg.path("artifacts") / DB_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `make warehouse` first.")
    return duckdb.connect(str(path), read_only=read_only)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--schema", action="store_true", help="list tables and columns")
    parser.add_argument("--demo", action="store_true", help="run the example slice-and-dice queries")
    parser.add_argument("--sql", default=None, help="run one query and print the result")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    cfg = config.load(args.config)

    if args.schema or args.demo or args.sql:
        con = connect(cfg)
        pd.set_option("display.width", 200, "display.max_columns", 50)
        if args.schema:
            print(con.execute("""
                SELECT table_name, table_type, COUNT(*) AS columns
                FROM information_schema.columns
                JOIN information_schema.tables USING (table_name)
                WHERE table_schema = 'main'
                GROUP BY ALL ORDER BY table_type, table_name
            """).df().to_string(index=False))
        if args.sql:
            print(con.execute(args.sql).df().to_string(index=False))
        if args.demo:
            for name, (caption, sql) in DEMO_QUERIES.items():
                print(f"\n\033[1m{name}\033[0m — {caption}")
                df = con.execute(sql).df()
                print(df.head(12).to_string(index=False) if len(df) else "  (no rows)")
        con.close()
        return 0

    build(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
