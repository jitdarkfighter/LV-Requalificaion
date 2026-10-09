"""Stage `predict`: score test candidates, apply the tuned decision rule, write submission files.

  output/matching_results.tsv  source1_entity_id \t matched_entity_ids   (every test S1, empty = none)
  output/candidate_pairs.tsv   source1_entity_id \t candidate_entity_ids (exact set the model scored)
The final model (trained on 100% of train) produces raw scores; the isotonic calibrator and
decision rule tuned on out-of-fold predictions turn them into match lists.
"""
import json
import pickle

import lightgbm as lgb
import numpy as np
import polars as pl

from . import decide
from .config import OUT, log, work_dir
from .data import str_id
from .train import load_features


def _write_lists(s1: pl.DataFrame, pairs: pl.DataFrame, col: str, path) -> None:
    lists = (pairs.unique(["a", "b"]).sort(["a", "b"])
             .with_columns(bid=str_id("b")).group_by("a").agg(pl.col("bid").str.join(",")))
    out = (s1.select(a="iid").join(lists, on="a", how="left")
           .select(source1_entity_id=str_id("a", source1=True), **{col: pl.col("bid").fill_null("")}))
    assert out["source1_entity_id"].n_unique() == len(s1)
    out.write_csv(path, separator="\t", quote_style="never")


def run() -> None:
    md, wd = work_dir("model"), work_dir("test")
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_features("test")
    booster = lgb.Booster(model_file=str(md / "final.txt"))
    feats = booster.feature_name()
    p_raw = booster.predict(df.select(pl.col(feats).cast(pl.Float32)).to_numpy())
    cal = pickle.load(open(md / "calibrator.pkl", "rb"))
    rule = json.loads((md / "decision.json").read_text())["best"]
    scored = df.select("a", "b").with_columns(p=pl.Series(cal.predict(p_raw).astype(np.float32)))
    scored.write_parquet(wd / "scored.parquet")
    chosen = decide.select(scored.filter(pl.col("p") >= 0.02), rule)

    s1 = pl.read_parquet(wd / "s1.parquet", columns=["iid", "country"])
    assert chosen.join(df.select("a", "b"), on=["a", "b"], how="anti").is_empty(), "match not in candidates"
    assert chosen["b"].n_unique() == len(chosen) or not rule.get("one_owner", True)
    _write_lists(s1, chosen, "matched_entity_ids", OUT / "matching_results.tsv")
    _write_lists(s1, df.select("a", "b"), "candidate_entity_ids", OUT / "candidate_pairs.tsv")

    per = s1.join(chosen.group_by("a").len("k"), left_on="iid", right_on="a", how="left").fill_null(0)
    stats = {c: {"s1": len(g), "empty_rate": float((g["k"] == 0).mean()), "mean_matches": float(g["k"].mean())}
             for (c,), g in per.group_by("country")}
    summary = {"rule": rule, "test_pairs_scored": len(df), "test_matches": len(chosen), "by_country": stats}
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2))
    log(f"predict: {len(chosen):,} matches; {json.dumps(stats)}")
