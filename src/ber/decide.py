"""Turning pair probabilities into each S1's final match list, and scoring it.

  macro_f05   the competition metric: F0.5 per S1 entity, averaged over ALL entities
              (singleton = 1.0 only for an empty prediction; entities with no candidates count)
  calibrate   isotonic regression on out-of-fold predictions
  one_owner   each S2/S3 record keeps only its most probable S1 (clusters are disjoint in the data)
  select      'thresh': keep p >= t
              'expf'  : per S1, choose the prefix of candidates (sorted by p) — or the empty set —
                        with the highest expected F0.5 (Lipton et al., ECML-PKDD 2014 idea)
  tune        grid-search the rule and its parameters on out-of-fold predictions
"""
import json

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

from .config import log


def macro_f05(pred: pl.DataFrame, truth: pl.DataFrame, n_true: pl.DataFrame) -> float:
    """pred/truth: (a, b); n_true: (a, n_true) for every S1 entity being scored."""
    tp = pred.join(truth, on=["a", "b"]).group_by("a").len("tp")
    npred = pred.group_by("a").len("np")
    e = n_true.join(npred, on="a", how="left").join(tp, on="a", how="left").fill_null(0)
    t, n, k = (e[c].to_numpy().astype(np.float64) for c in ("tp", "n_true", "np"))
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(t > 0, 1.25 * t / (1.25 * t + 0.25 * (n - t) + (k - t)), 0.0)
    f[(n == 0) & (k == 0)] = 1.0
    return float(f.mean())


def fit_calibrator(p: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)


def one_owner(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col("p") == pl.col("p").max().over("b")).unique("b", keep="first")


def select(df: pl.DataFrame, rule: dict) -> pl.DataFrame:
    """df: (a, b, p) calibrated. Returns the chosen (a, b) pairs."""
    df = df.filter(pl.col("p") >= rule["floor"])
    if rule.get("one_owner", True):
        df = one_owner(df)
    if rule["kind"] == "thresh":
        return df.filter(pl.col("p") >= rule["t"]).select("a", "b")
    miss = rule["miss"]                                   # expected true matches blocking lost
    d = (df.sort(["a", "p"], descending=[False, True])
         .with_columns(k=pl.int_range(1, pl.len() + 1).over("a"),
                       tp=pl.col("p").cum_sum().over("a"),
                       tot=pl.col("p").sum().over("a") + miss,
                       p0=(1 - pl.col("p").clip(0, 0.999)).log().sum().over("a").exp()))
    d = d.with_columns(ef=1.25 * pl.col("tp") / (1.25 * pl.col("tp") + 0.25 * (pl.col("tot") - pl.col("tp"))
                                                 + (pl.col("k") - pl.col("tp"))))
    best = d.group_by("a").agg(kbest=pl.col("k").get(pl.col("ef").arg_max()),
                               efbest=pl.col("ef").max(), p0=pl.col("p0").first())
    best = best.filter(pl.col("efbest") > pl.col("p0") * np.exp(-miss))   # else predict nothing
    return d.join(best.select("a", "kbest"), on="a").filter(pl.col("k") <= pl.col("kbest")).select("a", "b")


def tune(oof: pl.DataFrame, truth: pl.DataFrame, n_true: pl.DataFrame) -> dict:
    """oof: (a, b, p) calibrated out-of-fold probabilities over all train candidates."""
    grid = [{"kind": "thresh", "t": t, "floor": 0.0, "one_owner": oo}
            for t in np.round(np.arange(0.2, 0.96, 0.05), 2) for oo in (True, False)]
    grid += [{"kind": "expf", "miss": m, "floor": fl, "one_owner": True}
             for m in (0.0, 0.05, 0.1, 0.2) for fl in (0.02, 0.05, 0.1, 0.2)]
    results = []
    for rule in grid:
        s = macro_f05(select(oof, rule), truth, n_true)
        results.append({**rule, "macro_f05": s})
    results.sort(key=lambda r: r["macro_f05"], reverse=True)
    for r in results[:8]:
        log(f"decide: {json.dumps(r)}")
    return {"best": results[0], "all": results}
