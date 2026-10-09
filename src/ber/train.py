"""Stage `train`: LightGBM matcher with leakage-free validation, then a final model on ALL data.

  1. Folds = hash(S1 id) % K  (GroupKFold by entity: an S1 and all its candidates share a fold).
  2. Per fold: train on the other folds (all positives + all hard negatives + a re-weighted
     sample of easy negatives), early-stop on a held-out slice, predict the WHOLE held-out fold.
  3. Out-of-fold (OOF) predictions over every train candidate -> isotonic calibration ->
     decision-rule tuning with the exact competition metric (decide.tune).
  4. Final model on 100% of train with the mean best iteration x 1.1 (more data -> a few more trees).
Outputs in work/model/: fold models, final.txt, calibrator.pkl, decision.json, report.json.
"""
import json
import pickle

import lightgbm as lgb
import numpy as np
import polars as pl

from . import decide
from .config import P, log, work_dir
from .data import load_gt_pairs

ID_COLS = ["a", "b", "label"]
LGB_PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=127, min_data_in_leaf=500,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  max_bin=255, verbose=-1, seed=P.seed, num_threads=P.workers)


def load_features(split: str) -> pl.DataFrame:
    return pl.read_parquet(work_dir(split, "feats") / "part-*.parquet")


def feature_names(df: pl.DataFrame) -> list:
    return [c for c in df.columns if c not in ID_COLS]


def _train_mask(df: pl.DataFrame, rng) -> tuple:
    hard = ((pl.col("label") == 1) | (pl.col("rank_fwd") <= P.neg_keep_rank)
            | (pl.col("sp_rank") <= P.neg_keep_rank) | (pl.col("cos_rank_b") <= 2))
    is_hard = df.select(hard).to_series().to_numpy()
    keep = is_hard | (rng.random(len(df)) < P.neg_sample)
    w = np.where(is_hard, 1.0, 1.0 / P.neg_sample).astype(np.float32)
    return keep, w


def run() -> None:
    md = work_dir("model")
    if (md / "report.json").exists():          # written last, after the final model
        log("train: cached")
        return
    df = load_features("train")
    feats = feature_names(df)
    log(f"train: {len(df):,} pairs, {df['label'].sum():,} positives, {len(feats)} features")
    X = df.select(pl.col(feats).cast(pl.Float32)).to_numpy()   # cast first: avoids a float64 copy
    y = df["label"].to_numpy()
    fold = (df["a"].hash(seed=P.seed) % P.cv_folds).to_numpy()
    rng = np.random.default_rng(P.seed)
    keep, w = _train_mask(df, rng)
    oof = np.zeros(len(df), dtype=np.float32)
    best_iters, imp = [], np.zeros(len(feats))
    for k in range(P.cv_folds):
        tr = (fold != k) & keep
        va = np.flatnonzero(fold == k)
        es = rng.choice(va, size=min(len(va), 3_000_000), replace=False)   # early-stopping slice
        dtr = lgb.Dataset(X[tr], y[tr], weight=w[tr], feature_name=feats, free_raw_data=True)
        dva = lgb.Dataset(X[es], y[es], reference=dtr)
        m = lgb.train(LGB_PARAMS, dtr, num_boost_round=3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)])
        oof[va] = m.predict(X[va], num_iteration=m.best_iteration)
        best_iters.append(m.best_iteration)
        imp += m.feature_importance("gain")
        m.save_model(str(md / f"fold{k}.txt"), num_iteration=m.best_iteration)
        log(f"train: fold {k} best_iter={m.best_iteration}")

    # Calibration + decision rule on OOF only.
    cal = decide.fit_calibrator(oof, y)
    pickle.dump(cal, open(md / "calibrator.pkl", "wb"))
    pairs, n_true = load_gt_pairs()
    s1_ids = pl.read_parquet(work_dir("train") / "s1.parquet", columns=["iid", "country"])
    n_true = n_true.join(s1_ids.select(a="iid"), on="a")          # smoke: only loaded S1s
    pairs = pairs.join(n_true.select("a"), on="a")
    o = df.select("a", "b").with_columns(p=pl.Series(cal.predict(oof).astype(np.float32)))
    o.with_columns(p_raw=pl.Series(oof), label=df["label"]).write_parquet(md / "oof.parquet")
    o = o.filter(pl.col("p") >= 0.02)
    tuned = decide.tune(o, pairs, n_true)
    rule = tuned["best"]
    (md / "decision.json").write_text(json.dumps(tuned, indent=2))

    # Per-country / per-source OOF score with the chosen rule (diagnostics).
    chosen = decide.select(o, rule)
    by_country = {}
    for (c,), g in s1_ids.group_by("country"):
        nt = n_true.join(g.select(a="iid"), on="a")
        by_country[c] = decide.macro_f05(chosen.join(nt.select("a"), on="a"), pairs, nt)
    # Per-fold scores with the chosen rule: each fold's S1 entities scored on their OOF predictions.
    s1_fold = n_true.with_columns(fold=(pl.col("a").hash(seed=P.seed) % P.cv_folds).cast(pl.Int32))
    per_fold = []
    for k in range(P.cv_folds):
        nt = s1_fold.filter(pl.col("fold") == k).select("a", "n_true")
        row = {"fold": k, "s1": len(nt), "macro_f05": decide.macro_f05(chosen.join(nt.select("a"), on="a"), pairs, nt)}
        for (c,), g in s1_ids.group_by("country"):
            ntc = nt.join(g.select(a="iid"), on="a")
            row[c] = decide.macro_f05(chosen.join(ntc.select("a"), on="a"), pairs, ntc)
        per_fold.append(row)
    scores = np.array([r["macro_f05"] for r in per_fold])
    log(f"train: per-fold macro F0.5 {np.round(scores, 4).tolist()} mean={scores.mean():.4f} std={scores.std():.4f}")
    # Recall ceiling: GT pairs that never became candidates can't be predicted.
    reach = pairs.join(df.select("a", "b"), on=["a", "b"]).height / max(len(pairs), 1)
    report = {"oof_macro_f05": rule["macro_f05"], "per_fold": per_fold,
              "fold_mean": float(scores.mean()), "fold_std": float(scores.std()),
              "candidate_recall": reach, "rule": rule, "by_country": by_country,
              "best_iters": best_iters, "n_pairs": len(df), "positives": int(y.sum()),
              "feature_gain": dict(sorted(zip(feats, (imp / imp.sum()).round(4).tolist()),
                                          key=lambda kv: -kv[1]))}
    log(f"train: OOF macro F0.5 = {rule['macro_f05']:.4f} {by_country}")

    # Final model on all training data.
    n_final = int(np.mean(best_iters) * 1.1)
    dall = lgb.Dataset(X[keep], y[keep], weight=w[keep], feature_name=feats)
    final = lgb.train(LGB_PARAMS, dall, num_boost_round=n_final)
    final.save_model(str(md / "final.txt"))
    report["final_iters"] = n_final
    (md / "report.json").write_text(json.dumps(report, indent=2))
    log(f"train: final model with {n_final} trees saved")
