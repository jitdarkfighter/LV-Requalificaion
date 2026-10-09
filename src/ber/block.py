"""Stage `block`: candidate generation, always within country.

  dense_fwd  each S1 -> its top-k S2/S3 records by embedding cosine (GPU, exact)
  dense_rev  each S2/S3 record -> its top-k S1 entities (catches records their owner ranks low)
  sparse     each S1 -> top-k records by summed IDF of shared rare keys (name words, address
             words, house numbers) — a BM25-style lexical retriever in the spirit of Sparkly

Output work/<split>/cands.parquet: one row per (a, b) in the union, with retrieval features.
On train, recall of every route and of the union is written to block_stats.json.
"""
import json

import numpy as np
import polars as pl

from .config import P, log, work_dir


def _dense(split, s1, oth, E1, E2):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fwd, rev = [], []
    for country in s1["country"].unique().to_list():
        i1 = np.flatnonzero((s1["country"] == country).to_numpy())
        i2 = np.flatnonzero((oth["country"] == country).to_numpy())
        if not len(i1) or not len(i2):
            continue
        a_ids, b_ids = s1["iid"].to_numpy()[i1], oth["iid"].to_numpy()[i2]
        C = torch.from_numpy(E2[i2]).to(dev)
        Q = torch.from_numpy(E1[i1]).to(dev)
        k = min(P.dense_k_fwd, len(i2))
        for s in range(0, len(i1), 512):
            v, j = (Q[s:s + 512] @ C.T).topk(k, dim=1)       # fp16 top-k: no fp32 copy of the score matrix
            n = v.shape[0]
            fwd.append(pl.DataFrame({"a": np.repeat(a_ids[s:s + n], k), "b": b_ids[j.cpu().numpy().ravel()],
                                     "rank_fwd": np.tile(np.arange(1, k + 1, dtype=np.int16), n)}))
        k = min(P.dense_k_rev, len(i1))
        for s in range(0, len(i2), 2048):
            v, j = (C[s:s + 2048] @ Q.T).topk(k, dim=1)
            n = v.shape[0]
            rev.append(pl.DataFrame({"b": np.repeat(b_ids[s:s + n], k), "a": a_ids[j.cpu().numpy().ravel()],
                                     "rank_rev": np.tile(np.arange(1, k + 1, dtype=np.int16), n)}))
        del C, Q
        torch.cuda.empty_cache()
        log(f"block[{split}] dense {country}: {len(i1):,} x {len(i2):,}")
    return pl.concat(fwd), pl.concat(rev)


def _keys(df: pl.DataFrame) -> pl.DataFrame:
    h = lambda kind, t: pl.concat_str([pl.col("country"), pl.lit(kind), t], separator="|").hash(seed=7)
    n = (df.select("iid", "country", t=pl.col("core").str.split(" ")).explode("t")
         .filter(pl.col("t").str.len_chars() >= 2).select("iid", k=h("n", pl.col("t"))))
    a = (df.select("iid", "country", t="alpha").explode("t")
         .filter(pl.col("t").str.len_chars() >= 3).select("iid", k=h("a", pl.col("t"))))
    m = df.select("iid", "country", t="nums").explode("t").drop_nulls("t").select("iid", k=h("#", pl.col("t")))
    return pl.concat([n, a, m]).unique()


def _sparse(split, s1, oth):
    ko = _keys(oth)
    df = ko.group_by("k").len("df").filter(pl.col("df") <= P.key_df_cap)
    n_oth = oth.group_by("country").len("n")
    # IDF with each key's country size (keys embed the country, so one N per key is exact).
    kc = ko.join(oth.select("iid", "country"), on="iid").select("k", "country").unique("k")
    df = df.join(kc, on="k").join(n_oth, on="country").select(
        "k", w=(pl.col("n") / pl.col("df")).log().cast(pl.Float32))
    ko = ko.join(df.select("k"), on="k")
    ks = _keys(s1).join(df, on="k")
    ids = s1["iid"].to_numpy()
    out = []
    for s in range(0, len(ids), 20_000):
        chunk = ks.filter(pl.col("iid").is_in(pl.Series(ids[s:s + 20_000])))
        j = (chunk.rename({"iid": "a"}).join(ko.rename({"iid": "b"}), on="k")
             .group_by("a", "b").agg(sp_score=pl.col("w").sum(), sp_n=pl.len().cast(pl.Int16)))
        j = j.with_columns(sp_rank=pl.col("sp_score").rank("ordinal", descending=True).over("a"))
        out.append(j.filter(pl.col("sp_rank") <= P.sparse_k).with_columns(pl.col("sp_rank").cast(pl.Int16)))
    log(f"block[{split}] sparse done")
    return pl.concat(out)


def _cosine(cands, s1, oth, E1, E2):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pos1 = pl.DataFrame({"a": s1["iid"], "ia": np.arange(len(s1))})
    pos2 = pl.DataFrame({"b": oth["iid"], "ib": np.arange(len(oth))})
    c = cands.join(pos1, on="a").join(pos2, on="b")
    ia, ib = c["ia"].to_numpy(), c["ib"].to_numpy()
    cos = np.empty(len(c), dtype=np.float32)
    for s in range(0, len(c), 2_000_000):
        x = torch.from_numpy(E1[ia[s:s + 2_000_000]]).to(dev).float()
        y = torch.from_numpy(E2[ib[s:s + 2_000_000]]).to(dev).float()
        cos[s:s + 2_000_000] = (x * y).sum(1).cpu().numpy()
    return c.drop("ia", "ib").with_columns(cos=pl.Series(cos))


def run(split: str) -> None:
    out = work_dir(split)
    if (out / "cands.parquet").exists():
        log(f"block[{split}]: cached")
        return
    s1 = pl.read_parquet(out / "s1.parquet", columns=["iid", "country", "core", "alpha", "nums"])
    oth = pl.read_parquet(out / "oth.parquet", columns=["iid", "country", "core", "alpha", "nums"])
    E1, E2 = np.load(out / "emb_s1.npy"), np.load(out / "emb_oth.npy")
    fwd, rev = _dense(split, s1, oth, E1, E2)
    sp = _sparse(split, s1, oth)
    cands = (fwd.join(rev, on=["a", "b"], how="full", coalesce=True)
             .join(sp, on=["a", "b"], how="full", coalesce=True))
    cands = _cosine(cands, s1, oth, E1, E2).with_columns(
        rank_fwd=pl.col("rank_fwd").fill_null(P.dense_k_fwd + 1),
        rank_rev=pl.col("rank_rev").fill_null(P.dense_k_rev + 1),
        sp_rank=pl.col("sp_rank").fill_null(P.sparse_k + 1),
        sp_score=pl.col("sp_score").fill_null(0.0), sp_n=pl.col("sp_n").fill_null(0))
    cands.write_parquet(out / "cands.parquet")
    log(f"block[{split}]: {len(cands):,} candidates, {len(cands) / len(s1):.1f} per S1")

    if split == "train":
        from .data import load_gt_pairs
        pairs, _ = load_gt_pairs()
        pairs = pairs.join(s1.select(a="iid", country="country"), on="a").join(
            oth.select(b="iid"), on="b").with_columns(src=pl.when(pl.col("b") >= 10**10).then(pl.lit("S3")).otherwise(pl.lit("S2")))
        hit = pairs.join(cands.select("a", "b", "rank_fwd", "rank_rev", "sp_rank"), on=["a", "b"], how="left")
        within = lambda c, k: (hit[c] <= k).fill_null(False).mean()   # misses count as 0, not skipped
        stats = {
            "cands_per_s1": len(cands) / len(s1),
            "recall_union": hit["rank_fwd"].is_not_null().mean(),
            "recall_dense_fwd": within("rank_fwd", P.dense_k_fwd),
            "recall_dense_rev": within("rank_rev", P.dense_k_rev),
            "recall_sparse": within("sp_rank", P.sparse_k),
            "by_country_src": {f"{c}/{s}": r for c, s, r in hit.group_by("country", "src").agg(
                r=pl.col("rank_fwd").is_not_null().mean()).sort("country", "src").iter_rows()},
            "entities_all_found": hit.group_by("a").agg(ok=pl.col("rank_fwd").is_not_null().all())["ok"].mean(),
        }
        (out / "block_stats.json").write_text(json.dumps(stats, indent=2))
        log(f"block[train] recall: {json.dumps(stats)}")
