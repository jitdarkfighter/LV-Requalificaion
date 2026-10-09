"""Stage `features`: pairwise features for every candidate (a = S1, b = S2/S3).

Country is deliberately NOT a feature (open set; France is unseen), and nothing here is
country-specific: similarities are string/number based, weights are IDF from the split itself.

Groups
  retrieval    cosine, dense ranks both ways, sparse score/rank/#shared keys
  competition  how this pair compares with the other candidates of the same S1 (a-side) and of
               the same S2/S3 record (b-side): gaps to the best, ranks, #competitors ("select")
  name         fuzzy ratios on romanized / core / compact / OCR-repaired names, initials,
               IDF-weighted token overlap, generic/legal-word overlap
  address      fuzzy ratios, IDF-weighted word overlap, abbreviation-aware coverage,
               house/unit number agreement
Output work/<split>/feats/part-*.parquet (+ `label` on train).
"""
import gc
import math
from multiprocessing import Pool

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from .config import P, log, work_dir

FIELDS = ["iid", "name_n", "core", "compact", "ocr", "acr", "stop", "aliases", "addr_n", "addr_core",
          "alpha", "nums", "first_num", "has_addr", "country"]
_IDF_N, _IDF_A = {}, {}


def _competition(c: pl.DataFrame) -> pl.DataFrame:
    return c.with_columns(
        n_cand_a=pl.len().over("a").cast(pl.Int32),
        n_cand_b=pl.len().over("b").cast(pl.Int32),
        cos_gap_a=pl.col("cos") - pl.col("cos").max().over("a"),
        cos_gap_b=pl.col("cos") - pl.col("cos").max().over("b"),
        cos_rank_a=pl.col("cos").rank("min", descending=True).over("a").cast(pl.Int32),
        cos_rank_b=pl.col("cos").rank("min", descending=True).over("b").cast(pl.Int32),
        sp_gap_a=pl.col("sp_score") - pl.col("sp_score").max().over("a"),
        sp_gap_b=pl.col("sp_score") - pl.col("sp_score").max().over("b"),
        sp_rank_b=pl.col("sp_score").rank("min", descending=True).over("b").cast(pl.Int32),
    )


def _wjacc(x, y, idf, default):
    if not x or not y:
        return math.nan
    sx, sy = set(x), set(y)
    w = lambda s: sum(idf.get(t, default) for t in s)
    union = w(sx | sy)
    return w(sx & sy) / union if union else math.nan


def _abbr_cov(x, y):
    """Share of b's address words that equal or abbreviate (prefix) a word of a: 'av' ~ 'avenue'.
    (Typos are covered by the fuzzy ratios; a per-word Jaro-Winkler loop was too slow at 100M pairs.)"""
    if not x or not y:
        return math.nan
    sx = set(x)
    hit = sum(1 for t in set(y) if t in sx or any(u.startswith(t) or t.startswith(u) for u in sx))
    return hit / len(set(y))


def _py_feats(rows):
    out = []
    for country, na, nb, aa, ab, ma, mb, sa, sb, fa, fb in rows:
        idf_n, idf_a = _IDF_N.get(country, {}), _IDF_A.get(country, {})
        dn, da = idf_n.get("\0max", 10.0), idf_a.get("\0max", 10.0)
        sma, smb = set(ma or ()), set(mb or ())
        num_j = len(sma & smb) / len(sma | smb) if (sma or smb) else math.nan
        stop_a, stop_b = set(sa.split()), set(sb.split())
        out.append((
            _wjacc(na.split(), nb.split(), idf_n, dn),
            _wjacc(aa, ab, idf_a, da),
            _abbr_cov(aa or (), ab or ()),
            num_j,
            len(sma & smb) / len(sma) if sma else math.nan,
            float(fa == fb) if fa and fb else math.nan,
            len(stop_a & stop_b) / len(stop_a | stop_b) if (stop_a or stop_b) else math.nan,
        ))
    return out


PY_COLS = ["n_idf_jacc", "a_idf_jacc", "a_abbr_cov", "num_jacc", "num_a_in_b", "first_num_eq", "stop_jacc"]


def _load_idf(split):
    out = {}
    for field in ("name", "addr"):
        df = pl.read_parquet(work_dir(split) / f"df_{field}.parquet").with_columns(
            idf=(pl.col("n") / pl.col("df")).log())
        d = {}
        for (c,), g in df.group_by("country"):
            m = dict(zip(g["t"].to_list(), g["idf"].to_list()))
            m["\0max"] = float(g["idf"].max())
            d[c] = m
        out[field] = d
    return out["name"], out["addr"]


def _cp(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=P.workers, dtype=np.float32)


def _chunk_features(c: pl.DataFrame, pool) -> pl.DataFrame:
    g = lambda col: c[col].fill_null("").to_list()
    na, nb = g("name_n"), g("name_n_b")
    ca, cb = g("core"), g("core_b")
    ka, kb = g("compact"), g("compact_b")
    has_b = c["has_addr_b"].to_numpy()
    aa, ab = g("addr_n"), g("addr_n_b")
    nan_if_no_addr = lambda x: np.where(has_b, x, np.nan).astype(np.float32)
    f = {
        "n_ratio": _cp(na, nb, fuzz.ratio), "n_tsort": _cp(na, nb, fuzz.token_sort_ratio),
        "n_tset": _cp(na, nb, fuzz.token_set_ratio), "n_partial": _cp(na, nb, fuzz.partial_ratio),
        "n_jw": _cp(na, nb, JaroWinkler.normalized_similarity),
        "c_ratio": _cp(ca, cb, fuzz.ratio), "c_tset": _cp(ca, cb, fuzz.token_set_ratio),
        "k_ratio": _cp(ka, kb, fuzz.ratio), "k_partial": _cp(ka, kb, fuzz.partial_ratio),
        "o_ratio": _cp(g("ocr"), g("ocr_b"), fuzz.ratio),
        "a_ratio": nan_if_no_addr(_cp(aa, ab, fuzz.ratio)),
        "a_tset": nan_if_no_addr(_cp(aa, ab, fuzz.token_set_ratio)),
        "a_tsort": nan_if_no_addr(_cp(aa, ab, fuzz.token_sort_ratio)),
        "a_partial": nan_if_no_addr(_cp(aa, ab, fuzz.partial_ratio)),
    }
    # Best name match over alternative names (dba / formerly / bracketed) on either side.
    best = f["n_tset"].copy()
    al_a, al_b = c["aliases"].to_list(), c["aliases_b"].to_list()
    for i in np.flatnonzero([bool(x) or bool(y) for x, y in zip(al_a, al_b)]):
        opts_a, opts_b = [na[i]] + (al_a[i] or []), [nb[i]] + (al_b[i] or [])
        best[i] = max(fuzz.token_set_ratio(x, y) for x in opts_a for y in opts_b)
    f["n_alias_best"] = best
    xa, xb = g("addr_core"), g("addr_core_b")      # landmark parts removed
    f["a_core_tset"] = nan_if_no_addr(_cp(xa, xb, fuzz.token_set_ratio))
    f["a_core_ratio"] = nan_if_no_addr(_cp(xa, xb, fuzz.ratio))
    acr_a, acr_b = g("acr"), g("acr_b")
    f["acr_match"] = np.array([float(bool(x and x == kb_) or bool(y and y == ka_))
                               for x, y, ka_, kb_ in zip(acr_a, acr_b, ka, kb)], dtype=np.float32)
    rows = list(zip(c["country"].to_list(), na, nb, c["alpha"].to_list(), c["alpha_b"].to_list(),
                    c["nums"].to_list(), c["nums_b"].to_list(), g("stop"), g("stop_b"),
                    g("first_num"), g("first_num_b")))
    parts = pool.map(_py_feats, [rows[i:i + 50_000] for i in range(0, len(rows), 50_000)])
    py = np.array([r for part in parts for r in part], dtype=np.float32).reshape(-1, len(PY_COLS))
    for i, col in enumerate(PY_COLS):
        f[col] = py[:, i]
    f["len_name_a"] = c["name_n"].str.len_chars().cast(pl.Float32).to_numpy()
    f["len_name_b"] = c["name_n_b"].str.len_chars().cast(pl.Float32).to_numpy()
    f["has_addr_b"] = has_b.astype(np.float32)
    f["is_s3"] = (c["b"] >= 10**10).cast(pl.Float32).to_numpy()
    keep = ["a", "b", "cos", "rank_fwd", "rank_rev", "sp_score", "sp_rank", "sp_n", "n_cand_a",
            "n_cand_b", "cos_gap_a", "cos_gap_b", "cos_rank_a", "cos_rank_b", "sp_gap_a", "sp_gap_b",
            "sp_rank_b"] + (["label"] if "label" in c.columns else [])
    return c.select(keep).with_columns(**{k: pl.Series(v) for k, v in f.items()})


def run(split: str) -> None:
    global _IDF_N, _IDF_A
    out = work_dir(split, "feats")
    if (out / "_DONE").exists():
        log(f"features[{split}]: cached")
        return
    wd = work_dir(split)
    cands = _competition(pl.read_parquet(wd / "cands.parquet"))
    if split == "train":
        from .data import load_gt_pairs
        pairs, _ = load_gt_pairs()
        cands = cands.join(pairs.with_columns(label=pl.lit(1, pl.Int8)), on=["a", "b"], how="left") \
                     .with_columns(pl.col("label").fill_null(0))
    s1 = pl.read_parquet(wd / "s1.parquet", columns=FIELDS)
    oth = pl.read_parquet(wd / "oth.parquet", columns=FIELDS).drop("country")
    oth = oth.rename({c: f"{c}_b" for c in oth.columns if c != "iid"})
    _IDF_N, _IDF_A = _load_idf(split)
    ids = s1["iid"].to_numpy()
    step = 200_000
    gc.collect()
    gc.freeze()          # workers inherit the IDF dicts; keep their GC from copying them (OOM)
    with Pool(P.pool_workers) as pool:
        for i, s in enumerate(range(0, len(ids), step)):
            c = (cands.filter(pl.col("a").is_in(pl.Series(ids[s:s + step])))
                 .join(s1, left_on="a", right_on="iid").join(oth, left_on="b", right_on="iid"))
            _chunk_features(c, pool).write_parquet(out / f"part-{i:04d}.parquet")
            log(f"features[{split}]: chunk {i} ({len(c):,} pairs)")
    (out / "_DONE").touch()
