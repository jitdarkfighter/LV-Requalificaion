"""Stage `prepare`: normalize every record of a split.

Output work/<split>/{s1,oth}.parquet with, per record:
  name_n     romanized name tokens (lexicon + transliteration)
  core       name without the country's frequent generic/legal words
  compact    core with spaces removed (catches 'foo.com', 'FooBar')
  ocr        compact with digit look-alikes repaired ('6allego')
  acr        initials of the core ('sunrise producer' -> 'sp')
  stop       the generic/legal words that were removed
  addr_n     romanized address tokens
  addr_core  address minus landmark parts ('near X'; cue words learned from train pairs)
  alpha      address word tokens;  nums  house/unit numbers (zero-stripped);  first_num;  has_addr
Frequent-word lists and document frequencies are computed per split from that split's own
records (unsupervised), so France gets its own ('sarl', 'sas', ...) without any French rules.
"""
import gc
import json
from multiprocessing import Pool

import polars as pl

from . import lexicon
from .config import P, log, work_dir
from .data import load_gt_pairs, load_split
from .text import (PLACEHOLDERS, acronym, apply_phrases, fold_tokens, merge_letters, norm_number,
                   number_parts, ocr_fix, romanize, split_aliases, strip_domain, strip_urls)

_LEX_N, _LEX_A, _STOP, _STOP_DF, _CUES = {}, {}, {}, {}, frozenset()


def _pmap(fn, items, chunk=100_000):
    chunks = [items[i:i + chunk] for i in range(0, len(items), chunk)]
    # Forked workers inherit the parent's heap; without gc.freeze() each worker's garbage
    # collector touches every inherited object and copy-on-write duplicates GBs per worker (OOM).
    gc.collect()
    gc.freeze()
    try:
        with Pool(P.pool_workers) as pool:   # fork: workers inherit the module globals
            parts = pool.map(fn, chunks)
    finally:
        gc.unfreeze()
    return [x for part in parts for x in part]


def _name_tokens(s):
    return merge_letters(fold_tokens(strip_domain(s)))


def _fold_names(xs):
    """-> (main name tokens, [alternative-name tokens]) per record."""
    out = []
    for x in xs:
        x = strip_urls(x)
        main = _name_tokens(x)
        alts = [a for a in (_name_tokens(s) for s in split_aliases(x)) if a and a != main]
        out.append((main, alts))
    return out


def _fold_addrs(xs):
    """Folded tokens per comma-separated address part."""
    return [[p for p in (fold_tokens(c) for c in x.split(",")) if p] for x in xs]


def _rn(toks):
    return [r for t in apply_phrases(toks, _LEX_N) for r in romanize(t, _LEX_N)]


def _roman_names(items):
    return [(_rn(main), [_rn(a) for a in alts]) for main, alts in items]


def _clean_part(comp):
    return apply_phrases([t for t in comp if t not in PLACEHOLDERS], _LEX_A)


def _lex_parts(comp_lists):
    """Address parts with the learned lexicon applied and placeholders removed (for cue learning)."""
    out = []
    for comps in comp_lists:
        parts = [[w for t in _clean_part(comp) for w in _LEX_A.get(t, t).split()] for comp in comps]
        out.append([p for p in parts if p])
    return out


def _roman_addrs(comp_lists):
    """-> (all tokens, core tokens without parts that start with a learned landmark cue)."""
    out = []
    for comps in comp_lists:
        full, core = [], []
        for comp in comps:
            toks = _clean_part(comp)
            r = [w for t in toks for w in ([norm_number(t)] if t.isdigit() else romanize(t, _LEX_A))]
            full.extend(r)
            if not r or r[0] not in _CUES:                # cues are in lexicon-normalized form
                core.extend(r)
        out.append((full, core))
    return out


def _core(tokens, country):
    """Tokens minus the country's generic words, but never fewer than 2 words: the rarest
    removed words are added back ('raj foods limited' -> 'raj foods', not 'raj')."""
    stop = _STOP.get(country, frozenset())
    core = [t for t in tokens if t not in stop]
    if len(core) >= 2:
        return core
    removed = sorted({t for t in tokens if t in stop}, key=lambda t: _STOP_DF[country].get(t, 0))
    keep = set(removed[:2 - len(core)])
    return [t for t in tokens if t not in stop or t in keep]


def _derive(rows):
    """rows: ((name_tokens, alias_token_lists), (addr_tokens, addr_core_tokens), country)."""
    out = []
    for (name, alts), (addr, addr_core), country in rows:
        stop = _STOP.get(country, frozenset())
        core = _core(name, country)
        nums = sorted({d for t in addr for d in number_parts(t)})
        first = next((number_parts(t)[0] for t in addr if any(c.isdigit() for c in t)), "")
        aliases = [" ".join(a) for a in alts if any(t not in stop for t in a)]   # skip '(Pllc)'
        out.append((" ".join(name), " ".join(core), "".join(core), "".join(ocr_fix(t) for t in core),
                    acronym(core), " ".join(t for t in name if t in stop and t not in core),
                    aliases, " ".join(addr), " ".join(addr_core),
                    [t for t in addr if not any(c.isdigit() for c in t)], nums, first))
    return out


def _alpha(toks):
    return [t for t in toks if not any(c.isdigit() for c in t)]


def run(split: str) -> None:
    global _LEX_N, _LEX_A, _STOP, _STOP_DF, _CUES
    out = work_dir(split)
    if (out / "oth.parquet").exists():
        log(f"prepare[{split}]: cached")
        return
    s1, oth = load_split(split)
    log(f"prepare[{split}]: {len(s1):,} S1, {len(oth):,} S2/S3")
    recs = pl.concat([s1.with_columns(src=pl.lit("S1")), oth])
    names = _pmap(_fold_names, recs["name"].to_list())
    addrs = _pmap(_fold_addrs, recs["addr"].to_list())
    log("folded")

    lex_path = work_dir("train") / "lexicon.json"
    if split == "train":
        pairs, _ = load_gt_pairs()
        n1 = len(s1)
        mk = lambda ids, toks: pl.DataFrame({"iid": ids, "toks": toks},
                                            schema={"iid": pl.Int64, "toks": pl.List(pl.String)})
        ids = recs["iid"].to_list()
        mains = [m for m, _ in names]
        _LEX_N = lexicon.learn(pairs, mk(ids[:n1], mains[:n1]), mk(ids[n1:], mains[n1:]), "name")
        del mains
        a_alpha = [_alpha([t for part in comps for t in part]) for comps in addrs]
        _LEX_A = lexicon.learn(pairs, mk(ids[:n1], a_alpha[:n1]), mk(ids[n1:], a_alpha[n1:]), "addr")
        del a_alpha
        cues = []
        if P.landmark:
            mc = lambda ids_, comps: pl.DataFrame(
                {"iid": ids_, "comps": comps}, schema={"iid": pl.Int64, "comps": pl.List(pl.List(pl.String))})
            # Learn cues on lexicon-normalized parts: otherwise 'mh' vs 'maharashtra' looks "dropped"
            # and state names get mistaken for landmark words.
            mapped = _pmap(_lex_parts, addrs)
            cues = lexicon.learn_drop_cues(pairs, mc(ids[:n1], mapped[:n1]), mc(ids[n1:], mapped[n1:]))
            del mapped
        lex_path.write_text(json.dumps({"name": _LEX_N, "addr": _LEX_A, "addr_cues": cues},
                                       ensure_ascii=False), encoding="utf-8")
    else:
        lex = json.loads(lex_path.read_text(encoding="utf-8"))
        _LEX_N, _LEX_A, cues = lex["name"], lex["addr"], lex["addr_cues"]
    _CUES = frozenset(cues) if P.landmark else frozenset()

    names = _pmap(_roman_names, names)
    addrs = _pmap(_roman_addrs, addrs)
    log("romanized")

    # Generic/legal words per country: in >= stop_df of that country's names (this split only).
    tok = (pl.DataFrame({"country": recs["country"], "t": [m for m, _ in names]},
                        schema={"country": pl.String, "t": pl.List(pl.String)})
           .with_row_index("r").explode("t"))
    df_t = tok.drop_nulls("t").unique(["r", "t"]).group_by("country", "t").len("df")
    n_c = recs.group_by("country").len("n")
    stop = df_t.join(n_c, on="country").filter(pl.col("df") >= P.stop_df * pl.col("n"))
    _STOP = {c: frozenset(g["t"].to_list()) for (c,), g in stop.group_by("country")}
    _STOP_DF = {c: dict(zip(g["t"].to_list(), g["df"].to_list())) for (c,), g in stop.group_by("country")}
    log("generic words: " + "; ".join(f"{c}: {' '.join(sorted(s)[:20])}" for c, s in _STOP.items()))
    (out / "generic_words.json").write_text(json.dumps(
        {c: sorted(s) for c, s in _STOP.items()}, ensure_ascii=False, indent=1), encoding="utf-8")

    derived = _pmap(_derive, list(zip(names, addrs, recs["country"].to_list())))
    cols = ["name_n", "core", "compact", "ocr", "acr", "stop", "aliases", "addr_n", "addr_core",
            "alpha", "nums", "first_num"]
    lists = ("aliases", "alpha", "nums")
    d = pl.DataFrame(derived, schema=[(c, pl.List(pl.String) if c in lists else pl.String)
                                      for c in cols], orient="row")
    full = pl.concat([recs.select("iid", "country", "src", "name", "addr"), d], how="horizontal")
    full = full.with_columns(has_addr=pl.col("addr_n").str.len_chars() > 0)

    # Document frequencies for IDF weighting (per country, this split, unsupervised).
    for field, col in [("name", "name_n"), ("addr", "addr_n")]:
        (full.select("country", t=pl.col(col).str.split(" ")).with_row_index("r").explode("t")
         .filter(pl.col("t") != "").unique(["r", "t"]).group_by("country", "t").len("df")
         .join(full.group_by("country").len("n"), on="country")
         .write_parquet(out / f"df_{field}.parquet"))

    full.filter(pl.col("src") == "S1").drop("src").write_parquet(out / "s1.parquet")
    full.filter(pl.col("src") != "S1").write_parquet(out / "oth.parquet")
    log(f"prepare[{split}]: done")
