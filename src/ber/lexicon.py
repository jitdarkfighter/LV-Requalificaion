"""Learn token equivalences from TRAIN ground-truth pairs (no external data).

For every matched pair (S1 record A, S2/S3 record B), units of B missing from A (`u`) are
counted against units of A missing from B (`v`). A unit is a word or two adjacent words. A
mapping u -> v is kept when it is frequent and consistent. From the provided data alone this learns:
  - abbreviations / variants:   st -> street, pvt -> private, texas -> tx, mh -> maharashtra
  - multi-word:                 up -> uttar pradesh, new york -> ny, wb -> west bengal
  - native-script words:        महाराष्ट्र -> maharashtra, उत्तर प्रदेश -> uttar pradesh
Mappings point to the S1 (clean) vocabulary. Names also use positional alignment when both names
have the same number of words (translated names keep word order).
"""
import polars as pl

from .config import P, log

_ASCII = r"^[\x00-\x7F]*$"


def _with_units(df: pl.DataFrame) -> pl.DataFrame:
    """(iid, toks) -> + bigrams of adjacent words."""
    return df.with_columns(bi=pl.col("toks").list.eval(
        pl.element() + " " + pl.element().shift(-1)).list.drop_nulls())


def _cooccurrence(pairs, a_tok, b_tok) -> pl.DataFrame:
    a, b = _with_units(a_tok), _with_units(b_tok)
    j = (pairs.join(a.rename({"iid": "a", "toks": "ta", "bi": "ba"}), on="a")
         .join(b.rename({"iid": "b", "toks": "tb", "bi": "bb"}), on="b")
         .with_row_index("pid")
         .with_columns(ua=pl.col("tb").list.set_difference("ta"),     # words of B missing in A
                       va=pl.col("ta").list.set_difference("tb")))
    # A two-word unit counts only when both of its words are missing from the other side.
    def units(side_bi, missing):
        bi = (j.select("pid", missing, w=side_bi).explode("w").drop_nulls("w")
              .with_columns(p=pl.col("w").str.split(" "))
              .filter(pl.col(missing).list.contains(pl.col("p").list.get(0))
                      & pl.col(missing).list.contains(pl.col("p").list.get(1)))
              .select("pid", "w"))
        uni = j.select("pid", w=missing).explode("w").drop_nulls("w")
        return pl.concat([uni, bi])
    u = units("bb", "ua").rename({"w": "u"})
    v = units("ba", "va").rename({"w": "v"})
    n_u = u.unique().group_by("u").len("n_u")
    co = u.join(v, on="pid").unique().group_by("u", "v").len("n_uv")
    return co.join(n_u, on="u")


def _positional(pairs, a_tok, b_tok) -> pl.DataFrame:
    j = (pairs.join(a_tok.rename({"iid": "a", "toks": "ta"}), on="a")
         .join(b_tok.rename({"iid": "b", "toks": "tb"}), on="b")
         .filter(pl.col("ta").list.len() == pl.col("tb").list.len())
         .select("ta", "tb").explode("ta", "tb").rename({"tb": "u", "ta": "v"}))
    j = j.filter(~pl.col("u").str.contains(_ASCII))                 # only native-script sources
    n_u = j.group_by("u").len("n_u")
    return j.group_by("u", "v").len("n_uv").join(n_u, on="u")


def _ok(u: str, v: str, s1_freq: dict, oth_freq: dict, strong: bool) -> bool:
    if u == v or not v.isascii():
        return False
    nu, nv = len(u.split()), len(v.split())
    if not u.isascii():
        return True                                  # translation: word or phrase -> Latin
    if nu == 2 and nv == 2:
        return False                                 # Latin phrase -> phrase: too ambiguous
    # Latin source must look like an abbreviation/variant of v and be rare in S1 (non-canonical).
    if u[0] != v[0] or s1_freq.get(u, 0) > 0.05 * max(oth_freq.get(u, 1), 1):
        return False
    return len(u) < len(v) or strong


def _select(co: pl.DataFrame, s1_freq: dict, oth_freq: dict, min_count: int) -> dict:
    co = co.filter(pl.col("n_uv") >= min_count).with_columns(nt=pl.col("v").str.count_matches(" ") + 1)
    top = (co.sort(["n_uv", "nt"], descending=True)
           .group_by("u", maintain_order=True)
           .agg(pl.col("v").head(10), pl.col("n_uv").head(10), n_u=pl.col("n_u").first()))
    out = {}
    for u, vs, ns, n_u in top.iter_rows():
        # prefer the longer phrase among near-ties ('up': 'uttar pradesh' over 'uttar')
        best = max(range(len(vs)), key=lambda i: (ns[i] >= 0.9 * ns[0], len(vs[i].split()), ns[i]))
        v, n = vs[best], ns[best]
        words = set(v.split())
        rivals = [ns[i] for i in range(len(vs))
                  if i != best and not (set(vs[i].split()) <= words or words <= set(vs[i].split()))]
        second = max(rivals, default=0)
        ratio = n / n_u
        if ratio < P.lex_min_ratio or (second and n < 1.5 * second):
            continue
        if _ok(u, v, s1_freq, oth_freq, strong=ratio >= 0.8):
            out[u] = v
    return out


def _freq(tok: pl.DataFrame) -> dict:
    t = _with_units(tok)
    words = t.select(x="toks").explode("x")
    bis = t.select(x="bi").explode("x")
    return dict(pl.concat([words, bis]).drop_nulls().group_by("x").len().iter_rows())


def learn(pairs, s1_tok: pl.DataFrame, oth_tok: pl.DataFrame, field: str) -> dict:
    """s1_tok / oth_tok: (iid, toks: list[str]) of folded tokens for one field."""
    if len(pairs) > 1_500_000:
        pairs = pairs.sample(n=1_500_000, seed=P.seed)
    s1_freq, oth_freq = _freq(s1_tok), _freq(oth_tok)
    lex = _select(_cooccurrence(pairs, s1_tok, oth_tok), s1_freq, oth_freq, P.lex_min_count)
    if field == "name":
        pos = _select(_positional(pairs, s1_tok, oth_tok), s1_freq, oth_freq, 3)
        lex = {**pos, **lex}
    phrases = {u: v for u, v in lex.items() if " " in u or " " in v}
    log(f"lexicon[{field}]: {len(lex):,} mappings ({len(phrases)} multi-word); e.g. "
        + ", ".join(f"{u}->{v}" for u, v in list(lex.items())[:10])
        + " | phrases: " + ", ".join(f"{u}->{v}" for u, v in list(phrases.items())[:15]))
    return lex


def learn_drop_cues(pairs, s1_comp: pl.DataFrame, oth_comp: pl.DataFrame,
                    min_count: int = 300, min_rate: float = 0.7) -> list:
    """Words that start optional address parts ('near X', 'opp Y'), learned from train pairs.

    s1_comp / oth_comp: (iid, comps: list[list[str]]) lexicon-normalized tokens per comma part.
    A part counts as dropped when <= 20% of its words appear anywhere in the matched record.
    A first word is a cue when parts it starts are dropped >= min_rate of the time.
    """
    if len(pairs) > 1_000_000:
        pairs = pairs.sample(n=1_000_000, seed=P.seed)
    flat = lambda df: df.select("iid", toks=pl.col("comps").list.eval(pl.element().flatten()))
    parts = lambda df: (df.explode("comps").rename({"comps": "part"})
                        .filter(pl.col("part").list.len() > 0)
                        .with_columns(first=pl.col("part").list.first()))
    rows = []
    for mine, theirs, me, other in [(s1_comp, oth_comp, "a", "b"), (oth_comp, s1_comp, "b", "a")]:
        other_toks = flat(theirs).filter(pl.col("toks").list.len() > 0)   # skip empty addresses
        j = (parts(mine).rename({"iid": me}).join(pairs.select("a", "b"), on=me)
             .join(other_toks.rename({"iid": other}), on=other)
             .with_columns(kept=pl.col("part").list.set_intersection("toks").list.len()
                           / pl.col("part").list.len()))
        rows.append(j.select("first", dropped=pl.col("kept") <= 0.2))
    stats = (pl.concat(rows).group_by("first")
             .agg(n=pl.len(), rate=pl.col("dropped").mean())
             .filter((pl.col("n") >= min_count) & (pl.col("rate") >= min_rate)
                     & pl.col("first").str.contains(r"^[a-z]{2,12}$"))
             .sort("n", descending=True))
    cues = stats["first"].to_list()
    log(f"landmark cues ({len(cues)}): " + ", ".join(
        f"{f}({r:.2f},{n})" for f, n, r in stats.head(30).select("first", "n", "rate").iter_rows()))
    return cues
