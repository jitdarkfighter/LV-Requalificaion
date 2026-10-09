"""Loading the provided TSVs. Integer ids keep joins fast: S1-123 -> 123, S2-123 -> 123,
S3-123 -> 10**10 + 123 (S2/S3 share one id space; S1 has its own)."""
import polars as pl

from .config import DATA, SMOKE

S3_OFFSET = 10**10


def read_tsv(path) -> pl.DataFrame:
    # Fields contain literal '"' characters: disable quoting; files are plain tab-split.
    df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)
    return df.with_columns(pl.all().fill_null(""))


def int_id(col: str) -> pl.Expr:
    num = pl.col(col).str.slice(3).cast(pl.Int64)
    return pl.when(pl.col(col).str.starts_with("S3")).then(num + S3_OFFSET).otherwise(num)


def str_id(col: str, source1: bool = False) -> pl.Expr:
    if source1:
        return pl.lit("S1-") + pl.col(col).cast(pl.String)
    return (pl.when(pl.col(col) >= S3_OFFSET)
            .then(pl.lit("S3-") + (pl.col(col) - S3_OFFSET).cast(pl.String))
            .otherwise(pl.lit("S2-") + pl.col(col).cast(pl.String)))


def load_split(split: str):
    """Return (s1, oth): columns iid, entity_id, name, addr, country (+ src on oth)."""
    def load(n):
        df = read_tsv(DATA / split / f"{split}_source{n}.tsv")
        return df.select(iid=int_id("entity_id"), entity_id="entity_id", name="business_name",
                         addr="business_address", country="country")
    s1 = load(1)
    oth = pl.concat([load(2).with_columns(src=pl.lit("S2")), load(3).with_columns(src=pl.lit("S3"))])
    if SMOKE:   # consistent subsample: some S1s, all their matches, plus random decoys
        s1 = s1.sample(n=min(40_000, len(s1)), seed=0)
        keep = pl.Series(dtype=pl.Int64)
        if split == "train":
            pairs, _ = load_gt_pairs()
            keep = pairs.filter(pl.col("a").is_in(s1["iid"].implode()))["b"]
        decoys = oth.sample(n=min(200_000, len(oth)), seed=0)["iid"]
        oth = oth.filter(pl.col("iid").is_in(pl.concat([keep, decoys]).implode()))
    return s1, oth


def load_gt_pairs():
    """Ground-truth pairs (a = S1 iid, b = S2/S3 iid) plus n_true per S1 (incl. zeros).
    Not subsampled in smoke mode: callers restrict to the S1s they loaded."""
    gt = read_tsv(DATA / "train" / "train_ground_truth.tsv")
    pairs = (gt.select(s1="source1_entity_id", ids=pl.col("matched_entity_ids").str.split(","))
             .explode("ids").filter(pl.col("ids").is_not_null() & (pl.col("ids") != ""))
             .select(a=int_id("s1"), b=int_id("ids")))
    n_true = gt.select(a=int_id("source1_entity_id")).join(
        pairs.group_by("a").len().rename({"len": "n_true"}), on="a", how="left").fill_null(0)
    return pairs, n_true
