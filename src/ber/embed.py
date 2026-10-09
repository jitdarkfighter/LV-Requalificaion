"""Stage `embed`: multilingual-e5 (MIT) embeddings of 'name | address' on the GPU.

Raw text is embedded (the model handles native scripts itself). Output:
work/<split>/emb_{s1,oth}.npy — float16, L2-normalized, row order = the prepared parquet.
"""
import numpy as np
import polars as pl

from .config import P, log, work_dir


def _texts(df: pl.DataFrame) -> list:
    prefix = "query: " if "e5" in P.emb_model else ""
    return df.select(pl.lit(prefix) + pl.col("name") + " | " + pl.col("addr")).to_series().to_list()


def run(split: str) -> None:
    out = work_dir(split)
    if (out / "emb_oth.npy").exists():
        log(f"embed[{split}]: cached")
        return
    import torch
    from sentence_transformers import SentenceTransformer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(P.emb_model, device=dev)
    if dev == "cuda":
        model.half()
    model.max_seq_length = 64
    for part in ("s1", "oth"):
        df = pl.read_parquet(out / f"{part}.parquet", columns=["name", "addr"])
        texts = _texts(df)
        embs = []
        for i in range(0, len(texts), 500_000):
            e = model.encode(texts[i:i + 500_000], batch_size=P.emb_batch, normalize_embeddings=True,
                             convert_to_tensor=True, show_progress_bar=False)
            embs.append(e.half().cpu().numpy())
            log(f"embed[{split}/{part}]: {min(i + 500_000, len(texts)):,}/{len(texts):,}")
        np.save(out / f"emb_{part}.npy", np.concatenate(embs))
