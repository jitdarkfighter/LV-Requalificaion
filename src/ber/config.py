"""Paths and hyper-parameters (paths can be overridden with BER_DATA / BER_WORK / BER_OUT)."""
import os
import time
from dataclasses import dataclass
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]          # .../src
_ROOT = _SRC.parent                                   # project folder (holds dataset/)

DATA = Path(os.environ.get("BER_DATA", _ROOT / "dataset"))          # dataset/{train,test}/*.tsv
WORK = Path(os.environ.get("BER_WORK", _ROOT / "work"))             # cached stage outputs
OUT = Path(os.environ.get("BER_OUT", _ROOT / "output"))             # submission files
SMOKE = os.environ.get("BER_SMOKE") == "1"                           # tiny subsample for bug-hunting


@dataclass(frozen=True)
class Params:
    emb_model: str = os.environ.get("BER_EMB_MODEL", "intfloat/multilingual-e5-small")
    emb_batch: int = 1024
    dense_k_fwd: int = 25        # per S1: nearest S2/S3 records
    dense_k_rev: int = 5         # per S2/S3 record: nearest S1 entities
    sparse_k: int = 25           # per S1: top records by IDF-weighted shared keys
    key_df_cap: int = 2000       # ignore keys shared by more S2/S3 records than this
    stop_df: float = 0.01        # name tokens in >= 1% of a country's names = generic/legal words
    landmark: bool = os.environ.get("BER_LANDMARK", "1") != "0"   # learned landmark-part removal
    lex_min_count: int = 20      # learned lexicon: min co-occurrences
    lex_min_ratio: float = 0.5   # learned lexicon: min P(target | source)
    cv_folds: int = 5
    neg_keep_rank: int = 10      # training: keep every negative ranked within this; subsample the rest
    neg_sample: float = 0.15     # ... at this rate (reweighted)
    seed: int = 13
    # Use at most 50% of the machine (user rule) so memory/CPU spikes can't crash a run.
    workers: int = max(1, (os.cpu_count() or 2) // 2)           # threads (rapidfuzz, LightGBM)
    pool_workers: int = max(1, (os.cpu_count() or 2) // 2)      # forked processes (memory-bound)


P = Params()


def _mem() -> str:
    try:   # Linux: current / total memory of the machine
        info = dict(line.split(":", 1) for line in open("/proc/meminfo"))
        total = int(info["MemTotal"].split()[0]) / 2**20
        avail = int(info["MemAvailable"].split()[0]) / 2**20
        return f" [mem {total - avail:.0f}/{total:.0f} GB]"
    except (OSError, KeyError, ValueError):
        return ""


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]{_mem()} {msg}", flush=True)


def work_dir(*parts) -> Path:
    p = WORK.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p
