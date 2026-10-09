"""End-to-end entry point: data -> prepare -> embed -> block -> features -> train -> predict.

  python src/run_pipeline.py                     # all stages
  python src/run_pipeline.py block features      # selected stages (earlier outputs are cached)
Env: BER_STAGES (same as args), BER_SMOKE=1 (small subsample for a quick test), BER_DATA /
BER_WORK / BER_OUT (paths). Every stage skips work whose output exists.
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ber import block, embed, features, predict, prepare, train  # noqa: E402
from ber.config import DATA, OUT, SMOKE, WORK, log  # noqa: E402

STAGES = {
    "prepare": lambda: (prepare.run("train"), prepare.run("test")),
    "embed": lambda: (embed.run("train"), embed.run("test")),
    "block": lambda: (block.run("train"), block.run("test")),
    "features": lambda: (features.run("train"), features.run("test")),
    "train": train.run,
    "predict": predict.run,
}


def main(argv):
    wanted = [a for a in argv if not a.startswith("-")] or os.environ.get("BER_STAGES", "").split()
    wanted = wanted or list(STAGES)
    log(f"data={DATA} work={WORK} out={OUT} smoke={SMOKE} stages={wanted}")
    for name in STAGES:
        if name in wanted:
            t0 = time.time()
            STAGES[name]()
            log(f"== {name} done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main(sys.argv[1:])
