"""Land source files in the immutable raw layer.

The raw layer is written once and never modified. Every later layer is
reproducible from it plus the configuration, which is what makes a wrong
transform a re-run rather than a data recovery exercise (docs/PRIMER.md §9).

This stage deliberately does not parse anything. It copies bytes and records a
SHA-256 per file, so that if a source file is ever silently replaced upstream we
find out rather than quietly producing different numbers.

Usage::

    python -m avionics_etl.extract.raw
    python -m avionics_etl.extract.raw --verify   # check digests, copy nothing
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from avionics_etl import config, lineage

#: Files that are not data but are worth keeping beside it — the source's own
#: documentation, which the audit cross-checks against the actual files.
COMPANION_PATTERNS = ("readme.txt", "*.pdf")


def source_files(cfg: config.Config) -> list[Path]:
    src = cfg.path("source")
    if not src.exists():
        raise FileNotFoundError(
            f"source directory not found: {src}\n"
            "Extract the source archive there, or set paths.source in configs/pipeline.yaml"
        )
    found: list[Path] = sorted(p for p in src.glob("*.txt") if p.is_file())
    for pattern in COMPANION_PATTERNS:
        found += [p for p in sorted(src.glob(pattern)) if p.is_file() and p not in found]
    if not found:
        raise FileNotFoundError(f"no source files found in {src}")
    return found


def stage_raw(cfg: config.Config, *, verify_only: bool = False) -> lineage.Lineage:
    dst = cfg.ensure("raw")
    rec = lineage.Lineage.start("raw", cfg.hash)

    copied, unchanged, drifted = 0, 0, []
    for src_file in source_files(cfg):
        target = dst / src_file.name
        src_digest = lineage.file_digest(src_file)

        if target.exists():
            if lineage.file_digest(target) == src_digest:
                unchanged += 1
            elif verify_only:
                drifted.append(src_file.name)
            else:
                # The raw layer is immutable, but the SOURCE changing is a real
                # event worth surfacing rather than silently overwriting.
                print(f"  source changed since last run: {src_file.name}")
                shutil.copy2(src_file, target)
                copied += 1
        elif not verify_only:
            shutil.copy2(src_file, target)
            copied += 1

        rec.add_source(target if target.exists() else src_file)

    if verify_only:
        if drifted:
            print(f"DRIFT: {len(drifted)} file(s) differ from the source: {drifted}")
            raise SystemExit(1)
        print(f"raw    : {unchanged} file(s) verified against source, no drift")
        return rec

    rec.notes.update({"copied": copied, "unchanged": unchanged})
    rec.finish(cfg.ensure("artifacts") / "manifests")
    print(f"raw    : {copied} copied, {unchanged} already current -> {dst}")
    return rec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verify", action="store_true", help="check digests without copying")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    stage_raw(config.load(args.config), verify_only=args.verify)
    return 0


if __name__ == "__main__":
    sys.exit(main())
