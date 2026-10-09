"""Provenance stamping.

Every table the pipeline writes records where it came from: which source files,
which run, and which configuration. In an aerospace qualification context this
is not bookkeeping — an engineer signing off on a requalification decision has
to be able to trace the number back to the bytes it came from.

See docs/PRIMER.md section 9.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1


def file_digest(path: Path, *, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, read in bounded-memory chunks."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def new_run_id() -> str:
    """A sortable, human-readable run identifier."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}-{os.getpid():05d}"


@dataclass
class Lineage:
    """Provenance record for one pipeline stage."""

    stage: str
    run_id: str
    config_hash: str
    started_at: str
    schema_version: int = SCHEMA_VERSION
    host: str = field(default_factory=socket.gethostname)
    python: str = field(default_factory=platform.python_version)
    sources: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, int] = field(default_factory=dict)
    finished_at: str | None = None
    notes: dict[str, object] = field(default_factory=dict)

    @classmethod
    def start(cls, stage: str, config_hash: str) -> "Lineage":
        return cls(
            stage=stage,
            run_id=new_run_id(),
            config_hash=config_hash,
            started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    def add_source(self, path: Path) -> None:
        """Record a source file and its digest."""
        self.sources[path.name] = file_digest(path)

    def add_output(self, name: str, rows: int) -> None:
        self.outputs[name] = rows

    def stamp(self) -> dict[str, str]:
        """Columns appended to every written table.

        Deliberately small — three short strings per row, which dictionary-encode
        to almost nothing in Parquet (PRIMER section 7).
        """
        return {
            "_run_id": self.run_id,
            "_config_hash": self.config_hash,
            "_stage": self.stage,
        }

    def finish(self, manifest_dir: Path) -> Path:
        """Close the record and write it as a JSON manifest."""
        self.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        manifest_dir.mkdir(parents=True, exist_ok=True)
        out = manifest_dir / f"{self.stage}_{self.run_id}.json"
        out.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))

        # A stable pointer to the most recent run of this stage, so downstream
        # stages and tests can find it without globbing by timestamp.
        latest = manifest_dir / f"{self.stage}_latest.json"
        latest.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))
        return out
