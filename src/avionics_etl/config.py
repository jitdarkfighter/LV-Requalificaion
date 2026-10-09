"""Configuration loading and path resolution.

One entry point — :func:`load` — returns an immutable :class:`Config`. The parsed
configuration is hashed so that every output table can record exactly which
configuration produced it (see :mod:`avionics_etl.lineage`).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import yaml

# repo root = three levels up from this file: src/avionics_etl/config.py
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "pipeline.yaml"


@dataclass(frozen=True)
class Config:
    """Parsed pipeline configuration with resolved paths."""

    data: dict[str, Any]
    source_file: Path

    # -- access -------------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, dotted: str, default: Any = None) -> Any:
        """Fetch a nested value, e.g. ``cfg.get("labels.rul_clip")``."""
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    # -- paths --------------------------------------------------------------

    def path(self, name: str) -> Path:
        """Resolve a named path from the ``paths:`` block.

        ``~`` is expanded; relative paths resolve against the repository root so
        the pipeline behaves identically regardless of the working directory.
        """
        try:
            raw = self.data["paths"][name]
        except KeyError:
            raise KeyError(
                f"unknown path {name!r}; configured paths: "
                f"{sorted(self.data.get('paths', {}))}"
            ) from None
        p = Path(raw).expanduser()
        return p if p.is_absolute() else (REPO_ROOT / p)

    def ensure(self, name: str) -> Path:
        """Resolve a named path and create the directory if it does not exist."""
        p = self.path(name)
        p.mkdir(parents=True, exist_ok=True)
        return p

    # -- convenience --------------------------------------------------------

    @property
    def subsets(self) -> list[str]:
        return list(self.data["dataset"]["subsets"])

    @property
    def splits(self) -> list[str]:
        return list(self.data["dataset"]["splits"])

    @property
    def rul_clip(self) -> int:
        return int(self.data["labels"]["rul_clip"])

    # -- identity -----------------------------------------------------------

    @cached_property
    def hash(self) -> str:
        """Stable SHA-256 over the parsed config.

        Hashing the *parsed* structure rather than the file text means comments
        and formatting changes do not invalidate previously produced tables,
        while any change to an actual value does.
        """
        canonical = json.dumps(self.data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def load(path: str | Path | None = None) -> Config:
    """Load the pipeline configuration."""
    source = Path(path) if path else DEFAULT_CONFIG
    if not source.exists():
        raise FileNotFoundError(f"configuration not found: {source}")
    with source.open() as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{source} did not parse to a mapping")
    return Config(data=data, source_file=source)
