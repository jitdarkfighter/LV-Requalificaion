"""Model selection: the thing the dropdown reads.

One place maps a name to an implementation. Adding a model means writing the
class and registering it; nothing else in the pipeline, the benchmark or the UI
changes.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Any, Iterator

from avionics_ad.models.base import AnomalyModel, Family, ParamSpec

_REGISTRY: dict[str, type[AnomalyModel]] = {}


@dataclass(frozen=True)
class ModelInfo:
    """Everything a UI needs to offer a model and configure it."""

    name: str
    label: str
    family: Family
    description: str
    params_spec: tuple[ParamSpec, ...]


def register(cls: type[AnomalyModel]) -> type[AnomalyModel]:
    """Class decorator that adds a model to the registry."""
    if not issubclass(cls, AnomalyModel):
        raise TypeError(f"{cls.__name__} does not subclass AnomalyModel")
    if cls.name in _REGISTRY and _REGISTRY[cls.name] is not cls:
        raise ValueError(f"model name {cls.name!r} is already registered")
    _REGISTRY[cls.name] = cls
    return cls


def create(name: str, **params: Any) -> AnomalyModel:
    """Instantiate a registered model by name."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown model {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**params)


def get(name: str) -> type[AnomalyModel]:
    if name not in _REGISTRY:
        raise KeyError(f"unknown model {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def names() -> list[str]:
    return sorted(_REGISTRY)


def available(family: Family | None = None) -> list[ModelInfo]:
    """Registered models, optionally filtered by family. Sorted for a stable UI."""
    out = [
        ModelInfo(
            name=cls.name,
            label=cls.label,
            family=cls.family,
            # Dedent, then collapse each paragraph to one line. Class-body
            # indentation would otherwise render as a Markdown code block.
            description="\n\n".join(
                " ".join(para.split())
                for para in textwrap.dedent(cls.description).strip().split("\n\n")
            ),
            params_spec=cls.params_spec,
        )
        for cls in _REGISTRY.values()
        if family is None or cls.family == family
    ]
    return sorted(out, key=lambda m: (m.family, m.name))


def families() -> list[str]:
    return sorted({cls.family for cls in _REGISTRY.values()})


def snapshot() -> dict[str, type[AnomalyModel]]:
    """A copy of the current registry.

    Exists so a test that registers a throwaway model can put the registry back
    afterwards. Without it, registering a model in one test silently changes
    what every later test sees — which is how a passing suite starts depending
    on the order it runs in.
    """
    return dict(_REGISTRY)


def restore(state: dict[str, type[AnomalyModel]]) -> None:
    """Replace the registry with a previous snapshot."""
    _REGISTRY.clear()
    _REGISTRY.update(state)


def __iter__() -> Iterator[str]:
    return iter(sorted(_REGISTRY))
