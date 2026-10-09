"""Model implementations and the selection registry.

Importing this package registers every model, so the registry is populated by
the time anything asks what is available.
"""

from avionics_ad.models import classical, forecast, reconstruct, tfa  # noqa: F401  (registration side effect)
from avionics_ad.models.base import AnomalyModel, Family, ParamSpec
from avionics_ad.models.registry import (
    ModelInfo, available, create, families, get, names, register, restore, snapshot,
)

__all__ = [
    "AnomalyModel",
    "Family",
    "ParamSpec",
    "ModelInfo",
    "available",
    "create",
    "families",
    "get",
    "names",
    "register",
    "restore",
    "snapshot",
]
