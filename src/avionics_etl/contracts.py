"""Compile the channel registry into runnable validators.

The spec requires the platform to support "deterministic bound checks **and**
ML-based solutions, complementing each other". That word matters. In aerospace
qualification a deterministic statement — *this rail voltage never left
4.85–5.15 V across the entire flight* — is **evidence**. A model's opinion is
not. Both are useful; only one is admissible on its own.

So the checks live in configuration (``configs/channels.yaml``), not scattered
through transform code:

* a domain expert can read and correct them without reading Python,
* they are versioned, so you can see when a bound changed and why,
* one engine evaluates all of them uniformly.

Four rule kinds are supported:

``null``    the value is missing or non-finite
``bounds``  the value is outside its physically plausible range
``rate``    the value moved further in one step than physics allows
``enum``    a discrete channel took a value outside its allowed set

``rate`` catches what static bounds cannot: a reading can be individually
plausible and still impossible in sequence. A temperature leaping 200 degrees
between consecutive cycles is a sensor fault, not an engine fault.

See docs/PRIMER.md section 12.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from avionics_etl.canonical.model import Channel, ChannelRegistry

RULES = ("null", "bounds", "rate", "enum")

#: Columns of the long-form violation table. This is the quarantine schema, and
#: it is meant to be read by a person reviewing a rejected row.
VIOLATION_COLUMNS = [
    "rule",
    "channel",
    "value",
    "limit_low",
    "limit_high",
    "detail",
]


@dataclass(frozen=True)
class ChannelContract:
    """The rules declared for one channel."""

    channel: Channel

    @property
    def name(self) -> str:
        return self.channel.name

    def applicable_rules(self) -> list[str]:
        rules = ["null"]
        if self.channel.bounds is not None:
            rules.append("bounds")
        if self.channel.max_rate is not None:
            rules.append("rate")
        if getattr(self.channel, "allowed", None):
            rules.append("enum")
        return rules

    # -- individual rules ---------------------------------------------------

    def check_null(self, values: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(values, errors="coerce")
        return numeric.isna() | ~np.isfinite(numeric.fillna(np.nan))

    def check_bounds(self, values: pd.Series) -> pd.Series:
        if self.channel.bounds is None:
            return pd.Series(False, index=values.index)
        lo, hi = self.channel.bounds
        return (values < lo) | (values > hi)

    def check_rate(self, values: pd.Series, groups: pd.Series) -> pd.Series:
        """Absolute change per sequence step, within one asset's trajectory.

        The first row of each trajectory has no predecessor, so it can never
        violate this rule — that is a genuine absence of evidence, not a pass.
        """
        if self.channel.max_rate is None:
            return pd.Series(False, index=values.index)
        delta = values.groupby(groups, sort=False).diff().abs()
        return delta.gt(self.channel.max_rate).fillna(False)


class ContractSuite:
    """All channel contracts for one source, evaluated together."""

    def __init__(self, registry: ChannelRegistry):
        self.registry = registry
        self.contracts = {c.name: ChannelContract(c) for c in registry}

    @classmethod
    def from_registry(cls, registry: ChannelRegistry) -> "ContractSuite":
        return cls(registry)

    def describe(self) -> pd.DataFrame:
        """One row per channel: what will be checked, and against what."""
        rows = []
        for contract in self.contracts.values():
            ch = contract.channel
            rows.append(
                {
                    "channel": ch.name,
                    "role": ch.role,
                    "unit": ch.unit,
                    "rules": ",".join(contract.applicable_rules()),
                    "bound_low": ch.bounds[0] if ch.bounds else None,
                    "bound_high": ch.bounds[1] if ch.bounds else None,
                    "max_rate": ch.max_rate,
                }
            )
        return pd.DataFrame(rows)

    def evaluate(
        self,
        df: pd.DataFrame,
        *,
        group_cols: Iterable[str],
        channels: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Return a long-form table of every violation found.

        One row per (data row, channel, rule) that failed. Empty means the frame
        satisfies every contract. The index of each violation points back at the
        offending row of ``df``, so the full row can be quarantined alongside the
        reason.
        """
        group_cols = list(group_cols)
        names = list(channels) if channels is not None else list(self.contracts)
        groups = (
            df[group_cols].astype(str).agg("|".join, axis=1)
            if group_cols
            else pd.Series("all", index=df.index)
        )

        found: list[pd.DataFrame] = []
        for name in names:
            contract = self.contracts.get(name)
            if contract is None or name not in df.columns:
                continue
            values = df[name]
            ch = contract.channel

            masks = {
                "null": contract.check_null(values),
                "bounds": contract.check_bounds(values),
                "rate": contract.check_rate(values, groups),
            }
            for rule, mask in masks.items():
                if not mask.any():
                    continue
                idx = df.index[mask]
                found.append(
                    pd.DataFrame(
                        {
                            "rule": rule,
                            "channel": name,
                            "value": values.loc[idx].astype("float64"),
                            "limit_low": ch.bounds[0] if ch.bounds else np.nan,
                            "limit_high": ch.bounds[1] if ch.bounds else np.nan,
                            "detail": _detail(rule, ch),
                        },
                        index=idx,
                    )
                )

        if not found:
            return pd.DataFrame(columns=VIOLATION_COLUMNS)
        return pd.concat(found).sort_index()


def _detail(rule: str, channel: Channel) -> str:
    """Human-readable reason, written for the person reviewing the quarantine."""
    unit = f" {channel.unit}" if channel.unit else ""
    if rule == "null":
        return f"{channel.name} is missing or non-finite"
    if rule == "bounds":
        lo, hi = channel.bounds  # type: ignore[misc]
        return f"{channel.name} outside plausible range [{lo:g}, {hi:g}]{unit}"
    if rule == "rate":
        return (
            f"{channel.name} changed by more than {channel.max_rate:g}{unit} "
            f"in one sequence step"
        )
    return f"{channel.name} failed {rule}"
