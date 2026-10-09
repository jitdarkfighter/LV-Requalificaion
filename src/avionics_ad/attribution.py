"""Which channel was responsible for a detection.

A score saying "this window is anomalous" is not actionable on its own. An
engineer clearing a stage needs to know *which* parameter went wrong, because
that is what points at a subsystem. The reference study calls this feature
contribution analysis and uses it for root-cause attribution.

The method is simple and that is a virtue — it is explainable to someone who has
to sign off on the result:

1. take the per-channel error the model already produced
2. normalise it across channels so the contributions sum to 1
3. rank

What it can and cannot tell you
-------------------------------
It names the channel whose behaviour the model failed to predict. That is not
always the channel that *failed*: when a current surge drags the bus voltage
down, both channels misbehave and the voltage may well carry the larger error
even though the current caused it. Attribution narrows a 3,500-parameter search
to a handful; it does not replace the engineer.

Classical detectors collapse a window into one distance and genuinely cannot
decompose it. They return ``None`` and attribution reports that honestly rather
than fabricating a ranking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def normalise_contributions(feature_errors: np.ndarray) -> np.ndarray:
    """Scale per-channel errors so each window's contributions sum to 1.

    Normalising per window matters: raw error magnitudes vary enormously
    between a quiet stretch and an active one, so un-normalised numbers cannot
    be compared across detections.

    A window with no error at all gets a uniform split rather than a division by
    zero — with nothing to explain, no channel is more to blame than another.
    """
    e = np.asarray(feature_errors, dtype="float64")
    if e.ndim != 2:
        raise ValueError(f"expected (B, P) feature errors, got shape {e.shape}")
    total = e.sum(axis=1, keepdims=True)
    uniform = np.full_like(e, 1.0 / e.shape[1])
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(total > 0, e / np.where(total > 0, total, 1.0), uniform)
    return out


def rank_channels(
    feature_errors: np.ndarray,
    channels: list[str],
    *,
    window_index: int | None = None,
    top: int = 5,
) -> pd.DataFrame:
    """Rank channels by their share of the error.

    With ``window_index`` the ranking is for that single detection. Without it,
    contributions are averaged across every supplied window, which answers the
    broader question of which channel the model struggles with overall.
    """
    e = np.asarray(feature_errors, dtype="float64")
    if e.ndim != 2:
        raise ValueError(f"expected (B, P) feature errors, got shape {e.shape}")
    if e.shape[1] != len(channels):
        raise ValueError(
            f"channel count mismatch: errors have {e.shape[1]} columns, "
            f"{len(channels)} names given"
        )

    shares = normalise_contributions(e)
    if window_index is not None:
        if not 0 <= window_index < len(shares):
            raise IndexError(f"window_index {window_index} out of range for {len(shares)} windows")
        values = shares[window_index]
        raw = e[window_index]
    else:
        values = shares.mean(axis=0)
        raw = e.mean(axis=0)

    out = pd.DataFrame({"channel": channels, "contribution": values, "mean_error": raw})
    return out.sort_values("contribution", ascending=False).head(top).reset_index(drop=True)


def attribute(
    model,
    X: np.ndarray,
    channels: list[str],
    *,
    flagged: np.ndarray | None = None,
    top: int = 5,
) -> pd.DataFrame | None:
    """Attribute the flagged windows, or ``None`` if the model cannot decompose.

    Restricting to flagged windows is the point: averaging over everything
    dilutes the signal with the overwhelming majority of normal windows, where
    there is nothing to attribute.
    """
    errors = model.score_features(X)
    if errors is None:
        return None
    errors = np.asarray(errors, dtype="float64")
    if flagged is not None:
        flagged = np.asarray(flagged, dtype=bool)
        if flagged.shape != (len(X),):
            raise ValueError(f"flagged must be ({len(X)},), got {flagged.shape}")
        if not flagged.any():
            return None
        errors = errors[flagged]
    return rank_channels(errors, channels, top=top)


def summarise_by_kind(
    feature_errors: np.ndarray,
    channels: list[str],
    kinds: np.ndarray,
    *,
    top: int = 3,
) -> pd.DataFrame:
    """Top contributing channels per anomaly kind.

    Useful for checking attribution actually works: an anomaly injected into
    ``Force_1`` ought to attribute to ``Force_1``, and if it does not, either the
    model or the attribution is wrong.
    """
    kinds = np.asarray(kinds)
    if len(kinds) != len(feature_errors):
        raise ValueError("kinds must have one entry per window")

    rows = []
    for kind in sorted({k for k in kinds if k}):
        mask = kinds == kind
        ranked = rank_channels(np.asarray(feature_errors)[mask], channels, top=top)
        ranked.insert(0, "anomaly_kind", kind)
        rows.append(ranked)
    if not rows:
        return pd.DataFrame(columns=["anomaly_kind", "channel", "contribution", "mean_error"])
    return pd.concat(rows, ignore_index=True)
