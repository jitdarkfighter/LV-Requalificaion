"""Inject known anomalies, and record exactly where they are.

Why this is required rather than optional
-----------------------------------------
Recall, F-beta and false-negative rate are the metrics that matter here, and all
three need ground truth: you cannot measure how many anomalies a detector missed
without knowing how many were there. Neither C-MAPSS nor the synthetic telemetry
contains labelled anomalies, and real labelled launch-vehicle faults are exactly
what nobody has.

The reference study resolved this the same way — it simulated anomalies into
synthetic data and scored detection against the simulation. So injection *is*
the labelling step, not a convenience.

The three kinds
---------------
These are the study's three, and they are deliberately different in character,
because a detector can be good at one and blind to another:

``level_shift``
    The signal moves to a new value and stays there. Sustained, not transient —
    a power supply drooping, or a sensor whose calibration has drifted. Easiest
    to catch, since a plain limit check often sees it.

``pattern``
    The waveform itself changes: oscillation, noise burst, instability. The mean
    may barely move, so anything watching only the level will miss it.

``time_shift``
    The signal is *correct in value but late*. A command response arrives at
    t+11 instead of t+10. Amplitude statistics are entirely unchanged, so bound
    checks and level-based detectors cannot see it at all; only something
    modelling temporal structure can. In the reference study this is the one a
    plain LSTM failed to detect.

Injection breaks couplings, which is the point
----------------------------------------------
Shifting the current channel without touching voltage breaks the linear
relationship between them. Each channel stays individually plausible while the
*set* becomes inconsistent — which is the case that justifies multivariate
modelling over per-channel thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd

Kind = Literal["level_shift", "pattern", "time_shift"]

KINDS: tuple[Kind, ...] = ("level_shift", "pattern", "time_shift")

LABEL_COL = "is_anomaly"
KIND_COL = "anomaly_kind"
CHANNEL_COL = "anomaly_channel"


@dataclass(frozen=True)
class Anomaly:
    """One injected fault.

    ``magnitude`` is relative to the channel's own standard deviation over the
    run, so a single setting means the same severity on a 28 V rail and a 500 N
    actuator.
    """

    kind: Kind
    channel: str
    start: int
    end: int
    magnitude: float = 3.0
    shift: int = 25              # time_shift only: samples of delay
    frequency: float = 0.12      # pattern only: cycles per sample

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown anomaly kind {self.kind!r}; expected one of {KINDS}")
        if self.end <= self.start:
            raise ValueError(f"end must be after start, got {self.start}..{self.end}")


def _apply_level_shift(v: np.ndarray, a: Anomaly, scale: float) -> np.ndarray:
    out = v.copy()
    out[a.start : a.end] += a.magnitude * scale
    return out


def _apply_pattern(v: np.ndarray, a: Anomaly, scale: float, rng: np.random.Generator) -> np.ndarray:
    """Superimpose an oscillation plus noise, leaving the mean roughly intact.

    The mean is preserved on purpose: it is what makes this invisible to
    level-based detection and forces a model to look at shape.
    """
    out = v.copy()
    n = a.end - a.start
    k = np.arange(n)
    oscillation = a.magnitude * scale * np.sin(2 * np.pi * a.frequency * k)
    burst = rng.normal(0.0, a.magnitude * scale * 0.25, n)
    out[a.start : a.end] += oscillation + burst
    return out


def _apply_time_shift(v: np.ndarray, a: Anomaly) -> np.ndarray:
    """Delay the signal within the window — correct values, wrong time.

    The window is filled with what the channel was doing ``shift`` samples
    earlier. Amplitude statistics over the window are essentially unchanged,
    which is exactly why this kind is hard.
    """
    out = v.copy()
    s = max(int(a.shift), 1)
    src_lo = max(a.start - s, 0)
    src_hi = max(a.end - s, 0)
    segment = v[src_lo:src_hi]
    if len(segment) == 0:
        return out
    if len(segment) < (a.end - a.start):
        segment = np.pad(segment, (a.end - a.start - len(segment), 0), mode="edge")
    out[a.start : a.end] = segment[: a.end - a.start]
    return out


def inject(
    df: pd.DataFrame,
    anomalies: Sequence[Anomaly],
    *,
    seed: int = 0,
    label_col: str = LABEL_COL,
) -> pd.DataFrame:
    """Return a copy of ``df`` with the anomalies applied and labelled.

    Adds three columns: a boolean ``is_anomaly`` per sample, and the kind and
    channel responsible (empty string where there is no anomaly). The frame's
    index and row count are unchanged, so labels line up with the original.
    """
    rng = np.random.default_rng(seed)
    out = df.copy().reset_index(drop=True)
    n = len(out)

    labels = np.zeros(n, dtype=bool)
    kinds = np.array([""] * n, dtype=object)
    channels = np.array([""] * n, dtype=object)

    for a in anomalies:
        if a.channel not in out.columns:
            raise KeyError(f"channel {a.channel!r} not in frame")
        if a.start < 0 or a.end > n:
            raise ValueError(f"anomaly window {a.start}..{a.end} outside data of length {n}")

        v = out[a.channel].to_numpy(dtype="float64")
        scale = float(np.std(v)) or 1.0

        if a.kind == "level_shift":
            v = _apply_level_shift(v, a, scale)
        elif a.kind == "pattern":
            v = _apply_pattern(v, a, scale, rng)
        else:
            v = _apply_time_shift(v, a)

        out[a.channel] = v
        labels[a.start : a.end] = True
        kinds[a.start : a.end] = a.kind
        channels[a.start : a.end] = a.channel

    out[label_col] = labels
    out[KIND_COL] = kinds
    out[CHANNEL_COL] = channels
    return out


def most_active_window(
    values: np.ndarray, *, width: int, search_from: float = 0.5
) -> int:
    """Start index of the ``width``-sample window where the signal moves most.

    Needed because a time shift is only meaningful where there is something to
    shift. Delaying a constant signal changes nothing at all — the data is
    byte-identical — so an anomaly placed in a flat stretch is not a hard
    anomaly, it is an absent one, and scoring detectors against it would be
    measuring nothing.
    """
    v = np.asarray(values, dtype="float64")
    n = len(v)
    if n < width + 2:
        return 0
    lo = int(n * search_from)
    best_start, best_score = lo, -1.0
    for start in range(lo, n - width, max(width // 10, 1)):
        score = float(np.std(np.diff(v[start : start + width])))
        if score > best_score:
            best_start, best_score = start, score
    return best_start


def default_anomalies(
    n_samples: int,
    channels: Sequence[str],
    *,
    width: int = 300,
    frame: pd.DataFrame | None = None,
) -> list[Anomaly]:
    """One of each kind, on three different channels, well separated.

    Placed in the back half of the run so a model trained on an early,
    anomaly-free stretch has clean data to learn from.

    When ``frame`` is supplied, the time shift is positioned where its channel
    is actually changing (see :func:`most_active_window`). Without it the
    placement is positional only, which risks landing on a flat stretch where a
    time shift has no effect whatsoever.
    """
    if len(channels) < 3:
        raise ValueError("need at least three channels to place one anomaly of each kind")

    positions = {"level_shift": 0.55, "pattern": 0.70, "time_shift": 0.85}
    out: list[Anomaly] = []
    for kind, channel in zip(KINDS, channels):
        if kind == "time_shift" and frame is not None and channel in frame.columns:
            start = most_active_window(frame[channel].to_numpy(), width=width)
        else:
            start = int(n_samples * positions[kind])
        start = min(start, max(n_samples - width - 1, 0))
        out.append(Anomaly(kind=kind, channel=channel, start=start, end=start + width))
    return out


def window_labels(
    labels: np.ndarray,
    *,
    window: int,
    stride: int = 1,
    min_overlap: int = 1,
) -> np.ndarray:
    """Collapse per-sample labels onto the windows a model actually scores.

    A window counts as anomalous when at least ``min_overlap`` of its samples
    are. The default of 1 is deliberate: a detector that flags the window
    containing the onset of a fault has found it, and demanding majority overlap
    would score that as a miss.
    """
    labels = np.asarray(labels, dtype=bool)
    if window > len(labels):
        return np.zeros(0, dtype=bool)
    ends = range(window - 1, len(labels), stride)
    return np.array(
        [labels[e - window + 1 : e + 1].sum() >= min_overlap for e in ends], dtype=bool
    )


def summarise(df: pd.DataFrame, *, label_col: str = LABEL_COL) -> pd.DataFrame:
    """How much of the data each injected fault covers."""
    if label_col not in df.columns:
        raise KeyError(f"{label_col} not present; was inject() run?")
    anomalous = df[df[label_col]]
    if anomalous.empty:
        return pd.DataFrame(columns=["anomaly_kind", "anomaly_channel", "samples", "fraction"])
    out = (
        anomalous.groupby([KIND_COL, CHANNEL_COL], sort=True)
        .size()
        .rename("samples")
        .reset_index()
    )
    out["fraction"] = (out.samples / len(df)).round(4)
    return out
