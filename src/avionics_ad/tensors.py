"""Turn tidy time series into the 3-D tensors the models consume.

Every model here, deep or classical, scores *windows* rather than individual
samples. A single telemetry sample carries almost no information — anomalies
live in how a signal behaves over a stretch of time, and a time-shift anomaly
is invisible at any single instant by definition.

The tensor is ``(B, W, P)``: batch, window length, parameters — the shape the
reference study specifies.

Two rules, both load-bearing:

**Windows never cross a group boundary.** A window spanning the end of one
asset and the start of another is a fabricated signal that no hardware ever
produced. Groups are windowed independently.

**Every window keeps an index back to its source rows.** Without it a score
cannot be attributed to a time, an asset or a mission, and a detection that
cannot be located is not actionable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class WindowSet:
    """Windows plus everything needed to trace them back.

    ``X`` is ``(B, W, P)``. ``end_index`` gives, for each window, the positional
    row index of its final sample — windows are labelled and located by where
    they end, because that is the "now" a model would be predicting from.
    """

    X: np.ndarray
    end_index: np.ndarray
    channels: list[str]
    window: int
    stride: int
    keys: pd.DataFrame | None = None

    def __len__(self) -> int:
        return len(self.X)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.X.shape)  # type: ignore[return-value]

    def labels_from(self, per_sample: np.ndarray, *, min_overlap: int = 1) -> np.ndarray:
        """Collapse per-sample ground truth onto these exact windows."""
        per_sample = np.asarray(per_sample, dtype=bool)
        out = np.zeros(len(self), dtype=bool)
        for i, end in enumerate(self.end_index):
            lo = end - self.window + 1
            out[i] = per_sample[lo : end + 1].sum() >= min_overlap
        return out

    def describe(self) -> str:
        b, w, p = self.shape
        return f"{b:,} windows of {w} steps x {p} channels (stride {self.stride})"


def make_windows(values: np.ndarray, *, window: int, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Slice a ``(T, P)`` array into ``(B, W, P)`` windows.

    Returns the tensor and the positional index of each window's final sample.
    Uses a strided view rather than copying, so memory stays proportional to the
    original series even at stride 1, where naive slicing would multiply it by
    the window length.
    """
    values = np.asarray(values, dtype="float32")
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError(f"expected a (T, P) array, got shape {values.shape}")
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    t, p = values.shape
    if t < window:
        return np.empty((0, window, p), dtype="float32"), np.empty(0, dtype="int64")

    n = 1 + (t - window) // stride
    s0, s1 = values.strides
    view = np.lib.stride_tricks.as_strided(
        values, shape=(n, window, p), strides=(s0 * stride, s0, s1), writeable=False
    )
    ends = np.arange(n) * stride + window - 1
    return np.ascontiguousarray(view), ends


def windows_from_frame(
    df: pd.DataFrame,
    channels: Sequence[str],
    *,
    window: int,
    stride: int = 1,
    group_cols: Sequence[str] = (),
) -> WindowSet:
    """Window a tidy frame, never crossing a group boundary.

    Rows are taken in their existing order within each group, so the caller is
    responsible for sorting by time first. That is deliberate: silently sorting
    here would hide an unsorted source rather than surfacing it.
    """
    channels = list(channels)
    missing = [c for c in [*channels, *group_cols] if c not in df.columns]
    if missing:
        raise KeyError(f"columns not present: {missing}")

    blocks: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    key_rows: list[pd.DataFrame] = []

    groups = [((), df)] if not group_cols else df.groupby(list(group_cols), sort=False)
    for keys, part in groups:
        positions = np.asarray(part.index if df.index.is_unique else range(len(part)))
        offsets = np.searchsorted(np.asarray(df.index), positions) if df.index.is_unique else None
        x, local_ends = make_windows(part[channels].to_numpy(), window=window, stride=stride)
        if len(x) == 0:
            continue
        blocks.append(x)

        absolute = np.asarray(part.index)[local_ends] if offsets is None else positions[local_ends]
        ends.append(np.asarray(absolute))

        if group_cols:
            keys = keys if isinstance(keys, tuple) else (keys,)
            key_rows.append(pd.DataFrame([dict(zip(group_cols, keys))] * len(x)))

    if not blocks:
        return WindowSet(
            X=np.empty((0, window, len(channels)), dtype="float32"),
            end_index=np.empty(0, dtype="int64"),
            channels=channels,
            window=window,
            stride=stride,
        )

    return WindowSet(
        X=np.concatenate(blocks),
        end_index=np.concatenate(ends),
        channels=channels,
        window=window,
        stride=stride,
        keys=pd.concat(key_rows, ignore_index=True) if key_rows else None,
    )


def flatten(X: np.ndarray) -> np.ndarray:
    """``(B, W, P)`` -> ``(B, W*P)``, for models that take flat feature vectors."""
    X = np.asarray(X)
    return X.reshape(len(X), -1)


def split_input_target(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split windows into forecasting inputs and their next-step targets.

    A forecasting model sees ``window-1`` steps and predicts the step after.
    Returns ``(B, W-1, P)`` and ``(B, P)``.
    """
    X = np.asarray(X)
    if X.shape[1] < 2:
        raise ValueError("need a window of at least 2 steps to forecast one ahead")
    return X[:, :-1, :], X[:, -1, :]
