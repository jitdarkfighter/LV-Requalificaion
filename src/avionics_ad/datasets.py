"""Assemble benchmark datasets: clean training windows, and a test set with
known anomalies.

The experimental design matters more than it first appears
----------------------------------------------------------
An early version trained on the first 40% of a run and tested on the rest. Every
model scored recall 1.00 and precision 0.22 — because the run is event-driven,
so the back half contains operating regimes the front half never showed. The
models were correctly reporting that powered operation looks nothing like idle;
they just had no way to know that was normal.

The fix is to train on a **complete clean run** and score an **injected copy of
the same run**. Training then covers every regime the test contains, so a
detection means "this is not normal behaviour" rather than "I have not seen this
part of the test before". The only difference between the two is the injected
faults, which is what makes the comparison clean.

Scaling is fitted on training data only and applied read-only to the test set,
exactly as in the ETL pipeline and for the same reason. Min-max is the default
because the reference study uses it: absolute level carries the spatial
relationships between parameters, and bounded inputs train more stably.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from avionics_ad import tensors as T
from avionics_ad.anomalies import inject as I
from avionics_etl.synthetic import lv_telemetry as L

DEFAULT_WINDOW = 100
DEFAULT_STRIDE = 5


@dataclass
class Dataset:
    """Everything an experiment needs, with the ground truth attached."""

    name: str
    train: np.ndarray                 # (B, W, P) clean
    test: np.ndarray                  # (B, W, P) containing anomalies
    labels: np.ndarray                # (B,) bool
    kinds: np.ndarray                 # (B,) which kind, '' where normal
    channels: list[str]
    window: int
    stride: int
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.test)

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def anomaly_rate(self) -> float:
        return float(self.labels.mean())

    def describe(self) -> str:
        return (
            f"{self.name}: {len(self.train):,} train / {len(self.test):,} test windows, "
            f"{self.n_channels} channels, window {self.window}, "
            f"{self.labels.sum():,} anomalous ({self.anomaly_rate:.1%})"
        )


def _scale(train: pd.DataFrame, test: pd.DataFrame, channels: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Min-max, fitted on train, applied to both.

    Test values outside the training range are clipped rather than allowed past
    [0, 1]: the deep models assume bounded inputs, and an unclipped outlier
    would be detected by its magnitude alone rather than by its behaviour.
    """
    lo = train[channels].min()
    span = (train[channels].max() - lo).replace(0.0, 1.0)
    out = []
    for frame in (train, test):
        scaled = frame.copy()
        scaled[channels] = ((frame[channels] - lo) / span).clip(0.0, 1.0)
        out.append(scaled)
    return out[0], out[1]


def build_lv_synthetic(
    *,
    window: int = DEFAULT_WINDOW,
    stride: int = DEFAULT_STRIDE,
    n_samples: int = 9000,
    seed: int = 0,
    anomaly_width: int = 300,
    profile: str = "checkout",
) -> Dataset:
    """Synthetic launch-vehicle checkout telemetry with three injected faults."""
    builder = L.default_profile if profile == "checkout" else L.power_only_profile
    clean = L.generate(builder("benchmark", n_samples), seed=seed)
    channels = list(L.CHANNEL_NAMES)

    targets = ["Battery_Voltage", "Force_1", "RMS_Current_1"]
    anomalies = I.default_anomalies(len(clean), targets, width=anomaly_width, frame=clean)
    dirty = I.inject(clean, anomalies, seed=seed + 1)

    train_df, test_df = _scale(clean, dirty, channels)

    train = T.windows_from_frame(train_df, channels, window=window, stride=stride)
    test = T.windows_from_frame(test_df, channels, window=window, stride=stride)
    labels = test.labels_from(dirty[I.LABEL_COL].to_numpy())

    kinds = np.array([""] * len(test), dtype=object)
    per_sample_kind = dirty[I.KIND_COL].to_numpy()
    for i, end in enumerate(test.end_index):
        present = {k for k in per_sample_kind[max(end - window + 1, 0) : end + 1] if k}
        if present:
            kinds[i] = sorted(present)[0]

    return Dataset(
        name="lv_synthetic",
        train=train.X,
        test=test.X,
        labels=labels,
        kinds=kinds,
        channels=channels,
        window=window,
        stride=stride,
        meta={
            "source": "synthetic launch-vehicle checkout telemetry",
            "n_samples": n_samples,
            "profile": profile,
            "anomalies": [
                {"kind": a.kind, "channel": a.channel, "start": a.start, "end": a.end}
                for a in anomalies
            ],
        },
    )


def build_cmapss(
    *,
    window: int = 50,
    stride: int = 5,
    source_id: str = "FD001",
    n_train_assets: int = 40,
    n_test_assets: int = 20,
    seed: int = 0,
) -> Dataset:
    """C-MAPSS engine telemetry, with the same three fault kinds injected.

    Included to prove the platform is not tied to one source. C-MAPSS has no
    labelled anomalies — it is a degradation dataset, not an anomaly one — so
    faults are injected exactly as for the synthetic telemetry. What carries
    over is the shape of the problem: multivariate, non-periodic, event-driven.
    """
    from avionics_etl import config
    from avionics_etl.canonical import model
    from avionics_etl.load import parquet_writer

    cfg = config.load()
    silver = cfg.path("silver") / "measurements"
    if not silver.exists():
        raise FileNotFoundError(
            f"{silver} not found. Run `make all` to build the C-MAPSS layers first."
        )

    df = parquet_writer.read_partitioned(silver)
    df = df[(df[model.SOURCE_ID] == source_id) & df[model.TRAJECTORY_COMPLETE]]
    channels = [c for c in df.columns if c.endswith("_z") and df[c].nunique() > 2]
    if not channels:
        raise ValueError(f"no usable channels for {source_id}")

    assets = sorted(df[model.ASSET_UID].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(assets)
    train_assets = assets[:n_train_assets]
    test_assets = assets[n_train_assets : n_train_assets + n_test_assets]
    if not test_assets:
        raise ValueError("not enough assets to form a test split")

    sort_by = [model.ASSET_UID, model.SEQUENCE_NO]
    train_df = df[df[model.ASSET_UID].isin(train_assets)].sort_values(sort_by).reset_index(drop=True)
    test_df = df[df[model.ASSET_UID].isin(test_assets)].sort_values(sort_by).reset_index(drop=True)

    # Inject one fault per kind, each into a different engine, placed late in
    # its trajectory where there is activity to disturb.
    anomalies = []
    for i, (kind, asset) in enumerate(zip(I.KINDS, test_assets)):
        rows = np.flatnonzero(test_df[model.ASSET_UID].to_numpy() == asset)
        if len(rows) < window * 2:
            continue
        channel = channels[i % len(channels)]
        width = max(window // 2, 20)
        local = I.most_active_window(test_df[channel].to_numpy()[rows], width=width)
        start = int(rows[0] + local)
        anomalies.append(I.Anomaly(kind=kind, channel=channel, start=start, end=start + width))

    dirty = I.inject(test_df, anomalies, seed=seed + 1) if anomalies else test_df.assign(
        **{I.LABEL_COL: False, I.KIND_COL: "", I.CHANNEL_COL: ""}
    )

    train_scaled, test_scaled = _scale(train_df, dirty, channels)
    train = T.windows_from_frame(train_scaled, channels, window=window, stride=stride,
                                 group_cols=[model.ASSET_UID])
    test = T.windows_from_frame(test_scaled, channels, window=window, stride=stride,
                                group_cols=[model.ASSET_UID])
    labels = test.labels_from(dirty[I.LABEL_COL].to_numpy())

    kinds = np.array([""] * len(test), dtype=object)
    per_sample_kind = dirty[I.KIND_COL].to_numpy()
    for i, end in enumerate(test.end_index):
        present = {k for k in per_sample_kind[max(end - window + 1, 0) : end + 1] if k}
        if present:
            kinds[i] = sorted(present)[0]

    return Dataset(
        name=f"cmapss_{source_id.lower()}",
        train=train.X,
        test=test.X,
        labels=labels,
        kinds=kinds,
        channels=channels,
        window=window,
        stride=stride,
        meta={
            "source": f"C-MAPSS {source_id}, regime-normalised",
            "train_assets": len(train_assets),
            "test_assets": len(test_assets),
            "anomalies": [
                {"kind": a.kind, "channel": a.channel, "start": a.start, "end": a.end}
                for a in anomalies
            ],
        },
    )


#: Everything the UI and the benchmark can offer.
BUILDERS: dict[str, Callable[..., Dataset]] = {
    "lv_synthetic": build_lv_synthetic,
    "cmapss": build_cmapss,
}


def validate(ds: Dataset) -> Dataset:
    """Raise a clear ValueError if a dataset cannot support an experiment.

    Catches the cases that otherwise surface as an obscure failure deep inside a
    model, or worse, as a metric computed from nothing: a window longer than the
    series, and a test set with no anomalies to find.
    """
    if len(ds.train) == 0:
        raise ValueError(
            f"{ds.name}: no training windows at window={ds.window}, stride={ds.stride}. "
            "The window is longer than the (clean) series; use a smaller window."
        )
    if len(ds.test) == 0:
        raise ValueError(
            f"{ds.name}: no test windows at window={ds.window}, stride={ds.stride}. "
            "The window is longer than the test series; use a smaller window."
        )
    if not ds.labels.any():
        raise ValueError(
            f"{ds.name}: no test window overlaps an injected anomaly at "
            f"window={ds.window}, stride={ds.stride}, so recall is undefined. "
            "Use a smaller window or stride, or a dataset with more test data."
        )
    return ds


def build(name: str, **kwargs) -> Dataset:
    if name not in BUILDERS:
        raise KeyError(f"unknown dataset {name!r}; available: {sorted(BUILDERS)}")
    return validate(BUILDERS[name](**kwargs))


def readiness(name: str) -> str | None:
    """None when a dataset can be built now, otherwise the real reason it cannot.

    Distinguishes a missing configuration from layers that simply were not built,
    because the remedy differs.
    """
    if name not in BUILDERS:
        return f"unknown dataset {name!r}"
    if name != "cmapss":
        return None
    from avionics_etl import config

    try:
        cfg = config.load()
    except FileNotFoundError as exc:
        return f"{exc}. The C-MAPSS ETL needs configs/pipeline.yaml and configs/channels.yaml."
    silver = cfg.path("silver") / "measurements"
    if not silver.exists():
        return f"C-MAPSS silver layer not built ({silver} missing). Run `make all`."
    return None


def available() -> list[str]:
    return sorted(BUILDERS)
