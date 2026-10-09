"""The ETL to ML handoff.

**This module is the interface.** If you are on the modelling side, this is the
only thing you need to import, and everything upstream of it is an
implementation detail you can ignore.

    from avionics_etl.serve import contract

    train = contract.load(fold="train")
    val   = contract.load(fold="validation")

    model.fit(train.X, train.y)
    preds = model.predict(val.X)

Three guarantees, all structural rather than conventional:

**1. Identifiers cannot reach the model.**  ``load`` returns ``ids``, ``X`` and
``y`` as separate objects. ``X`` contains features and nothing else. In
particular ``sequence_no`` is absent — it is algebraically part of the RUL
label, and a model given it will partially invert the formula instead of
learning degradation. You cannot accidentally include it, because it is not
there.

**2. No feature looks into the future.**  Every rolling statistic is
backward-looking and grouped by asset. A feature computed here could be computed
in real time, mid-flight, from history alone.

**3. Splits are by asset.**  No asset appears in two folds. Rows within a
trajectory are near-duplicates of each other; a random row split would let the
model recognise rather than predict.

The folds:

``train``       complete trajectories, for fitting
``validation``  complete trajectories held out, for model selection
``holdout``     incomplete trajectories — assets that never failed. This is the
                real evaluation set, and the real-world case: a recovered
                avionics box has not failed either.

Normalisation statistics and any fitted reduction are computed on ``train``
only, persisted, and applied read-only. **Do not refit them.** If you rescale
the data on your side, your transform and ours will disagree and neither of us
will notice for a week.

See docs/PRIMER.md section 8.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from avionics_etl import config
from avionics_etl.canonical import model
from avionics_etl.load import parquet_writer

FEATURES = "features"

#: Folds available to :func:`load`.
TRAIN = "train"
VALIDATION = "validation"
HOLDOUT = "holdout"


@dataclass
class Dataset:
    """Features, labels and identifiers — deliberately kept apart.

    ``ids`` is carried so predictions can be joined back to assets and written to
    the warehouse. It is never passed to a model.
    """

    ids: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series | None
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.X)

    @property
    def feature_names(self) -> list[str]:
        return list(self.X.columns)

    @property
    def assets(self) -> np.ndarray:
        return self.ids[model.ASSET_UID].unique()

    def describe(self) -> str:
        parts = [
            f"{len(self):,} rows",
            f"{len(self.feature_names)} features",
            f"{len(self.assets):,} assets",
        ]
        if self.y is not None:
            parts.append(f"RUL {self.y.min()}-{self.y.max()}")
        return ", ".join(parts)

    def to_sequences(self, *, length: int, stride: int = 1) -> "Sequences":
        """Reshape into fixed-length windows for a sequence model.

        A window ending at row *t* contains rows *t-length+1 .. t*, **all from
        the same asset**, and carries the label at *t*. Windows never span two
        assets, and never extend past the end of available history.

        Windowing happens here rather than in storage on purpose. Materialising
        every window would multiply the dataset by ``length``; doing it at load
        time keeps ``length`` a model hyperparameter you can sweep freely
        instead of a pipeline rebuild you have to request.
        """
        return build_sequences(self, length=length, stride=stride)


@dataclass
class Sequences:
    """Windowed view of a :class:`Dataset`, ready for an LSTM/TCN/Transformer."""

    ids: pd.DataFrame       # one row per window: the asset and end position
    X: np.ndarray           # (n_windows, length, n_features)
    y: np.ndarray | None    # (n_windows,)
    feature_names: list[str]

    def __len__(self) -> int:
        return len(self.X)

    def describe(self) -> str:
        return f"{len(self):,} windows of shape {self.X.shape[1:]} from {len(self.ids[model.ASSET_UID].unique()):,} assets"


def build_sequences(dataset: Dataset, *, length: int, stride: int = 1) -> Sequences:
    if length < 1:
        raise ValueError(f"length must be >= 1, got {length}")

    order = dataset.ids.sort_values([model.ASSET_UID, model.SEQUENCE_NO]).index
    ids = dataset.ids.loc[order].reset_index(drop=True)
    values = dataset.X.loc[order].to_numpy(dtype="float32")
    labels = dataset.y.loc[order].to_numpy() if dataset.y is not None else None

    blocks: list[np.ndarray] = []
    label_rows: list[float] = []
    end_rows: list[int] = []

    for _, group in ids.groupby(model.ASSET_UID, sort=False):
        idx = group.index.to_numpy()
        if len(idx) < length:
            # Too short to form a full window. Padding would invent history that
            # the asset does not have, so it is skipped and reported instead.
            continue
        for end in range(length - 1, len(idx), stride):
            rows = idx[end - length + 1 : end + 1]
            blocks.append(values[rows])
            end_rows.append(idx[end])
            if labels is not None:
                label_rows.append(labels[idx[end]])

    if not blocks:
        raise ValueError(
            f"no asset has at least {length} observations; reduce the window length"
        )

    return Sequences(
        ids=ids.loc[end_rows].reset_index(drop=True),
        X=np.stack(blocks),
        y=np.asarray(label_rows) if labels is not None else None,
        feature_names=dataset.feature_names,
    )


def load(
    *,
    fold: str | None = None,
    source_id: str | list[str] | None = None,
    features: list[str] | None = None,
    label: str = "rul",
    cfg: config.Config | None = None,
) -> Dataset:
    """Load an analysis-ready dataset.

    Parameters
    ----------
    fold
        ``"train"``, ``"validation"``, ``"holdout"``, or ``None`` for everything.
    source_id
        Restrict to one or more sources, e.g. ``"FD004"``.
    features
        Restrict to a subset of feature columns. Names outside the legal feature
        set are rejected rather than silently dropped.
    label
        ``"rul"`` (clipped — the training target) or ``"rul_raw"`` (unclipped,
        for analysis). ``None`` returns no labels.
    """
    cfg = cfg or config.load()
    root = cfg.path("gold") / FEATURES
    if not root.exists():
        raise FileNotFoundError(f"{root} not found. Run `make gold` first.")

    df = parquet_writer.read_partitioned(root)

    if fold is not None:
        if fold not in (TRAIN, VALIDATION, HOLDOUT):
            raise ValueError(f"unknown fold {fold!r}; expected train/validation/holdout")
        df = df[df.fold == fold]
    if source_id is not None:
        wanted = [source_id] if isinstance(source_id, str) else list(source_id)
        df = df[df[model.SOURCE_ID].isin(wanted)]
    if df.empty:
        raise ValueError(f"no rows for fold={fold!r}, source_id={source_id!r}")

    legal = model.feature_columns(df.columns)
    if features is not None:
        illegal = [f for f in features if f not in legal]
        if illegal:
            raise KeyError(
                f"not usable as features: {illegal[:5]}. They are identifiers, "
                f"labels or lineage columns — see docs/PRIMER.md section 8."
            )
        legal = list(features)

    ids = df[[c for c in model.IDENTIFIER_COLS if c in df.columns] + ["fold"]].reset_index(drop=True)
    X = df[legal].reset_index(drop=True)
    y = df[label].reset_index(drop=True) if label and label in df.columns else None

    # Belt and braces: the policy is defined in one place, and asserted here.
    assert model.SEQUENCE_NO not in X.columns, "sequence_no must never be a feature"

    return Dataset(
        ids=ids,
        X=X,
        y=y,
        meta={
            "fold": fold,
            "source_id": source_id,
            "label": label,
            "rul_clip": cfg.rul_clip,
            "n_features": len(legal),
        },
    )


def available(cfg: config.Config | None = None) -> pd.DataFrame:
    """What is on offer: rows, assets and features per fold and source."""
    cfg = cfg or config.load()
    root = cfg.path("gold") / FEATURES
    if not root.exists():
        raise FileNotFoundError(f"{root} not found. Run `make gold` first.")
    df = parquet_writer.read_partitioned(root)
    return (
        df.groupby([model.SOURCE_ID, "fold"], sort=True)
        .agg(rows=(model.ASSET_UID, "size"), assets=(model.ASSET_UID, "nunique"))
        .reset_index()
    )
