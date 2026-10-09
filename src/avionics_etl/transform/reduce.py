"""Dimensionality reduction.

Trend features multiply quickly — channels x windows x statistics — and they are
heavily correlated with one another, because a 5-cycle mean and a 10-cycle mean
of the same sensor are largely the same number. That correlation is not free: it
inflates storage, slows training, and makes any per-feature interpretation
misleading.

Two reductions are offered, and they answer different questions.

**Correlation pruning** — drop features that are near-duplicates of one already
kept. Cheap, and crucially it *keeps real columns*, so the output is still
interpretable: "the 10-cycle mean of T50" means something to an engineer in a
way that "component 3" does not. For a requalification tool that has to justify
its decisions, that matters.

**PCA** (Principal Component Analysis) — re-express the features as a smaller
set of uncorrelated combinations ordered by how much variance they explain.
Compresses harder, at the cost of interpretability.

Both are **fitted on the training fold only** and persisted, exactly like
normalisation. Fitting on everything would leak the evaluation set's covariance
structure into the transform (docs/PRIMER.md section 8c).

Neither is applied by default. Tree models generally prefer the full feature set
and handle correlation fine; the reduction is there for sequence models, for
plotting, and because the spec asks for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PRUNE_FILE = "feature_prune.json"
PCA_FILE = "feature_pca.npz"


# ---------------------------------------------------------------------------
# correlation pruning
# ---------------------------------------------------------------------------

def fit_prune(features: pd.DataFrame, *, threshold: float = 0.98) -> list[str]:
    """Columns to drop because a retained column already carries them.

    Walks the columns in order, keeping the first of any near-duplicate group.
    Deterministic, and it prefers earlier columns — which, given how
    :mod:`~avionics_etl.transform.features` orders its output, means level
    features are preferred over trend features and short windows over long ones.
    """
    numeric = features.select_dtypes("number")
    if numeric.shape[1] < 2:
        return []

    corr = numeric.corr().abs().to_numpy(copy=True)
    np.fill_diagonal(corr, 0.0)
    names = list(numeric.columns)

    dropped: set[str] = set()
    for i, name in enumerate(names):
        if name in dropped:
            continue
        duplicates = np.where(corr[i] >= threshold)[0]
        for j in duplicates:
            if j > i:
                dropped.add(names[j])
    return [n for n in names if n in dropped]


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

@dataclass
class PCAModel:
    """A fitted PCA, stored as plain arrays rather than a pickled estimator.

    Pickles bind the artifact to a library version; arrays do not. A transform
    this pipeline produced two years ago should still be applicable.
    """

    mean: np.ndarray
    components: np.ndarray          # (n_components, n_features)
    explained_variance_ratio: np.ndarray
    feature_names: list[str]

    @property
    def n_components(self) -> int:
        return int(self.components.shape[0])

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.feature_names if c not in features.columns]
        if missing:
            raise KeyError(f"PCA was fitted on features not present now: {missing[:5]}")
        x = features[self.feature_names].to_numpy(dtype="float64") - self.mean
        scores = x @ self.components.T
        return pd.DataFrame(
            scores.astype("float32"),
            columns=[f"pc{i + 1}" for i in range(self.n_components)],
            index=features.index,
        )

    def save(self, artifacts_dir: Path) -> Path:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        path = artifacts_dir / PCA_FILE
        np.savez(
            path,
            mean=self.mean,
            components=self.components,
            explained_variance_ratio=self.explained_variance_ratio,
            feature_names=np.array(self.feature_names, dtype=object),
        )
        return path

    @classmethod
    def load(cls, artifacts_dir: Path) -> "PCAModel":
        path = artifacts_dir / PCA_FILE
        if not path.exists():
            raise FileNotFoundError(f"no fitted PCA at {path}")
        data = np.load(path, allow_pickle=True)
        return cls(
            mean=data["mean"],
            components=data["components"],
            explained_variance_ratio=data["explained_variance_ratio"],
            feature_names=list(data["feature_names"]),
        )


def fit_pca(features: pd.DataFrame, *, variance: float = 0.95) -> PCAModel:
    """Fit PCA retaining enough components to explain ``variance`` of the total.

    Implemented directly via SVD rather than through scikit-learn, to keep the
    dependency surface small and the stored artifact a plain array file.
    """
    numeric = features.select_dtypes("number").fillna(0.0)
    names = list(numeric.columns)
    x = numeric.to_numpy(dtype="float64")
    mean = x.mean(axis=0)
    centred = x - mean

    # economy SVD: centred = U @ diag(s) @ Vt, and Vt rows are the components
    _, s, vt = np.linalg.svd(centred, full_matrices=False)
    explained = (s**2) / max(1, (len(x) - 1))
    ratio = explained / explained.sum()

    keep = int(np.searchsorted(np.cumsum(ratio), variance) + 1)
    keep = max(1, min(keep, len(ratio)))

    return PCAModel(
        mean=mean,
        components=vt[:keep],
        explained_variance_ratio=ratio[:keep],
        feature_names=names,
    )
