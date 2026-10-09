"""Classical outlier detectors.

Neither of these knows anything about time. They see a window as a point in
feature space and ask how isolated it is. That makes them a genuine baseline:
if a deep sequence model cannot beat a one-class SVM on summary statistics,
the sequence modelling is not earning its complexity.

They are also the fast option — seconds rather than minutes — which matters for
a UI where somebody is clicking through models.

Both are fitted on data assumed anomaly-free, which is the novelty-detection
setting rather than the outlier-detection one. The distinction matters for LOF
in particular: scikit-learn's ``LocalOutlierFactor`` only exposes scoring for
unseen data when constructed with ``novelty=True``.
"""

from __future__ import annotations

import numpy as np

from avionics_ad.models.base import AnomalyModel, ParamSpec, summary_features
from avionics_ad.models.registry import register

FEATURE_MODES = ("summary", "flatten")


def _features(X: np.ndarray, mode: str) -> np.ndarray:
    if mode == "summary":
        return summary_features(X)
    if mode == "flatten":
        return np.asarray(X, dtype="float64").reshape(len(X), -1)
    raise ValueError(f"unknown feature mode {mode!r}; expected one of {FEATURE_MODES}")


class _ClassicalModel(AnomalyModel):
    """Shared plumbing: build features, fit an sklearn estimator, negate scores.

    scikit-learn's novelty detectors return *higher is more normal*. Everything
    here is *higher is stranger*, so the sign is flipped once, in one place.
    """

    def _build(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def _fit(self, X: np.ndarray) -> None:
        feats = _features(X, self.params["features"])
        self._estimator = self._build()
        self._estimator.fit(feats)

    def _score(self, X: np.ndarray) -> np.ndarray:
        feats = _features(X, self.params["features"])
        return -np.asarray(self._estimator.score_samples(feats), dtype="float64")


@register
class LocalOutlierFactorModel(_ClassicalModel):
    name = "lof"
    label = "Local Outlier Factor + threshold"
    family = "classical"
    description = """
    Compares each window's local density to that of its neighbours. A window in
    a sparse region surrounded by dense ones is anomalous. Makes no assumption
    about the shape of normality, so it handles the case where normal operation
    has several distinct modes — different test phases, for instance — which a
    single-boundary method struggles with.
    """
    params_spec = (
        ParamSpec("n_neighbors", "int", 20, "Neighbourhood size for the density estimate", 2, 200),
        ParamSpec("features", "choice", "summary", "Window representation", choices=FEATURE_MODES),
        ParamSpec("metric", "choice", "euclidean", "Distance metric",
                  choices=("euclidean", "manhattan", "chebyshev")),
    )

    def _build(self):
        from sklearn.neighbors import LocalOutlierFactor

        return LocalOutlierFactor(
            n_neighbors=int(self.params["n_neighbors"]),
            metric=self.params["metric"],
            novelty=True,
        )


@register
class OneClassSVMModel(_ClassicalModel):
    name = "ocsvm"
    label = "One-Class SVM"
    family = "classical"
    description = """
    Fits a boundary enclosing the normal data and scores how far outside it a
    window falls. A single connected region, so it is the stricter assumption of
    the two classical methods — effective when normal operation is one coherent
    mode, weaker when it is several.

    Inputs are standardised first: an RBF kernel measures plain Euclidean
    distance, so a channel in Newtons would otherwise dominate one in volts
    purely through its units.
    """
    params_spec = (
        ParamSpec("nu", "float", 0.05, "Expected fraction of training data outside the boundary", 0.001, 0.5),
        ParamSpec("gamma", "choice", "scale", "RBF kernel width", choices=("scale", "auto")),
        ParamSpec("features", "choice", "summary", "Window representation", choices=FEATURE_MODES),
    )

    def _build(self):
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import OneClassSVM

        return make_pipeline(
            StandardScaler(),
            OneClassSVM(nu=float(self.params["nu"]), gamma=self.params["gamma"], kernel="rbf"),
        )
