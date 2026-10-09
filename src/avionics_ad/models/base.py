"""The interface every detector implements.

Seven models span four quite different families — forecasters, reconstructors,
classical outlier detectors and a clustering method. They are only
interchangeable from a dropdown if they agree on something, and this is it:

    fit(X)    learn what normal looks like, from anomaly-free windows
    score(X)  return one number per window, higher meaning stranger

What that number *means* differs by family — prediction error, reconstruction
error, local density, distance to the nearest cluster — and deliberately so.
Everything downstream (thresholding, metrics, attribution, the UI) only needs
the ordering to be meaningful, not the units. That is what makes adding an
eighth model a matter of writing one class.

``score_features`` is optional, because only some families can answer it. A
forecaster knows exactly which channel it mispredicted; a one-class SVM has
collapsed the whole window into a single distance and genuinely cannot say.
Returning ``None`` is the honest answer, and attribution degrades rather than
inventing a number.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import numpy as np

Family = Literal["forecast", "reconstruct", "classical", "cluster"]


@dataclass(frozen=True)
class ParamSpec:
    """One tunable parameter, described well enough for a UI to render it."""

    name: str
    kind: Literal["int", "float", "bool", "choice"]
    default: Any
    help: str
    low: float | None = None
    high: float | None = None
    choices: tuple[Any, ...] = ()


class AnomalyModel(abc.ABC):
    """Base class for every detector."""

    #: Stable identifier, used by the registry and persisted in results.
    name: ClassVar[str] = "unnamed"
    #: Human-readable label for the dropdown.
    label: ClassVar[str] = "Unnamed"
    family: ClassVar[Family] = "classical"
    description: ClassVar[str] = ""
    params_spec: ClassVar[tuple[ParamSpec, ...]] = ()

    def __init__(self, **params: Any) -> None:
        defaults = {p.name: p.default for p in self.params_spec}
        unknown = set(params) - set(defaults)
        if unknown:
            raise TypeError(
                f"{self.name}: unknown parameter(s) {sorted(unknown)}; "
                f"known: {sorted(defaults)}"
            )
        self.params: dict[str, Any] = {**defaults, **params}
        self.training_scores_: np.ndarray | None = None
        self._fitted = False

    # -- the contract -------------------------------------------------------

    @abc.abstractmethod
    def _fit(self, X: np.ndarray) -> None:
        """Learn normal behaviour from ``(B, W, P)`` windows."""

    @abc.abstractmethod
    def _score(self, X: np.ndarray) -> np.ndarray:
        """One anomaly score per window, higher meaning stranger."""

    def score_features(self, X: np.ndarray) -> np.ndarray | None:
        """Per-channel contribution, ``(B, P)``, or None if unavailable."""
        return None

    # -- provided -----------------------------------------------------------

    def fit(self, X: np.ndarray) -> "AnomalyModel":
        """Fit, then record scores on the training data.

        The training scores are kept because the threshold is derived from them
        — a detector that cannot say how strange normal data looks has no basis
        for calling anything abnormal.
        """
        X = self._check(X, "fit")
        self._fit(X)
        self._fitted = True
        self.training_scores_ = np.asarray(self._score(X), dtype="float64")
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(f"{self.name}: call fit() before score()")
        X = self._check(X, "score")
        s = np.asarray(self._score(X), dtype="float64")
        if s.shape != (len(X),):
            raise RuntimeError(
                f"{self.name}: score() returned {s.shape}, expected ({len(X)},)"
            )
        return s

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def _check(self, X: np.ndarray, where: str) -> np.ndarray:
        X = np.asarray(X, dtype="float32")
        if X.ndim != 3:
            raise ValueError(
                f"{self.name}.{where}: expected (B, W, P) windows, got shape {X.shape}"
            )
        if len(X) == 0:
            raise ValueError(f"{self.name}.{where}: no windows supplied")
        if not np.isfinite(X).all():
            raise ValueError(
                f"{self.name}.{where}: input contains non-finite values. "
                "Validation should have quarantined these upstream."
            )
        return X

    def __repr__(self) -> str:
        shown = ", ".join(f"{k}={v!r}" for k, v in sorted(self.params.items()))
        return f"{type(self).__name__}({shown})"


def summary_features(X: np.ndarray) -> np.ndarray:
    """Collapse ``(B, W, P)`` windows to ``(B, P*6)`` descriptive statistics.

    Classical detectors take flat vectors, and the obvious move is to flatten
    the window outright. For a 100-step window over 12 channels that is 1,200
    dimensions for a few thousand samples, where distance-based methods stop
    working — everything becomes equidistant from everything else.

    Six statistics per channel keep what matters for the three anomaly kinds:
    mean and the extremes catch a level shift, standard deviation and range
    catch a pattern change, and the first/last delta plus the mean absolute
    step carry enough temporal shape to give a time shift something to show up
    in. 72 dimensions instead of 1,200.
    """
    X = np.asarray(X, dtype="float64")
    if X.ndim != 3:
        raise ValueError(f"expected (B, W, P), got {X.shape}")
    diffs = np.diff(X, axis=1)
    return np.concatenate(
        [
            X.mean(axis=1),
            X.std(axis=1),
            X.min(axis=1),
            X.max(axis=1),
            X[:, -1, :] - X[:, 0, :],
            np.abs(diffs).mean(axis=1) if diffs.shape[1] else np.zeros_like(X[:, 0, :]),
        ],
        axis=1,
    )


def per_channel_error(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Mean absolute error per channel, ``(B, P)``.

    The shared basis for attribution across forecasting and reconstruction
    models: decompose the total error across input features so the channel most
    responsible for a detection can be named.
    """
    a = np.asarray(actual, dtype="float64")
    p = np.asarray(predicted, dtype="float64")
    if a.shape != p.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {p.shape}")
    err = np.abs(a - p)
    return err.mean(axis=1) if err.ndim == 3 else err
