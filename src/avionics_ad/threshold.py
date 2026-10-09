"""Turn a continuous anomaly score into a yes/no decision.

Every model here produces a score where higher means stranger. Converting that
to a detection needs a cut-off, and the cut-off has to be derived from data the
model considers normal — the whole premise is that anomalies are not available
to calibrate against.

Two rules are provided because they behave very differently.

``max_plus_sigma`` (the reference study's)
    ``max(train_loss) + sigma(train_loss) + offset``

    Anchored on the single worst training window. Deliberately conservative:
    nothing in training should ever fire. That minimises false alarms, but it is
    hostage to one outlier — a single bad training window pushes the threshold
    up and real anomalies slip under it. The ``offset`` is described as being
    set empirically from domain knowledge, which is an admission that the rule
    alone is not sufficient.

``mean_plus_k_sigma``
    ``mean(train_loss) + k * sigma(train_loss)``

    The same study's own equation 3. Anchored on the bulk of the distribution
    rather than its extreme, so it is robust to a single bad window and k gives
    a direct sensitivity dial.

``percentile``
    The q-th percentile of training scores. Makes no distributional assumption
    at all.

``median_plus_k_mad``
    ``median + k * 1.4826 * MAD``, where MAD is the median absolute deviation.

    The robust counterpart of mean-plus-k-sigma, and the one that handles the
    case the others get wrong. Several detectors here produce heavily skewed
    training scores — the clustering model's ran from a median of 1.0 to a
    maximum of 41.7 — because a handful of training windows sit at the edge of
    normal. Both the mean and the standard deviation are dragged upward by that
    tail, so the threshold lands above most genuine anomalies. The median and
    the MAD are not, so it does not.

Which rule is right depends on the shape of a given model's score distribution,
which is an empirical question rather than a choice to make once. All four are
evaluated on the same fitted model in the benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Rule = Literal["max_plus_sigma", "mean_plus_k_sigma", "percentile", "median_plus_k_mad"]

RULES: tuple[Rule, ...] = (
    "max_plus_sigma",
    "mean_plus_k_sigma",
    "percentile",
    "median_plus_k_mad",
)

#: Scale factor making the MAD a consistent estimator of the standard deviation
#: for normally distributed data, so k means the same thing in both rules.
MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True)
class Threshold:
    """A fitted decision boundary, carrying how it was derived."""

    value: float
    rule: Rule
    train_mean: float
    train_std: float
    train_max: float
    train_median: float
    train_mad: float
    n_train: int
    k: float = 3.0
    offset: float = 0.0
    percentile: float = 99.0

    def apply(self, scores: np.ndarray) -> np.ndarray:
        """Flag scores strictly above the boundary."""
        return np.asarray(scores, dtype="float64") > self.value

    def describe(self) -> str:
        return f"{self.rule} = {self.value:.6g}  (train mean {self.train_mean:.4g}, max {self.train_max:.4g})"


def fit(
    train_scores: np.ndarray,
    *,
    rule: Rule = "max_plus_sigma",
    k: float = 3.0,
    offset: float = 0.0,
    percentile: float = 99.0,
) -> Threshold:
    """Derive a threshold from scores on data assumed anomaly-free."""
    s = np.asarray(train_scores, dtype="float64")
    s = s[np.isfinite(s)]
    if s.size == 0:
        raise ValueError("cannot fit a threshold on empty or non-finite training scores")
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}; expected one of {RULES}")

    mean, std, mx = float(s.mean()), float(s.std(ddof=0)), float(s.max())
    median = float(np.median(s))
    mad = float(np.median(np.abs(s - median)))

    if rule == "max_plus_sigma":
        value = mx + std + offset
    elif rule == "mean_plus_k_sigma":
        value = mean + k * std + offset
    elif rule == "median_plus_k_mad":
        # Fall back to the standard deviation only if the MAD is degenerate,
        # which happens when more than half the training scores are identical.
        scale = MAD_TO_SIGMA * mad if mad > 1e-12 else std
        value = median + k * scale + offset
    else:
        if not 0 <= percentile <= 100:
            raise ValueError(f"percentile must be in [0, 100], got {percentile}")
        value = float(np.percentile(s, percentile)) + offset

    return Threshold(
        value=value,
        rule=rule,
        train_mean=mean,
        train_std=std,
        train_max=mx,
        train_median=median,
        train_mad=mad,
        n_train=int(s.size),
        k=k,
        offset=offset,
        percentile=percentile,
    )
