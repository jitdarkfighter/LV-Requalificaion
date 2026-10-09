"""Evaluation metrics.

Two groups, answering different questions.

**Forecast quality** — MSE and MAE. How well does the model reproduce normal
behaviour? This is the training objective, and a model that cannot predict
normal data has no basis for calling anything abnormal.

**Detection quality** — precision, recall, F-beta, false-negative rate.
Did it actually find the anomalies?

Why F2 and not F1
-----------------
In launch-vehicle testing the two errors are not symmetric. A false positive
costs an engineer some time investigating. A false negative means a real fault
was cleared for flight. The reference study is explicit that recall takes
precedence, and uses F-beta with ``beta > 1`` for exactly that reason.

``beta = 2`` weights recall four times as heavily as precision, because the
F-beta weighting goes as ``beta^2``. The false-negative rate is reported
alongside because it is the number a safety case actually cares about, and it is
easier to argue about than a composite score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

#: Recall is weighted beta^2 = 4 times as heavily as precision.
DEFAULT_BETA = 2.0


@dataclass
class ForecastMetrics:
    """How well the model reproduces normal behaviour."""

    mse: float
    mae: float
    rmse: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class DetectionMetrics:
    """How well the model finds anomalies, at one threshold."""

    precision: float
    recall: float
    f_beta: float
    beta: float
    false_negative_rate: float
    false_positive_rate: float
    accuracy: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    n_anomalies: int
    n_windows: int

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def forecast_metrics(actual: np.ndarray, predicted: np.ndarray) -> ForecastMetrics:
    """MSE, MAE and RMSE between actual and predicted values."""
    a = np.asarray(actual, dtype="float64")
    p = np.asarray(predicted, dtype="float64")
    if a.shape != p.shape:
        raise ValueError(f"shape mismatch: actual {a.shape} vs predicted {p.shape}")
    if a.size == 0:
        raise ValueError("cannot compute forecast metrics on empty input")
    err = a - p
    mse = float(np.mean(err**2))
    return ForecastMetrics(mse=mse, mae=float(np.mean(np.abs(err))), rmse=float(np.sqrt(mse)))


def detection_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, *, beta: float = DEFAULT_BETA
) -> DetectionMetrics:
    """Confusion-matrix metrics for a boolean prediction.

    Degenerate cases are defined rather than left to produce NaN:

    * no predictions at all -> precision 1.0 (it was never wrong when it spoke)
    * no actual anomalies   -> recall 1.0 (it missed none, because there were none)

    Both are the conventional choices, and both are worth knowing about when
    reading a result: a precision of 1.0 on zero detections is not a good model.
    """
    t = np.asarray(y_true, dtype=bool)
    p = np.asarray(y_pred, dtype=bool)
    if t.shape != p.shape:
        raise ValueError(f"shape mismatch: y_true {t.shape} vs y_pred {p.shape}")
    if t.size == 0:
        raise ValueError("cannot compute detection metrics on empty input")
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")

    tp = int(np.sum(t & p))
    fp = int(np.sum(~t & p))
    tn = int(np.sum(~t & ~p))
    fn = int(np.sum(t & ~p))

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0

    b2 = beta**2
    denominator = b2 * precision + recall
    f_beta = ((1 + b2) * precision * recall / denominator) if denominator else 0.0

    return DetectionMetrics(
        precision=precision,
        recall=recall,
        f_beta=f_beta,
        beta=beta,
        false_negative_rate=fn / (tp + fn) if (tp + fn) else 0.0,
        false_positive_rate=fp / (fp + tn) if (fp + tn) else 0.0,
        accuracy=(tp + tn) / t.size,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        n_anomalies=int(t.sum()),
        n_windows=int(t.size),
    )


def f_beta(precision: float, recall: float, *, beta: float = DEFAULT_BETA) -> float:
    """F-beta from precision and recall directly."""
    b2 = beta**2
    denominator = b2 * precision + recall
    return ((1 + b2) * precision * recall / denominator) if denominator else 0.0


def sweep_threshold(
    scores: np.ndarray, y_true: np.ndarray, *, beta: float = DEFAULT_BETA, n: int = 200
) -> list[dict[str, float]]:
    """Detection metrics across a range of thresholds.

    Lets a chosen threshold be judged against what was achievable, which matters
    because the reference study's threshold rule is a heuristic rather than an
    optimum.
    """
    s = np.asarray(scores, dtype="float64")
    if s.size == 0:
        return []
    lo, hi = float(np.min(s)), float(np.max(s))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        candidates = [lo]
    else:
        candidates = list(np.linspace(lo, hi, n))
    out = []
    for threshold in candidates:
        m = detection_metrics(y_true, s > threshold, beta=beta)
        out.append({"threshold": float(threshold), **m.as_dict()})
    return out


def best_achievable(
    scores: np.ndarray, y_true: np.ndarray, *, beta: float = DEFAULT_BETA, n: int = 200
) -> dict[str, float]:
    """The best F-beta any threshold on these scores could reach.

    An upper bound on the detector itself, separate from the threshold rule. If
    this is low, no threshold will save the model; if it is high while the
    chosen threshold scores badly, the threshold is the thing to fix.
    """
    sweep = sweep_threshold(scores, y_true, beta=beta, n=n)
    if not sweep:
        return {}
    return max(sweep, key=lambda r: r["f_beta"])
