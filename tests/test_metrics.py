import numpy as np
import pytest

from avionics_ad import metrics as MT


def test_perfect_detection():
    y = np.array([0, 0, 1, 1], bool)
    m = MT.detection_metrics(y, y)
    assert (m.precision, m.recall, m.f_beta, m.false_negative_rate) == (1, 1, 1, 0)


def test_no_predictions_has_precision_one_but_zero_recall():
    y = np.array([0, 1, 1, 0], bool)
    m = MT.detection_metrics(y, np.zeros(4, bool))
    assert m.precision == 1.0 and m.recall == 0.0 and m.f_beta == 0.0 and m.false_negative_rate == 1.0


def test_no_actual_anomalies_has_recall_one():
    m = MT.detection_metrics(np.zeros(4, bool), np.array([1, 0, 0, 0], bool))
    assert m.recall == 1.0 and m.precision == 0.0 and np.isfinite(m.f_beta)


def test_counts_and_f2():
    y = np.array([1, 1, 0, 0, 1], bool)
    p = np.array([1, 0, 1, 0, 1], bool)
    m = MT.detection_metrics(y, p)
    assert (m.true_positives, m.false_positives, m.false_negatives, m.true_negatives) == (2, 1, 1, 1)
    pr, rc = 2 / 3, 2 / 3
    assert m.f_beta == pytest.approx(5 * pr * rc / (4 * pr + rc))


def test_input_validation():
    with pytest.raises(ValueError):
        MT.detection_metrics(np.zeros(3, bool), np.zeros(4, bool))
    with pytest.raises(ValueError):
        MT.detection_metrics(np.array([], bool), np.array([], bool))
    with pytest.raises(ValueError):
        MT.detection_metrics(np.zeros(2, bool), np.zeros(2, bool), beta=0)


def test_best_achievable_beats_or_equals_any_fixed_threshold():
    rng = np.random.default_rng(0)
    y = rng.random(200) < 0.2
    s = rng.normal(size=200) + 3 * y
    best = MT.best_achievable(s, y)["f_beta"]
    assert best >= MT.detection_metrics(y, s > 1.0).f_beta - 1e-9


def test_forecast_metrics():
    m = MT.forecast_metrics(np.zeros(4), np.ones(4))
    assert m.mse == 1 and m.mae == 1 and m.rmse == 1
