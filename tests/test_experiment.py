import numpy as np
import pytest

from avionics_ad import datasets as D
from avionics_ad import experiment as E
from avionics_ad import metrics as MT
from avionics_ad import threshold as TH


@pytest.fixture(scope="module")
def ds():
    return D.build("lv_synthetic", window=100, stride=20)


@pytest.mark.parametrize("model", ["lof", "ocsvm"])
def test_run_produces_consistent_result(ds, model):
    r = E.run(model, ds, threshold_rule="median_plus_k_mad")
    assert np.isfinite(r.scores).all() and len(r.scores) == len(ds.test)
    assert r.attribution is None and r.extras["attribution_scope"] == "unavailable"
    assert r.forecast == {} and "forecast_error" not in r.extras
    # per-kind recall recomputed independently from the stored threshold
    flagged = r.scores > r.threshold["value"]
    for kind, rec in r.per_kind_recall.items():
        mask = (ds.kinds == kind) & ds.labels
        assert rec == pytest.approx(flagged[mask].mean())
    assert r.detection["recall"] == pytest.approx(MT.detection_metrics(ds.labels, flagged).recall)
    assert set(r.rule_comparison) == set(TH.RULES)
    assert r.timings["evaluate"] >= 0


def test_percentile_is_plumbed_through_run_and_comparison(ds):
    lo = E.run("ocsvm", ds, threshold_rule="percentile", threshold_percentile=50.0)
    hi = E.run("ocsvm", ds, threshold_rule="percentile", threshold_percentile=99.0)
    assert lo.threshold["percentile"] == 50.0 and hi.threshold["percentile"] == 99.0
    assert lo.threshold["value"] < hi.threshold["value"]
    # the comparison row for the active rule matches the headline metrics
    assert lo.rule_comparison["percentile"]["f_beta"] == pytest.approx(lo.detection["f_beta"])
    assert lo.detection["recall"] >= hi.detection["recall"]


def test_default_percentile_unchanged(ds):
    r = E.run("ocsvm", ds, threshold_rule="percentile")
    assert r.threshold["percentile"] == 99.0


def test_progress_stages(ds):
    seen = []
    E.run("ocsvm", ds, progress=seen.append)
    assert seen == ["fitting", "scoring", "evaluating"]


def test_deep_model_runs_with_history(ds):
    r = E.run("lstm", ds, params={"epochs": 1, "hidden_size": 8, "patience": 1})
    assert np.isfinite(r.scores).all()
    assert len(r.extras["history"]["train_loss"]) == 1
    assert r.forecast and "forecast_error" not in r.extras
    assert r.attribution is not None


def test_forecast_failure_is_recorded_not_swallowed(ds, monkeypatch):
    def boom(actual, predicted):
        raise ValueError("bad forecast")

    monkeypatch.setattr(MT, "forecast_metrics", boom)
    r = E.run("lstm", ds, params={"epochs": 1, "hidden_size": 8, "patience": 1})
    assert r.forecast == {}
    assert "ValueError: bad forecast" in r.extras["forecast_error"]
    assert np.isfinite(r.scores).all()  # detection results still stand


# -- edge cases --------------------------------------------------------------------

def test_window_longer_than_series_raises_clearly():
    with pytest.raises(ValueError, match="no training windows"):
        D.build("lv_synthetic", window=10000, stride=50)


def test_no_anomalous_windows_raises_clearly(ds):
    import dataclasses
    empty = dataclasses.replace(ds, labels=np.zeros_like(ds.labels))
    with pytest.raises(ValueError, match="no test window overlaps"):
        E.run("ocsvm", empty)


def test_no_test_windows_raises_clearly(ds):
    import dataclasses
    empty = dataclasses.replace(ds, test=ds.test[:0], labels=ds.labels[:0], kinds=ds.kinds[:0])
    with pytest.raises(ValueError, match="no test windows"):
        D.validate(empty)


def test_non_finite_scores_raise_not_hide(ds, monkeypatch):
    import avionics_ad.models as M

    cls = M.get("ocsvm")
    orig = cls._score

    def bad(self, X):
        s = orig(self, X)
        s = np.array(s, dtype="float64")
        s[0] = np.nan
        return s

    monkeypatch.setattr(cls, "_score", bad)
    with pytest.raises(ValueError, match="non-finite"):
        E.run("ocsvm", ds)


def test_cmapss_readiness_reports_missing_config(monkeypatch, tmp_path):
    from avionics_etl import config

    monkeypatch.setattr(config, "DEFAULT_CONFIG", tmp_path / "nope.yaml")
    reason = D.readiness("cmapss")
    assert "configuration not found" in reason and "pipeline.yaml" in reason


def test_synthetic_always_ready():
    assert D.readiness("lv_synthetic") is None
