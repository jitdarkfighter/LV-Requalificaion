import numpy as np
import pytest

from avionics_ad import datasets as D
from avionics_ad.ui import logic as L


def cfg(**kw):
    base = dict(dataset="lv_synthetic", window=100, stride=10, model="ocsvm", params={},
                rule="median_plus_k_mad", k=3.0, percentile=99.0)
    base.update(kw)
    return L.make_config(**base)


# -- params ---------------------------------------------------------------

def test_coerce_params_types_and_unknown_keys():
    out = L.coerce_params("ocsvm", {"nu": 1, "gamma": "auto", "stale_key": 3})
    assert out == {"nu": 1.0, "gamma": "auto"} and isinstance(out["nu"], float)
    out = L.coerce_params("lstm", {"epochs": 3.0, "hidden_size": 8.0})
    assert out == {"epochs": 3, "hidden_size": 8} and isinstance(out["epochs"], int)


def test_default_params_trim_deep_epochs_only():
    assert L.default_params("lstm")["epochs"] == L.UI_EPOCH_DEFAULT
    assert "epochs" not in L.default_params("ocsvm")


# -- threshold controls -----------------------------------------------------

@pytest.mark.parametrize("rule,expected", [
    ("max_plus_sigma", ()), ("mean_plus_k_sigma", ("k",)),
    ("median_plus_k_mad", ("k",)), ("percentile", ("percentile",)),
])
def test_rule_controls(rule, expected):
    assert L.rule_controls(rule) == expected


def test_every_rule_has_description_and_controls():
    from avionics_ad import threshold as TH
    assert set(L.RULE_DESCRIPTIONS) == set(TH.RULES) == set(L.RULE_CONTROLS)


def test_unknown_rule_controls_raises():
    with pytest.raises(KeyError):
        L.rule_controls("bogus")


# -- staleness ----------------------------------------------------------------

def test_identical_config_is_not_stale():
    assert L.config_diff(cfg(), cfg()) == []


@pytest.mark.parametrize("change", [
    dict(window=60), dict(stride=5), dict(model="lof"), dict(rule="max_plus_sigma"),
    dict(k=5.0), dict(params={"nu": 0.1}), dict(dataset="cmapss"),
])
def test_any_relevant_change_is_stale(change):
    assert L.config_diff(cfg(), cfg(**change))


def test_irrelevant_controls_do_not_mark_stale():
    a = cfg(rule="percentile", k=3.0, percentile=95.0)
    assert L.config_diff(a, cfg(rule="percentile", k=9.0, percentile=95.0)) == []
    assert L.config_diff(a, cfg(rule="percentile", k=3.0, percentile=99.0))
    b = cfg(rule="max_plus_sigma")
    assert L.config_diff(b, cfg(rule="max_plus_sigma", k=8.0, percentile=60.0)) == []


def test_default_param_value_equals_unspecified():
    assert L.config_diff(cfg(params={}), cfg(params={})) == []
    # widget values are coerced to the same types a run would use
    assert cfg(params={"nu": 1}) == cfg(params={"nu": 1.0})


def test_describe_config_mentions_rule_controls():
    assert "k=" in L.describe_config(cfg()) and "q=" in L.describe_config(cfg(rule="percentile"))


def test_validate_selection():
    assert L.validate_selection(window=100, stride=10, dataset_reason=None) is None
    assert "Stride" in L.validate_selection(window=40, stride=60, dataset_reason=None)
    assert L.validate_selection(window=40, stride=1, dataset_reason="not built") == "not built"


# -- datasets -----------------------------------------------------------------

def test_dataset_status_labels_unbuilt_ones(monkeypatch):
    monkeypatch.setattr(D, "readiness", lambda n: "missing" if n == "cmapss" else None)
    status = {n: (label, reason) for n, label, reason in L.dataset_status()}
    assert status["cmapss"] == ("C-MAPSS (FD001) (not built)", "missing")
    assert status["lv_synthetic"][1] is None and "not built" not in status["lv_synthetic"][0]


def test_build_dataset_reports_real_cause(monkeypatch):
    monkeypatch.setattr(D, "readiness", lambda n: "configuration not found: x")
    with pytest.raises(RuntimeError, match="configuration not found"):
        L.build_dataset("cmapss", window=50, stride=10)


# -- presentation: one result, consistent views ---------------------------------

@pytest.fixture(scope="module")
def result():
    ds = D.build("lv_synthetic", window=100, stride=20)
    return L.run_detection("ocsvm", ds, threshold_rule="percentile", threshold_percentile=95.0)


def test_score_frame_agrees_with_detection(result):
    f = L.score_frame(result)
    d = result.detection
    assert int(f.flagged.sum()) == result.extras["n_flagged"] == d["true_positives"] + d["false_positives"]
    assert int((f.flagged & f["actual anomaly"]).sum()) == d["true_positives"]
    assert (f.threshold == result.threshold["value"]).all()


def test_rule_comparison_active_row_matches_headline(result):
    rc = L.rule_comparison_table(result)
    active = rc[rc.active == "yes"]
    assert len(active) == 1 and active.iloc[0].rule == result.threshold["rule"] == "percentile"
    assert active.iloc[0].F2 == round(result.detection["f_beta"], 3)
    assert active.iloc[0].recall == round(result.detection["recall"], 3)


def test_confusion_frame_sums_to_windows(result):
    assert L.confusion_frame(result)["count"].sum() == result.detection["n_windows"]


def test_per_kind_table_matches_result(result):
    t = L.per_kind_table(result)
    assert len(t) == len(result.per_kind_recall)


def test_anomaly_spans_cover_labels(result):
    f = L.score_frame(result)
    spans = L.anomaly_spans(f)
    covered = sum(int(e - s) for s, e in zip(spans.start, spans.end))
    assert covered == int(f["actual anomaly"].sum())


def test_log_scale_disabled_for_non_positive_scores(result):
    assert not L.log_scale_allowed(result)  # one-class SVM scores are <= 0 somewhere


def test_training_curve_empty_for_classical(result):
    assert L.training_curve(result).empty


def test_headline_metrics_come_from_result(result):
    vals = {label: v for label, v, _ in L.headline_metrics(result)}
    assert vals["Recall"] == f"{result.detection['recall']:.1%}"


def test_interpretation_flags_silent_model(result):
    import copy
    r = copy.deepcopy(result)
    r.detection.update(true_positives=0, false_positives=0)
    assert any("flagged nothing" in n for n in L.interpretation(r))


def test_run_warnings_surface_forecast_error(result):
    import copy
    r = copy.deepcopy(result)
    r.extras["forecast_error"] = "ValueError: boom"
    assert any("boom" in w for w in L.run_warnings(r))
    assert L.run_warnings(result) == []


def test_threshold_stats_show_only_relevant_controls(result):
    names = set(L.threshold_stats(result)["statistic"])
    assert "percentile" in names and "k" not in names


# -- docs page --------------------------------------------------------------

def test_param_table_matches_registry():
    import avionics_ad.models as M
    for name in M.names():
        spec = [p.name for p in M.get(name).params_spec]
        assert list(L.param_table(name)["parameter"]) == spec
        specific = list(L.model_specific_params(name)["parameter"])
        assert not set(specific) & set(L.SHARED_DEEP_PARAMS)
    t = L.param_table("ocsvm").set_index("parameter")
    assert t.loc["nu", "range"] == "0.001 – 0.5" and t.loc["gamma", "range"] == "scale / auto"


def test_every_rule_has_plain_language():
    for rule in L.threshold_options():
        assert L.RULE_LABELS[rule] and L.RULE_PLAIN[rule] and L.RULE_DESCRIPTIONS[rule]
