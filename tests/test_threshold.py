import numpy as np
import pytest

from avionics_ad import threshold as TH

S = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])


def test_max_plus_sigma():
    t = TH.fit(S, rule="max_plus_sigma")
    assert t.value == pytest.approx(S.max() + S.std())


def test_mean_plus_k_sigma_uses_k():
    a, b = TH.fit(S, rule="mean_plus_k_sigma", k=1), TH.fit(S, rule="mean_plus_k_sigma", k=2)
    assert a.value == pytest.approx(S.mean() + S.std())
    assert b.value > a.value


def test_median_plus_k_mad():
    t = TH.fit(S, rule="median_plus_k_mad", k=3)
    med = np.median(S)
    mad = np.median(np.abs(S - med))
    assert t.value == pytest.approx(med + 3 * 1.4826 * mad)


def test_mad_degenerate_falls_back_to_std():
    s = np.array([1.0] * 8 + [5.0, 9.0])
    t = TH.fit(s, rule="median_plus_k_mad", k=2)
    assert t.train_mad == 0
    assert t.value == pytest.approx(1.0 + 2 * s.std())


def test_percentile_rule_and_validation():
    assert TH.fit(S, rule="percentile", percentile=50).value == pytest.approx(np.percentile(S, 50))
    assert TH.fit(S, rule="percentile", percentile=90).value > TH.fit(S, rule="percentile", percentile=50).value
    with pytest.raises(ValueError):
        TH.fit(S, rule="percentile", percentile=101)


def test_percentile_ignored_by_other_rules():
    assert TH.fit(S, rule="max_plus_sigma", percentile=10).value == TH.fit(S, rule="max_plus_sigma").value


def test_empty_and_non_finite_input_raise():
    with pytest.raises(ValueError):
        TH.fit(np.array([]))
    with pytest.raises(ValueError):
        TH.fit(np.array([np.nan, np.inf]))


def test_non_finite_values_are_filtered_not_propagated():
    t = TH.fit(np.array([1.0, 2.0, np.nan, 3.0]), rule="mean_plus_k_sigma")
    assert np.isfinite(t.value) and t.n_train == 3


def test_unknown_rule_raises():
    with pytest.raises(ValueError):
        TH.fit(S, rule="nope")  # type: ignore[arg-type]


def test_apply_is_strictly_greater():
    t = TH.fit(S, rule="percentile", percentile=50)
    assert not t.apply(np.array([t.value]))[0]
