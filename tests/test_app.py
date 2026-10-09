"""End-to-end through the real Streamlit script, using Streamlit's AppTest."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from avionics_ad.ui import logic as L

APP = str(Path(__file__).resolve().parents[1] / "src" / "avionics_ad" / "ui" / "app.py")


def fresh() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception
    return at


def pick(at: AppTest, label: str, value):
    next(s for s in at.sidebar.selectbox if s.label == label).select(value)
    return at.run()


def run_button(at: AppTest):
    return next(b for b in at.sidebar.button if b.label == "Run detection")


def fast_ocsvm(at: AppTest) -> AppTest:
    pick(at, "Family", "classical")
    pick(at, "Model", "One-Class SVM")
    pick(at, "Window (samples)", 100)
    pick(at, "Stride (samples)", 20)
    run_button(at).click()
    return at.run()


def test_empty_state_before_run():
    at = fresh()
    assert not at.metric
    assert any("How to use" in s.value for s in at.subheader)


def test_threshold_controls_follow_rule():
    at = fresh()
    assert [s.label for s in at.sidebar.slider] == ["Sensitivity (k)"]
    pick(at, "How to set it", "percentile")
    assert not at.sidebar.slider
    assert [n.label for n in at.sidebar.number_input if n.label == "Percentile"]
    pick(at, "How to set it", "max_plus_sigma")
    assert not at.sidebar.slider and not [n for n in at.sidebar.number_input if n.label == "Percentile"]


def test_run_shows_metrics_then_stale_warning_then_failure_clears():
    at = fast_ocsvm(fresh())
    assert not at.exception and not at.error
    # A bare expression returning a DeltaGenerator gets echoed by Streamlit magic.
    assert not at.get("help_info")
    labels = [m.label for m in at.metric]
    assert labels == ["Recall", "F2 score", "False negative rate", "Precision"]
    result = at.session_state["run"]["result"]
    assert at.metric[0].value == f"{result.detection['recall']:.1%}"
    assert not [w for w in at.warning if "Settings changed" in w.value]

    # change a setting: results must be marked stale, not silently shown as current
    pick(at, "Stride (samples)", 10)
    stale = [w for w in at.warning if "Settings changed" in w.value]
    assert stale and "stride: 20 -> 10" in stale[0].value
    assert at.metric  # still shown, but flagged

    # a different threshold rule is also a changed setting
    at = fast_ocsvm(fresh())
    pick(at, "How to set it", "max_plus_sigma")
    assert [w for w in at.warning if "Settings changed" in w.value]


def test_failed_run_clears_previous_result(monkeypatch):
    at = fast_ocsvm(fresh())
    assert at.metric and "run" in at.session_state

    def boom(*a, **k):
        raise RuntimeError("forced failure")

    monkeypatch.setattr(L, "run_detection", boom)
    run_button(at).click()
    at.run()
    assert [e for e in at.error if "RuntimeError" in e.value and "forced failure" in e.value]
    assert not at.metric
    assert "run" not in at.session_state


def test_percentile_run_uses_percentile():
    at = fresh()
    pick(at, "Family", "classical")
    pick(at, "Model", "One-Class SVM")
    pick(at, "Stride (samples)", 20)
    pick(at, "How to set it", "percentile")
    run_button(at).click()
    at.run()
    assert not at.exception and not at.error
    assert at.session_state["run"]["result"].threshold["rule"] == "percentile"


def test_unbuilt_dataset_is_labelled_and_run_disabled(monkeypatch):
    from avionics_ad import datasets as D

    monkeypatch.setattr(D, "readiness", lambda n: "layers missing" if n == "cmapss" else None)
    at = fresh()
    ds = next(s for s in at.sidebar.selectbox if s.label == "Dataset")
    assert any("not built" in o for o in ds.options)
    ds.select("cmapss").run()
    assert run_button(at).disabled
    assert any("layers missing" in e.value for e in at.sidebar.error)


@pytest.mark.skipif(L.D.readiness("cmapss") is not None, reason="C-MAPSS layers not built (run `make all`)")
def test_cmapss_end_to_end():
    at = fresh()
    pick(at, "Dataset", "cmapss")
    pick(at, "Family", "classical")
    pick(at, "Model", "Local Outlier Factor + threshold")
    pick(at, "Window (samples)", 60)
    run_button(at).click()
    at.run()
    assert not at.exception and not at.error
    assert at.session_state["run"]["result"].dataset.startswith("cmapss")


DOCS_PAGE = str(Path(APP).parent / "pages" / "1_Model_docs.py")


def test_front_page_links_docs():
    at = fresh()
    links = at.sidebar.get("page_link")
    assert any(l.proto.page == "Model_docs" for l in links)
    button = next(b for b in at.button if b.label == "Open model docs")
    button.click().run()
    assert not at.exception
    assert [t.value for t in at.title] == ["Model docs"]


def test_docs_page_renders_every_model():
    import avionics_ad.models as M
    at = AppTest.from_file(DOCS_PAGE, default_timeout=60).run()
    assert not at.exception and not at.error
    assert not at.get("help_info")
    titles = " ".join(e.label for e in at.expander)
    for info in M.available():
        assert f"`{info.name}`" in titles
    # Every model has hand-written prose, not the registry fallback.
    src = Path(DOCS_PAGE).read_text()
    for name in M.names():
        assert f'"{name}": (' in src, f"no docs entry for {name}"


def test_project_overview_page_renders_doc():
    page = str(Path(APP).parent / "pages" / "2_Project_overview.py")
    at = AppTest.from_file(page, default_timeout=30).run()
    assert not at.exception and not at.error
    text = " ".join(m.value for m in at.markdown)
    assert "Backend ETL" in text and "medallion" in text
    assert len(at.get("image")) == 1 and not at.warning  # figure found and shown
    links = fresh().get("page_link")
    assert any(l.proto.page == "Project_overview" for l in links)
