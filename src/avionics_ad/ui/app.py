"""Streamlit front end. All decisions live in :mod:`avionics_ad.ui.logic`.

  ./run_demo.sh
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import altair as alt  # noqa: E402  (ships with Streamlit)
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import avionics_ad.models as M  # noqa: E402
from avionics_ad.ui import logic as L  # noqa: E402

st.set_page_config(
    page_title="Avionics anomaly detection",
    page_icon=":material/flight:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Modest styling. Only theme-neutral values (inherit / translucent), so it reads
# in both the light and the dark Streamlit theme.
st.markdown(
    """
    <style>
    .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px; }
    [data-testid="stSidebar"] h3 { margin-top: .4rem; }
    .step { display:inline-block; width:1.5rem; height:1.5rem; line-height:1.5rem; text-align:center;
            border-radius:50%; background: rgba(128,128,128,.25); font-weight:600; margin-right:.4rem; }
    .runbar { padding:.55rem .9rem; border-radius:.5rem; background: rgba(128,128,128,.12);
              font-size:.9rem; margin-bottom:.8rem; }
    .st-key-results-stale { opacity:.5; }
    div[data-testid="stMetric"] { padding:.2rem .1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

BLUE, RED, ORANGE, GREY = "#4C78A8", "#E45756", "#F58518", "#8c8c8c"


@st.cache_data(show_spinner=False)
def get_dataset(name: str, window: int, stride: int):
    return L.build_dataset(name, window=window, stride=stride)


# ---------------------------------------------------------------------------
# header
# ---------------------------------------------------------------------------

st.title("Avionics anomaly detection")
st.caption(
    "Models are fitted on **clean** telemetry only and scored on a copy with injected faults. "
    "Detection is judged recall-first (F2) because a missed fault costs more than a false alarm."
)

# ---------------------------------------------------------------------------
# sidebar: 1 data -> 2 model -> 3 threshold -> run
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown('### <span class="step">1</span>Data', unsafe_allow_html=True)
    statuses = L.dataset_status()
    labels = {name: label for name, label, _ in statuses}
    reasons = {name: reason for name, _, reason in statuses}
    dataset_name = st.selectbox("Dataset", list(labels), format_func=labels.get)
    window = st.selectbox("Window (samples)", L.WINDOW_CHOICES, index=2,
                          help="Each score covers one window of this many samples.")
    stride = st.selectbox("Stride (samples)", L.STRIDE_CHOICES, index=2,
                          help="Step between consecutive windows. Smaller means more windows and a slower run.")

    preview = None
    problem = L.validate_selection(window=int(window), stride=int(stride), dataset_reason=reasons[dataset_name])
    if problem is None:
        try:
            preview = get_dataset(dataset_name, int(window), int(stride))
        except Exception as exc:  # shown inline; Run is disabled below
            problem = f"{type(exc).__name__}: {exc}"
    if preview is not None:
        st.caption(
            f"{len(preview.train):,} train / {len(preview.test):,} test windows, "
            f"{preview.n_channels} channels, {int(preview.labels.sum()):,} anomalous "
            f"({preview.anomaly_rate:.1%})"
        )
    else:
        st.error(problem)

    st.divider()
    st.markdown('### <span class="step">2</span>Model', unsafe_allow_html=True)
    options = L.model_options()
    family = st.selectbox("Family", sorted({o.family for o in options}))
    in_family = [o for o in options if o.family == family]
    option = st.selectbox("Model", in_family, format_func=lambda o: o.label)
    with st.expander("About this model"):
        st.write(option.description)
    st.page_link("pages/1_Model_docs.py", label="Model docs", icon=":material/menu_book:")

    raw: dict = {}
    defaults = L.default_params(option.name)
    with st.expander("Parameters"):
        for p in M.get(option.name).params_spec:
            key = f"{option.name}.{p.name}"
            d = defaults[p.name]
            lo = None if p.low is None else p.low
            hi = None if p.high is None else p.high
            if p.kind == "int":
                raw[p.name] = st.number_input(
                    p.name, None if lo is None else int(lo), None if hi is None else int(hi),
                    int(d), step=1, help=p.help, key=key)
            elif p.kind == "float":
                raw[p.name] = st.number_input(
                    p.name, None if lo is None else float(lo), None if hi is None else float(hi),
                    float(d), format="%g", help=p.help, key=key)
            elif p.kind == "bool":
                raw[p.name] = st.checkbox(p.name, bool(d), help=p.help, key=key)
            else:
                raw[p.name] = st.selectbox(p.name, list(p.choices), list(p.choices).index(d),
                                           help=p.help, key=key)

    st.divider()
    st.markdown('### <span class="step">3</span>Alert threshold', unsafe_allow_html=True)
    st.caption("A window is flagged when its score is above the threshold. "
               "The threshold is set from the scores on clean training data.")
    rules = L.threshold_options()
    rule = st.selectbox("How to set it", rules, index=rules.index(L.DEFAULT_RULE),
                        format_func=L.RULE_LABELS.get,
                        help="The exact formulas are on the Model docs page.")
    st.caption(L.RULE_PLAIN[rule])
    controls = L.rule_controls(rule)
    k = L.DEFAULT_K
    percentile = L.DEFAULT_PERCENTILE
    if "k" in controls:
        k = st.slider("Sensitivity (k)", 0.5, 10.0, L.DEFAULT_K, 0.5,
                      help=f"Lower k gives more alerts: more faults caught, more false alarms. "
                           f"Formula: {L.RULE_DESCRIPTIONS[rule]}")
    if "percentile" in controls:
        percentile = st.number_input("Percentile", 50.0, 100.0, L.DEFAULT_PERCENTILE, 0.5, format="%g",
                                     help="99 puts the threshold above 99% of clean training windows. "
                                          "A lower value gives more alerts.")

    run = st.button("Run detection", type="primary", width="stretch", disabled=problem is not None)

current = L.make_config(
    dataset=dataset_name, window=window, stride=stride, model=option.name,
    params=raw, rule=rule, k=k, percentile=percentile,
)

# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

if run:
    # Clear first: a failure must never be displayed next to a previous result.
    st.session_state.pop("run", None)
    stages: dict[str, float] = {}
    try:
        with st.status("Running detection", expanded=True) as status:
            t0 = time.perf_counter()
            st.write("Building dataset")
            ds = get_dataset(dataset_name, int(window), int(stride))
            stages["dataset"] = time.perf_counter() - t0

            def on_stage(stage: str) -> None:
                st.write(f"{stage.capitalize()} ({option.label})")

            result = L.run_detection(
                option.name, ds, params=raw, threshold_rule=rule, threshold_k=k,
                threshold_percentile=percentile, progress=on_stage,
            )
            total = time.perf_counter() - t0
            status.update(
                label=f"Done in {total:.1f}s (dataset {stages['dataset']:.1f}s, fit {result.timings['fit']:.1f}s, "
                      f"score {result.timings['score']:.2f}s, evaluate {result.timings['evaluate']:.2f}s)",
                state="complete", expanded=False,
            )
        st.session_state["run"] = {"result": result, "dataset": ds, "config": current,
                                   "build_seconds": stages["dataset"], "total_seconds": total}
    except Exception as exc:
        st.error(f"**{type(exc).__name__}**: {exc}")
        with st.expander("Traceback"):
            st.exception(exc)
        st.stop()

entry = st.session_state.get("run")

# ---------------------------------------------------------------------------
# empty state
# ---------------------------------------------------------------------------

if entry is None:
    st.subheader("How to use")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**1. Data**\n\nPick a dataset, window and stride in the sidebar. A summary appears as soon as it builds.")
    c2.markdown("**2. Model**\n\nChoose a family and model, optionally adjust its parameters.")
    c3.markdown("**3. Threshold, then run**\n\nChoose how a score becomes an alert, and press **Run detection**.")
    st.info("Start with LOF or One-Class SVM: they take seconds. Deep models train a network on CPU and take longer "
            "(the UI trains for 15 epochs by default; lower the epochs for a quicker look).")
    st.subheader("Models")
    st.dataframe(L.model_table(), hide_index=True, width="stretch")
    with st.container(border=True):
        d1, d2 = st.columns([3, 1], vertical_alignment="center")
        d1.markdown("**New to these models?** The docs explain how each one works, "
                    "what every parameter does, and what its score means.")
        if d2.button("Open model docs", icon=":material/menu_book:", type="primary", width="stretch"):
            st.switch_page("pages/1_Model_docs.py")
    st.page_link("pages/2_Project_overview.py", icon=":material/hub:",
                 label="Project overview: the problem, the backend ETL pipeline and its status")
    st.stop()

result, ds, run_cfg = entry["result"], entry["dataset"], entry["config"]
changes = L.config_diff(run_cfg, current)
stale = bool(changes)

if stale:
    st.warning(
        "**Settings changed since this run.** The results below are for the previous settings, not the ones in the "
        f"sidebar. Press **Run detection** to update.\n\nResults are for: {L.describe_config(run_cfg)}\n\n"
        "Changed: " + "; ".join(changes)
    )

# ---------------------------------------------------------------------------
# results (everything below reads from `result` only)
# ---------------------------------------------------------------------------

with st.container(key="results-stale" if stale else "results-fresh"):
    head = f"{result.model} on {result.dataset}"
    st.subheader(head + (" :orange-badge[stale]" if stale else ""))
    st.markdown(
        f'<div class="runbar">{L.describe_config(run_cfg)} &nbsp;|&nbsp; threshold '
        f'<b>{result.threshold["rule"]} = {result.threshold["value"]:.6g}</b> &nbsp;|&nbsp; '
        f'{entry["total_seconds"]:.1f}s total</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    for col, (label, value, hint) in zip(cols, L.headline_metrics(result)):
        col.metric(label, value, help=hint, border=True)
    sec = st.columns(6)
    for col, (label, value) in zip(sec, L.secondary_stats(result)):
        col.caption(label)
        col.markdown(f"**{value}**")

    for warning in L.run_warnings(result):
        st.warning(warning)
    for note in L.interpretation(result):
        if note.startswith("No caveats"):
            st.info(note)
        else:
            st.warning(note)

    tab_scores, tab_break, tab_train, tab_attr, tab_details = st.tabs(
        ["Scores", "Breakdown", "Training", "Attribution", "Details"]
    )

    # -- scores -------------------------------------------------------------
    with tab_scores:
        frame = L.score_frame(result)
        allowed = L.log_scale_allowed(result)
        use_log = st.toggle(
            "Log scale", value=False, disabled=not allowed, key="log_scale",
            help="Needs every score and the threshold to be positive."
                 if allowed else "Unavailable: this model's scores or threshold include values <= 0.",
        ) and allowed
        y = alt.Y("score:Q", title="anomaly score",
                  scale=alt.Scale(type="log" if use_log else "linear", zero=False))
        line = alt.Chart(frame).mark_line(color=BLUE, strokeWidth=1.5).encode(
            x=alt.X("window:Q", title="window index"), y=y,
            tooltip=["window", alt.Tooltip("score:Q", format=".5g"), "actual anomaly", "flagged"],
        )
        layers = []
        spans = L.anomaly_spans(frame)
        if len(spans):
            layers.append(alt.Chart(spans).mark_rect(color=ORANGE, opacity=0.18).encode(
                x="start:Q", x2="end:Q"))
        layers.append(line)
        flagged = frame[frame["flagged"]]
        if len(flagged):
            layers.append(alt.Chart(flagged).mark_point(color=RED, filled=True, size=36).encode(
                x="window:Q", y="score:Q",
                tooltip=["window", alt.Tooltip("score:Q", format=".5g"), "actual anomaly"]))
        rule_df = pd.DataFrame({"threshold": [float(result.threshold["value"])]})
        layers.append(alt.Chart(rule_df).mark_rule(color=GREY, strokeDash=[6, 4], strokeWidth=2).encode(
            y="threshold:Q", tooltip=[alt.Tooltip("threshold:Q", format=".6g")]))
        st.altair_chart(alt.layer(*layers).properties(height=340).interactive(bind_y=False), width="stretch")
        st.caption(
            f"Blue line: score per window. Dashed grey: threshold ({result.threshold['rule']} = "
            f"{result.threshold['value']:.6g}). Shaded orange: windows overlapping a true anomaly. "
            f"Red dots: flagged windows ({int(frame['flagged'].sum()):,})."
        )

    # -- breakdown ----------------------------------------------------------
    with tab_break:
        left, right = st.columns(2)
        with left:
            st.markdown("**Recall by anomaly kind**")
            kinds = L.per_kind_table(result)
            if kinds.empty:
                st.info("No anomaly kind present in the test windows.")
            else:
                bar = alt.Chart(kinds).mark_bar(color=BLUE).encode(
                    x=alt.X("recall:Q", scale=alt.Scale(domain=[0, 1])), y=alt.Y("anomaly kind:N", sort="-x"),
                    tooltip=["anomaly kind", "recall", "detected"])
                st.altair_chart(bar.properties(height=max(90, 44 * len(kinds))), width="stretch")
                st.dataframe(kinds, hide_index=True, width="stretch")
                missing = sorted(set(("level_shift", "pattern", "time_shift")) - set(result.per_kind_recall))
                if missing:
                    st.caption("Kinds with no test window at this window/stride: "
                               + ", ".join(m.replace("_", " ") for m in missing))
        with right:
            st.markdown("**Confusion matrix**")
            cm = L.confusion_frame(result)
            base = alt.Chart(cm).encode(
                x=alt.X("predicted:N", sort=["flagged", "not flagged"], title=None),
                y=alt.Y("actual:N", sort=["anomaly", "normal"], title=None))
            heat = base.mark_rect(opacity=0.35, color=BLUE)
            text = base.mark_text(fontSize=18, fontWeight="bold").encode(text=alt.Text("count:Q", format=","))
            st.altair_chart((heat + text).properties(height=170), width="stretch")

        st.markdown("**Threshold rules compared (same fitted model, same k and percentile)**")
        rc = L.rule_comparison_table(result)
        rbar = alt.Chart(rc).mark_bar().encode(
            x=alt.X("F2:Q", scale=alt.Scale(domain=[0, 1])), y=alt.Y("rule:N", sort="-x"),
            color=alt.condition(alt.datum.active == "yes", alt.value(BLUE), alt.value(GREY)),
            tooltip=["rule", "recall", "F2", "FNR", "precision", "active"])
        st.altair_chart(rbar.properties(height=150), width="stretch")
        st.dataframe(
            rc.style.apply(
                lambda r: ["font-weight:600; background-color: rgba(76,120,168,.30)" if r["active"] == "yes" else ""] * len(r),
                axis=1),
            hide_index=True, width="stretch")
        st.caption("The highlighted row is the active rule and matches the headline numbers above.")

    # -- training -----------------------------------------------------------
    with tab_train:
        curve = L.training_curve(result)
        if len(curve):
            long = curve.melt("epoch", var_name="split", value_name="loss")
            st.altair_chart(
                alt.Chart(long).mark_line(point=True).encode(
                    x=alt.X("epoch:Q"), y=alt.Y("loss:Q"), color=alt.Color("split:N", title=None),
                    tooltip=["epoch", "split", alt.Tooltip("loss:Q", format=".5g")]
                ).properties(height=300), width="stretch")
            bits = [f"{result.extras['epochs_run']} epochs run"]
            if "stopped_early" in result.extras:
                bits.append("stopped early" if result.extras["stopped_early"] else "ran to the epoch limit")
            if result.extras.get("best_val_loss") is not None:
                bits.append(f"best validation loss {result.extras['best_val_loss']:.5g}")
            if result.extras.get("n_parameters"):
                bits.append(f"{result.extras['n_parameters']:,} parameters")
            st.caption(", ".join(bits))
        else:
            st.info("No training curve: this model is not trained by gradient descent over epochs "
                    "(classical models are fitted in one step).")

    # -- attribution --------------------------------------------------------
    with tab_attr:
        attr = L.attribution_frame(result)
        scope = result.extras.get("attribution_scope")
        if attr is None:
            st.info("Channel attribution is not available for this model family: it reduces each window to a single "
                    "distance and cannot say which channel caused it." if result.extras.get("family") == "classical"
                    else "Channel attribution is not available for this model.")
        else:
            st.altair_chart(
                alt.Chart(attr).mark_bar(color=BLUE).encode(
                    x=alt.X("contribution:Q", title="share of error", axis=alt.Axis(format="%")),
                    y=alt.Y("channel:N", sort="-x", title=None),
                    tooltip=["channel", alt.Tooltip("contribution:Q", format=".1%"),
                             alt.Tooltip("mean_error:Q", format=".4g")]
                ).properties(height=max(120, 40 * len(attr))), width="stretch")
            if scope == "all_windows":
                st.warning("No window was flagged, so this shows the average over all windows: which channel the model "
                           "struggles with in general, not which one caused a detection.")
            else:
                st.caption("Average share of error over the flagged windows. It names the channel the model failed to "
                           "predict, which is not always the channel that failed.")
            st.dataframe(attr, hide_index=True, width="stretch")

    # -- details ------------------------------------------------------------
    with tab_details:
        a, b = st.columns(2)
        with a:
            st.markdown("**Parameters**")
            st.dataframe(pd.DataFrame([(k_, str(v)) for k_, v in result.params.items()], columns=["parameter", "value"]),
                         hide_index=True, width="stretch")
            st.markdown("**Threshold statistics (training scores)**")
            st.dataframe(L.threshold_stats(result), hide_index=True, width="stretch")
        with b:
            st.markdown("**Forecast quality**")
            ft = L.forecast_table(result)
            if ft is not None:
                st.dataframe(ft, hide_index=True, width="stretch")
            elif result.extras.get("forecast_error"):
                st.warning(f"Not computed: {result.extras['forecast_error']}")
            else:
                st.info("Not defined for density models: they produce no prediction or reconstruction.")
            st.markdown("**Timings**")
            t = result.timings
            st.dataframe(pd.DataFrame(
                [("dataset build", f"{entry['build_seconds']:.2f}s"), ("fit", f"{t['fit']:.2f}s"),
                 ("score", f"{t['score']:.2f}s"), ("evaluate", f"{t.get('evaluate', 0.0):.2f}s")],
                columns=["stage", "time"]), hide_index=True, width="stretch")
        st.markdown("**Dataset**")
        st.caption(ds.describe())
        meta = {k_: v for k_, v in ds.meta.items() if k_ != "anomalies"}
        st.dataframe(pd.DataFrame([(k_, str(v)) for k_, v in meta.items()], columns=["field", "value"]),
                     hide_index=True, width="stretch")
        anomalies = ds.meta.get("anomalies") or []
        if anomalies:
            st.markdown("**Injected anomalies** (sample indices)")
            st.dataframe(pd.DataFrame(anomalies), hide_index=True, width="stretch")
        if result.extras.get("clusters"):
            with st.expander("Cluster report"):
                st.json(result.extras["clusters"])
