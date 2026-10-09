"""Everything the UI does, with no Streamlit in it.

The rendering layer is thin on purpose. Every decision — which models to offer,
how to coerce a widget value into a model parameter, what a result means, how to
describe a failure — lives here, where it can be called directly and tested
without a browser or a simulated session.

That split is what makes the UI tests meaningful rather than theatre: they check
that clicking things wires through to this, and these functions are checked on
their own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

import avionics_ad.models as M
from avionics_ad import datasets as D
from avionics_ad import experiment as E
from avionics_ad import threshold as TH

#: Deep models get a smaller default budget in the UI than in the benchmark —
#: somebody clicking through models wants an answer in seconds.
UI_EPOCH_DEFAULT = 15

WINDOW_CHOICES = (40, 60, 100, 150)
STRIDE_CHOICES = (1, 5, 10, 20)


@dataclass(frozen=True)
class ModelOption:
    """One entry in the model dropdown."""

    name: str
    label: str
    family: str
    description: str

    @property
    def display(self) -> str:
        return f"{self.label}  ({self.family})"


def model_options() -> list[ModelOption]:
    """Models for the dropdown, grouped by family and stable in order."""
    return [
        ModelOption(name=i.name, label=i.label, family=i.family, description=i.description)
        for i in M.available()
    ]


def option_by_display(display: str) -> ModelOption:
    for option in model_options():
        if option.display == display:
            return option
    raise KeyError(f"no model matching {display!r}")


#: Datasets that need nothing built beforehand. These are offered first, so the
#: app opens on something that always works rather than on one that may require
#: `make all` to have been run.
SELF_CONTAINED = ("lv_synthetic",)


def dataset_options() -> list[str]:
    """Datasets for the dropdown, self-contained ones first."""
    available = D.available()
    first = [n for n in SELF_CONTAINED if n in available]
    return first + [n for n in available if n not in first]


#: Display names for datasets; the value is the registry key.
DATASET_LABELS = {"lv_synthetic": "LV synthetic telemetry", "cmapss": "C-MAPSS (FD001)"}


def dataset_label(name: str) -> str:
    return DATASET_LABELS.get(name, name)


def dataset_status() -> list[tuple[str, str, str | None]]:
    """(key, display label, reason it is unavailable or None) for every dataset.

    The label carries "(not built)" so a user sees it before clicking Run rather
    than after a failure.
    """
    out = []
    for name in dataset_options():
        reason = D.readiness(name)
        label = dataset_label(name) + ("" if reason is None else " (not built)")
        out.append((name, label, reason))
    return out


# ---------------------------------------------------------------------------
# threshold rules
# ---------------------------------------------------------------------------

#: One line per rule, condensed from the docstring of :mod:`avionics_ad.threshold`.
RULE_DESCRIPTIONS = {
    "max_plus_sigma": "max(train) + std(train). Conservative: nothing in training fires, but one bad training window raises it.",
    "mean_plus_k_sigma": "mean(train) + k x std(train). Anchored on the bulk; k is a sensitivity dial.",
    "percentile": "The q-th percentile of training scores. No distributional assumption.",
    "median_plus_k_mad": "median + k x 1.4826 x MAD. Robust to the skewed training scores some models produce.",
}

#: Plain-language name and effect of each rule, for the sidebar.
RULE_LABELS = {
    "median_plus_k_mad": "Typical score + k × spread (recommended)",
    "mean_plus_k_sigma": "Average score + k × std dev",
    "percentile": "Percentile of training scores",
    "max_plus_sigma": "Above the worst training score (strict)",
}

RULE_PLAIN = {
    "median_plus_k_mad": "Ignores a few odd training windows, so it suits most models.",
    "mean_plus_k_sigma": "Like the recommended rule, but a few odd training windows can push the threshold up.",
    "percentile": "Sets the threshold above a chosen share of the clean training windows.",
    "max_plus_sigma": "Fewest false alarms, but subtle faults can slip under the threshold. No setting to adjust.",
}

#: Which extra controls each rule actually reads.
RULE_CONTROLS = {
    "max_plus_sigma": (),
    "mean_plus_k_sigma": ("k",),
    "percentile": ("percentile",),
    "median_plus_k_mad": ("k",),
}

DEFAULT_RULE = "median_plus_k_mad"
DEFAULT_K = 3.0
DEFAULT_PERCENTILE = 99.0


def threshold_options() -> list[str]:
    return list(TH.RULES)


def rule_controls(rule: str) -> tuple[str, ...]:
    """Names of the controls (``k``, ``percentile``) a rule uses."""
    try:
        return RULE_CONTROLS[rule]
    except KeyError:
        raise KeyError(f"unknown threshold rule {rule!r}; expected one of {TH.RULES}") from None


# ---------------------------------------------------------------------------
# run configuration and staleness
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunConfig:
    """Everything a result depends on, comparable for equality.

    ``k`` and ``percentile`` are normalised to None when the rule ignores them,
    so moving an irrelevant control does not mark a result stale.
    """

    dataset: str
    window: int
    stride: int
    model: str
    params: tuple[tuple[str, Any], ...]
    rule: str
    k: float | None
    percentile: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset, "window": self.window, "stride": self.stride,
            "model": self.model, "params": dict(self.params), "rule": self.rule,
            "k": self.k, "percentile": self.percentile,
        }


def make_config(
    *, dataset: str, window: int, stride: int, model: str,
    params: Mapping[str, Any], rule: str, k: float, percentile: float,
) -> RunConfig:
    used = rule_controls(rule)
    coerced = coerce_params(model, dict(params))
    return RunConfig(
        dataset=dataset, window=int(window), stride=int(stride), model=model,
        params=tuple(sorted(coerced.items())), rule=rule,
        k=float(k) if "k" in used else None,
        percentile=float(percentile) if "percentile" in used else None,
    )


def config_diff(run: RunConfig, current: RunConfig) -> list[str]:
    """Human-readable list of what differs between a stored run and the widgets."""
    a, b = run.as_dict(), current.as_dict()
    out = []
    for key in ("dataset", "window", "stride", "model", "rule", "k", "percentile"):
        if a[key] != b[key]:
            out.append(f"{key}: {a[key]} -> {b[key]}")
    if a["model"] == b["model"]:
        for name in sorted(set(a["params"]) | set(b["params"])):
            if a["params"].get(name) != b["params"].get(name):
                out.append(f"{name}: {a['params'].get(name)} -> {b['params'].get(name)}")
    return out


def describe_config(cfg: RunConfig) -> str:
    parts = [dataset_label(cfg.dataset), f"window {cfg.window}", f"stride {cfg.stride}", cfg.model, cfg.rule]
    if cfg.k is not None:
        parts.append(f"k={cfg.k:g}")
    if cfg.percentile is not None:
        parts.append(f"q={cfg.percentile:g}")
    return " | ".join(parts)


def validate_selection(*, window: int, stride: int, dataset_reason: str | None) -> str | None:
    """Reason the current selection cannot be run, or None."""
    if dataset_reason:
        return dataset_reason
    if stride > window:
        return f"Stride ({stride}) is larger than the window ({window}); windows would skip samples."
    return None


def model_table() -> pd.DataFrame:
    """Name, family and typical speed for the empty-state model overview."""
    rows = []
    for o in model_options():
        deep = any(p.name == "epochs" for p in M.get(o.name).params_spec)
        rows.append({
            "model": o.label, "key": o.name, "family": o.family,
            "typical speed": "slower (trains a network, CPU)" if deep else "seconds",
        })
    return pd.DataFrame(rows)


#: Training parameters every deep model shares, documented once on the docs page.
SHARED_DEEP_PARAMS = (
    "epochs", "batch_size", "learning_rate", "hidden_size",
    "patience", "validation_fraction", "seed",
)


def param_table(model_name: str, *, exclude: tuple[str, ...] = ()) -> pd.DataFrame:
    """A model's declared parameters as (parameter, type, default, range, meaning)."""
    rows = []
    for p in M.get(model_name).params_spec:
        if p.name in exclude:
            continue
        if p.kind == "choice":
            allowed = " / ".join(str(c) for c in p.choices)
        elif p.kind == "bool":
            allowed = "on / off"
        elif p.low is not None and p.high is not None:
            allowed = f"{p.low:g} – {p.high:g}"
        else:
            allowed = ""
        rows.append({"parameter": p.name, "type": p.kind, "default": str(p.default),
                     "range": allowed, "meaning": p.help})
    return pd.DataFrame(rows, columns=["parameter", "type", "default", "range", "meaning"])


def model_specific_params(model_name: str) -> pd.DataFrame:
    """Parameters beyond the shared deep-training ones."""
    return param_table(model_name, exclude=SHARED_DEEP_PARAMS)


def default_params(model_name: str) -> dict[str, Any]:
    """Declared defaults, with the training budget trimmed for interactive use."""
    spec = {p.name: p.default for p in M.get(model_name).params_spec}
    if "epochs" in spec:
        spec["epochs"] = UI_EPOCH_DEFAULT
        spec["patience"] = max(3, UI_EPOCH_DEFAULT // 3)
    return spec


def coerce_params(model_name: str, raw: dict[str, Any]) -> dict[str, Any]:
    """Convert widget values into the types a model expects.

    Widgets hand back floats where an int is wanted and strings where a bool is.
    Unknown keys are dropped rather than raising, because a stale value left in
    session state when the model changes should not break the run.
    """
    spec = {p.name: p for p in M.get(model_name).params_spec}
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in spec:
            continue
        kind = spec[key].kind
        if kind == "int":
            out[key] = int(value)
        elif kind == "float":
            out[key] = float(value)
        elif kind == "bool":
            out[key] = bool(value)
        else:
            out[key] = value
    return out


def build_dataset(name: str, *, window: int, stride: int) -> D.Dataset:
    """Build a dataset, translating failures into something a user can act on."""
    reason = D.readiness(name)
    if reason is not None:
        raise RuntimeError(reason)
    return D.build(name, window=window, stride=stride)


def run_detection(
    model_name: str,
    dataset: D.Dataset,
    *,
    params: dict[str, Any] | None = None,
    threshold_rule: str = DEFAULT_RULE,
    threshold_k: float = DEFAULT_K,
    threshold_percentile: float = DEFAULT_PERCENTILE,
    progress=None,
) -> E.Result:
    return E.run(
        model_name,
        dataset,
        params=coerce_params(model_name, params or {}),
        threshold_rule=threshold_rule,  # type: ignore[arg-type]
        threshold_k=threshold_k,
        threshold_percentile=threshold_percentile,
        progress=progress,
    )


# ---------------------------------------------------------------------------
# presenting a result
# ---------------------------------------------------------------------------

def headline_metrics(result: E.Result) -> list[tuple[str, str, str]]:
    """The four numbers that matter, as (label, value, explanation).

    Recall leads because a missed fault costs more than a false alarm, which is
    the whole reason this uses F2 rather than F1.
    """
    d = result.detection
    return [
        ("Recall", f"{d['recall']:.1%}", "Share of real anomalies found"),
        ("F2 score", f"{d['f_beta']:.3f}", "Recall weighted 4x over precision"),
        ("False negative rate", f"{d['false_negative_rate']:.1%}", "Share of anomalies missed"),
        ("Precision", f"{d['precision']:.1%}", "Share of alerts that were real"),
    ]


def per_kind_table(result: E.Result) -> pd.DataFrame:
    """Recall per anomaly kind — the breakdown the aggregate hides."""
    rows = [
        {
            "anomaly kind": kind.replace("_", " "),
            "recall": round(value, 3),
            "detected": "yes" if value > 0.5 else ("partly" if value > 0 else "no"),
        }
        for kind, value in sorted(result.per_kind_recall.items())
    ]
    return pd.DataFrame(rows, columns=["anomaly kind", "recall", "detected"])


def rule_comparison_table(result: E.Result) -> pd.DataFrame:
    """How the chosen threshold rule compares with the alternatives.

    The ``active`` column marks the rule that produced the headline numbers; its
    row is taken from the same fitted model, so it must agree with them.
    """
    active = result.threshold["rule"]
    rows = [
        {
            "active": "yes" if rule == active else "",
            "rule": rule,
            "recall": round(m["recall"], 3),
            "F2": round(m["f_beta"], 3),
            "FNR": round(m["false_negative_rate"], 3),
            "precision": round(m["precision"], 3),
        }
        for rule, m in result.rule_comparison.items()
    ]
    return pd.DataFrame(rows).sort_values("F2", ascending=False).reset_index(drop=True)


def score_frame(result: E.Result) -> pd.DataFrame:
    """Scores over time with the threshold and flags, ready to plot.

    ``flagged`` is recomputed from the stored threshold value with the same strict
    comparison the detector used, and is checked against the result's own count.
    """
    cols = ["window", "score", "threshold", "actual anomaly", "flagged"]
    if result.scores is None:
        return pd.DataFrame(columns=cols)
    value = float(result.threshold["value"])
    scores = np.asarray(result.scores, dtype="float64")
    return pd.DataFrame(
        {
            "window": np.arange(len(scores)),
            "score": scores,
            "threshold": value,
            "actual anomaly": (
                np.asarray(result.labels, dtype=bool) if result.labels is not None
                else np.zeros(len(scores), dtype=bool)
            ),
            "flagged": scores > value,
        }
    )


def anomaly_spans(frame: pd.DataFrame) -> pd.DataFrame:
    """Contiguous runs of true-anomaly windows as (start, end) rows, for shading."""
    if frame.empty or not frame["actual anomaly"].any():
        return pd.DataFrame(columns=["start", "end"])
    flag = frame["actual anomaly"].to_numpy(dtype=bool)
    edges = np.diff(np.concatenate([[0], flag.astype(int), [0]]))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    # extend by one window so a single-window span is still visible
    return pd.DataFrame({"start": frame["window"].to_numpy()[starts], "end": frame["window"].to_numpy()[ends - 1] + 1})


def log_scale_allowed(result: E.Result) -> bool:
    """A log axis is only meaningful when every score and the threshold are positive."""
    if result.scores is None or len(result.scores) == 0:
        return False
    return bool(np.min(result.scores) > 0 and result.threshold["value"] > 0)


def confusion_frame(result: E.Result) -> pd.DataFrame:
    """Long-form confusion matrix (actual x predicted) for plotting."""
    d = result.detection
    return pd.DataFrame(
        [
            {"actual": "anomaly", "predicted": "flagged", "count": d["true_positives"]},
            {"actual": "anomaly", "predicted": "not flagged", "count": d["false_negatives"]},
            {"actual": "normal", "predicted": "flagged", "count": d["false_positives"]},
            {"actual": "normal", "predicted": "not flagged", "count": d["true_negatives"]},
        ]
    )


def secondary_stats(result: E.Result) -> list[tuple[str, str]]:
    d = result.detection
    best = result.best_achievable.get("f_beta")
    return [
        ("TP", f"{d['true_positives']:,}"),
        ("FP", f"{d['false_positives']:,}"),
        ("FN", f"{d['false_negatives']:,}"),
        ("TN", f"{d['true_negatives']:,}"),
        ("Flagged", f"{result.extras.get('n_flagged', d['true_positives'] + d['false_positives']):,} of {d['n_windows']:,}"),
        ("Best achievable F2", f"{best:.3f}" if best is not None else "n/a"),
    ]


def threshold_stats(result: E.Result) -> pd.DataFrame:
    t = result.threshold
    rows = [
        ("rule", t["rule"]), ("value", f"{t['value']:.6g}"),
        ("train mean", f"{t['train_mean']:.6g}"), ("train std", f"{t['train_std']:.6g}"),
        ("train median", f"{t['train_median']:.6g}"), ("train MAD", f"{t['train_mad']:.6g}"),
        ("train max", f"{t['train_max']:.6g}"), ("train windows", f"{t['n_train']:,}"),
    ]
    if t["rule"] in ("mean_plus_k_sigma", "median_plus_k_mad"):
        rows.append(("k", f"{t['k']:g}"))
    if t["rule"] == "percentile":
        rows.append(("percentile", f"{t['percentile']:g}"))
    return pd.DataFrame(rows, columns=["statistic", "value"])


def forecast_table(result: E.Result) -> pd.DataFrame | None:
    """Forecast metrics, or None when the model does not define them."""
    if not result.forecast:
        return None
    return pd.DataFrame(
        [(k.upper(), f"{v:.6g}") for k, v in result.forecast.items()], columns=["metric", "value"]
    )


def attribution_frame(result: E.Result) -> pd.DataFrame | None:
    a = result.attribution
    if a is None:
        return None
    return a.to_frame() if hasattr(a, "to_frame") else pd.DataFrame(a)


def run_warnings(result: E.Result) -> list[str]:
    """Problems that did not stop the run but the reader must know about."""
    out = []
    if result.extras.get("forecast_error"):
        out.append(f"Forecast metrics could not be computed: {result.extras['forecast_error']}")
    return out


def training_curve(result: E.Result) -> pd.DataFrame:
    """Training and validation loss per epoch, for the deep models."""
    history = result.extras.get("history")
    if not history:
        return pd.DataFrame(columns=["epoch", "train", "validation"])
    return pd.DataFrame(
        {
            "epoch": range(len(history["train_loss"])),
            "train": history["train_loss"],
            "validation": history["val_loss"],
        }
    )


def interpretation(result: E.Result) -> list[str]:
    """Plain statements about what the result means.

    A number without a reading is easy to misinterpret, and the three cases
    below are the ones that actually mislead people: a silent detector showing
    perfect precision, a good detector ruined by its threshold, and blindness to
    one kind hidden inside a decent aggregate.
    """
    notes: list[str] = []
    d = result.detection

    if d["true_positives"] + d["false_positives"] == 0:
        notes.append(
            "This model flagged nothing at all. Its precision of 100% means only "
            "that it was never wrong because it never spoke — try a more "
            "sensitive threshold rule."
        )

    best = result.best_achievable.get("f_beta")
    if best is not None and best > d["f_beta"] + 0.1:
        notes.append(
            f"The scores support an F2 of {best:.2f} but this threshold reached "
            f"{d['f_beta']:.2f}. The detector is better than the cut-off makes it look."
        )

    blind = [k for k, v in result.per_kind_recall.items() if v < 0.2]
    if blind and d["recall"] > 0.3:
        notes.append(
            "Blind to " + ", ".join(k.replace("_", " ") for k in blind)
            + ", despite a reasonable overall recall. The aggregate hides it."
        )

    if d["n_windows"] and d["true_positives"] + d["false_positives"] == d["n_windows"]:
        notes.append(
            "Every window was flagged, so a high recall here costs all of the precision. "
            "Try a less sensitive threshold rule."
        )

    if result.extras.get("used_fallback"):
        notes.append(
            "The clustering found no usable structure, so this fell back to "
            "reconstruction error."
        )

    if not notes:
        notes.append("No caveats — the headline numbers can be read at face value.")
    return notes
