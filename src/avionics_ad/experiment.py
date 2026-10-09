"""Run one model against one dataset, and benchmark all of them.

A single entry point produces everything needed to judge a detector:

* how well it reproduces normal behaviour (MSE, MAE)
* how well it finds anomalies at the chosen threshold (recall, F2, FNR)
* **how well it finds each kind of anomaly separately**
* what the best achievable threshold would have scored
* which channel it blames
* how long it took

Per-kind recall is the most informative column and the easiest to overlook. An
aggregate recall of 0.8 could mean a detector that is mediocre at everything, or
one that is perfect on level shifts and completely blind to time shifts. Those
are very different models to put in front of a safety case, and only the
breakdown distinguishes them — it is the result the reference study reports as
its main comparison.

The best-achievable column separates two failure modes that look identical from
the headline number: a detector whose scores carry no signal, and a good
detector let down by its threshold rule.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

import avionics_ad.models as M
from avionics_ad import attribution as A
from avionics_ad import datasets as D
from avionics_ad import metrics as MT
from avionics_ad import threshold as TH
from avionics_ad.datasets import Dataset


@dataclass
class Result:
    """Everything one (model, dataset) run produced."""

    model: str
    dataset: str
    params: dict[str, Any]
    forecast: dict[str, float]
    detection: dict[str, float]
    threshold: dict[str, Any]
    best_achievable: dict[str, float]
    per_kind_recall: dict[str, float]
    timings: dict[str, float]
    rule_comparison: dict[str, dict[str, float]] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)
    scores: np.ndarray | None = None
    labels: np.ndarray | None = None
    attribution: pd.DataFrame | None = None

    def summary_row(self) -> dict[str, Any]:
        """One flat row, for the comparison table."""
        return {
            "model": self.model,
            "dataset": self.dataset,
            "mse": self.forecast.get("mse"),
            "mae": self.forecast.get("mae"),
            "precision": self.detection["precision"],
            "recall": self.detection["recall"],
            "f2": self.detection["f_beta"],
            "fnr": self.detection["false_negative_rate"],
            **{f"recall_{k}": v for k, v in self.per_kind_recall.items()},
            "best_f2": self.best_achievable.get("f_beta"),
            **{f"f2_{rule}": m["f_beta"] for rule, m in self.rule_comparison.items()},
            "fit_seconds": self.timings["fit"],
            "score_seconds": self.timings["score"],
        }

    def as_json(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("scores", None)
        d.pop("labels", None)
        d["attribution"] = (
            self.attribution.to_dict("records") if self.attribution is not None else None
        )
        return d


def per_kind_recall(labels: np.ndarray, flagged: np.ndarray, kinds: np.ndarray) -> dict[str, float]:
    """Recall computed separately for each anomaly kind.

    Each kind is scored against only its own windows, so a detector blind to one
    kind shows up as a zero rather than being averaged away.
    """
    out: dict[str, float] = {}
    for kind in sorted({k for k in kinds if k}):
        mask = (kinds == kind) & labels
        if not mask.any():
            continue
        out[kind] = float(flagged[mask].mean())
    return out


def run(
    model_name: str,
    dataset: Dataset,
    *,
    params: dict[str, Any] | None = None,
    threshold_rule: TH.Rule = "max_plus_sigma",
    threshold_k: float = 3.0,
    threshold_percentile: float = 99.0,
    threshold_offset: float = 0.0,
    beta: float = MT.DEFAULT_BETA,
    attribution_top: int = 5,
    progress: Callable[[str], None] | None = None,
) -> Result:
    """Fit a model on the clean training windows and score the test set.

    ``progress`` is called with "fitting", "scoring" and "evaluating" as each
    stage begins, so a front end can report where a long run is.
    """
    D.validate(dataset)
    model = M.create(model_name, **(params or {}))
    notify = progress or (lambda _stage: None)

    notify("fitting")
    started = time.perf_counter()
    model.fit(dataset.train)
    fit_seconds = time.perf_counter() - started

    notify("scoring")
    started = time.perf_counter()
    scores = model.score(dataset.test)
    score_seconds = time.perf_counter() - started

    notify("evaluating")
    eval_started = time.perf_counter()

    # Non-finite scores would make the threshold sweep and every metric
    # meaningless, so refuse them loudly rather than computing around them.
    train_scores = np.asarray(model.training_scores_, dtype="float64")
    for what, arr in (("training", train_scores), ("test", np.asarray(scores, dtype="float64"))):
        bad = int((~np.isfinite(arr)).sum())
        if bad:
            raise ValueError(
                f"{model_name} produced {bad} non-finite {what} score(s) out of {arr.size}; "
                "results would be meaningless. Try different parameters (e.g. lower learning rate)."
            )

    boundary = TH.fit(
        model.training_scores_,
        rule=threshold_rule,
        k=threshold_k,
        percentile=threshold_percentile,
        offset=threshold_offset,
    )
    flagged = boundary.apply(scores)

    detection = MT.detection_metrics(dataset.labels, flagged, beta=beta)

    # Every rule evaluated on the same fitted model, so the threshold rule can
    # be compared without refitting anything. The rule turns out to cost more
    # F2 than the choice of model does, which is only visible side by side.
    rule_comparison: dict[str, dict[str, float]] = {}
    for rule in TH.RULES:
        candidate = TH.fit(
            model.training_scores_, rule=rule, k=threshold_k,
            percentile=threshold_percentile, offset=threshold_offset,
        )
        rule_comparison[rule] = MT.detection_metrics(
            dataset.labels, candidate.apply(scores), beta=beta
        ).as_dict()

    # Forecast quality is only defined for models that produce a reconstruction
    # or a prediction. A density-based detector has none, and reporting zeros
    # there would be a lie dressed as a measurement.
    forecast: dict[str, float] = {}
    forecast_error: str | None = None
    if hasattr(model, "_predict"):
        try:
            target, prediction = model._predict(dataset.test)
            forecast = MT.forecast_metrics(target, prediction).as_dict()
        except Exception as exc:  # detection results stand; the failure is recorded, not hidden
            forecast_error = f"{type(exc).__name__}: {exc}"

    extras: dict[str, Any] = {"family": model.family, "n_flagged": int(flagged.sum())}
    if forecast_error:
        extras["forecast_error"] = forecast_error
    if getattr(model, "history", None) is not None and model.history.epochs_run:
        extras["epochs_run"] = model.history.epochs_run
        extras["stopped_early"] = model.history.stopped_early
        extras["best_val_loss"] = model.history.best_val_loss
        extras["history"] = {
            "train_loss": list(model.history.train_loss),
            "val_loss": list(model.history.val_loss),
        }
    if getattr(model, "cluster_report", None):
        extras["clusters"] = model.cluster_report
        extras["used_fallback"] = model.used_fallback
    if hasattr(model, "n_parameters"):
        extras["n_parameters"] = model.n_parameters

    # Attribute over the flagged windows where there are any. When a model
    # flags nothing, fall back to every window — "which channel does this model
    # struggle with in general" is still worth showing, and is recorded as a
    # different question rather than passed off as the same one.
    ranked = A.attribute(model, dataset.test, dataset.channels, flagged=flagged, top=attribution_top)
    extras["attribution_scope"] = "flagged"
    if ranked is None and flagged.sum() == 0:
        ranked = A.attribute(model, dataset.test, dataset.channels, top=attribution_top)
        extras["attribution_scope"] = "all_windows" if ranked is not None else "unavailable"
    elif ranked is None:
        extras["attribution_scope"] = "unavailable"

    return Result(
        model=model_name,
        dataset=dataset.name,
        params=dict(model.params),
        forecast=forecast,
        detection=detection.as_dict(),
        threshold={**asdict(boundary)},
        best_achievable=MT.best_achievable(scores, dataset.labels, beta=beta),
        per_kind_recall=per_kind_recall(dataset.labels, flagged, dataset.kinds),
        timings={
            "fit": fit_seconds,
            "score": score_seconds,
            "evaluate": time.perf_counter() - eval_started,
        },
        rule_comparison=rule_comparison,
        extras=extras,
        scores=scores,
        labels=dataset.labels,
        attribution=ranked,
    )


def benchmark(
    datasets: Sequence[Dataset],
    *,
    models: Sequence[str] | None = None,
    params: dict[str, dict[str, Any]] | None = None,
    threshold_rule: TH.Rule = "max_plus_sigma",
    beta: float = MT.DEFAULT_BETA,
    on_result=None,
) -> tuple[pd.DataFrame, list[Result]]:
    """Run every model against every dataset.

    A model that raises is recorded as a failed row rather than aborting the
    run: one broken configuration should not cost the other fifteen results.
    """
    models = list(models) if models is not None else M.names()
    params = params or {}
    rows: list[dict[str, Any]] = []
    results: list[Result] = []

    for dataset in datasets:
        for name in models:
            try:
                result = run(
                    name,
                    dataset,
                    params=params.get(name),
                    threshold_rule=threshold_rule,
                    beta=beta,
                )
                rows.append(result.summary_row())
                results.append(result)
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                rows.append({"model": name, "dataset": dataset.name, "error": f"{type(exc).__name__}: {exc}"})
            if on_result is not None:
                on_result(name, dataset.name, rows[-1])

    return pd.DataFrame(rows), results


def save(results: Sequence[Result], table: pd.DataFrame, out_dir: Path) -> Path:
    """Persist the benchmark so a later run can be compared against it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_dir / "benchmark.parquet", index=False)
    (out_dir / "benchmark.json").write_text(
        json.dumps([r.as_json() for r in results], indent=2, default=str)
    )
    return out_dir / "benchmark.parquet"


def comparison_table(table: pd.DataFrame) -> pd.DataFrame:
    """The reference study's comparison, ordered by the metric that matters."""
    cols = [c for c in ["model", "dataset", "mse", "mae", "recall", "f2", "fnr"] if c in table]
    kinds = sorted(c for c in table.columns if c.startswith("recall_"))
    out = table[cols + kinds].copy()
    numeric = out.select_dtypes("number").columns
    out[numeric] = out[numeric].round(4)
    return out.sort_values(["dataset", "f2"], ascending=[True, False]).reset_index(drop=True)
