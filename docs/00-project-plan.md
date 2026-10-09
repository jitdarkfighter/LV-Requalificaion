# Project Plan — AI-Powered Analytics Platform for Avionics Requalification

*Deliverable type: research proposal / design document (with a small evidence PoC). Timeline: 4–6 weeks, solo. Stack for PoC: lightweight local.*

---

## 1. Context

**What you were given.** Two things sit in `/home/jitdarkfighter/Projects/ETL-Avionics`:

- `Project-Avionics.pdf` — an ISRO/VSSC problem statement: *"AI-Powered Analytics Platform for Requalification of Avionics from Recovered Vehicle Stages."* It asks for ETL pipelines, a TB-scale warehouse, deterministic bound checks **plus** ML, anomaly detection with human-confirmed conditional clearance, dashboards, and image processing — handling ~64 GB of flight data.
- `data/` — the NASA **C-MAPSS** turbofan degradation dataset (FD001–FD004, ~160k rows, ~27 MB) and its dataset paper, Saxena, Goebel, Simon & Eklund, *"Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation,"* PHM'08.

**The gap that defines this project.** The problem statement is about *recovered launch-vehicle avionics*; the only data is *simulated jet-engine gas-path sensors*. Nobody has run-to-failure data for recovered avionics boards — that is exactly why the prof handed over C-MAPSS. The intellectual work of this proposal is therefore **not** "get a good RMSE on FD001." It is:

> Argue rigorously that C-MAPSS is a defensible *surrogate* for recovered-stage avionics telemetry, define the mapping precisely, state where the analogy holds and where it breaks, and design a platform whose architecture survives the swap when real VSSC data arrives.

**Intended outcome.** A submittable proposal document + architecture design + a small reproducible PoC whose figures back up the claims made in the document. Not a production platform.

---

## 2. Deliverables (what you hand in at week 6)

| # | Artifact | Form | Approx size |
|---|----------|------|-------------|
| D1 | **Proposal document** | Markdown → PDF (`docs/proposal.md`) | 20–30 pages |
| D2 | **Architecture design** | Diagrams (Mermaid/draw.io) embedded in D1 + `docs/architecture.md` | 5–8 figures |
| D3 | **Dataset justification & mapping** | A section of D1, built on a defensible mapping table | 3–5 pages |
| D4 | **Evidence PoC** | A small repo: `src/`, notebooks, ~8 result figures | ~1–2k LOC |
| D5 | **Feasibility & risk analysis** | Section of D1 — incl. the honest 64 GB / TB-scale argument | 3–4 pages |
| D6 | **Annotated bibliography** | `docs/references.bib` + one-line relevance note per entry | 20–30 refs |

D4 exists to serve D1. Every PoC experiment must earn its place by supporting a specific claim in the document. If it does not produce a figure or a number you cite, do not build it.

---

## 3. The core argument: C-MAPSS as an avionics surrogate

### 3.1 The mapping table (write this early — everything else depends on it)

| C-MAPSS concept | Avionics-requalification analogue | Analogy strength |
|---|---|---|
| Engine unit (100–260 per dataset) | A recovered avionics subsystem / LRU serial number | **Strong** — fleet of same-type units, unknown initial wear |
| Time in cycles | Mission number / flight-hours since manufacture | **Strong** |
| 3 operational settings (altitude, Mach, TRA) | Flight regime: ascent/coast/re-entry/landing phase, thermal-vac condition | **Strong** — both are exogenous context that must be normalized out |
| 21 gas-path sensors (T24, T30, T50, P30, Nf, Nc, epr, phi, NRf, BPR, htBleed, W31, W32 …) | Continuous avionics telemetry: rail voltages, board temperatures, current draw, clock drift, BER, link margin | **Moderate** — both are noisy continuous channels monotonically drifting with wear |
| Health index H(t) = min(margin_Fan, margin_HPC, margin_HPT, margin_EGT), failure at H = 0 | Requalification margin = min over subsystem operability margins (thermal, timing, power, signal-integrity) | **Strong** — *this is the single most transferable idea in the paper* |
| Exponential wear h(t) = 1 − d − exp{a·t^b}, a ∈ [0.001, 0.003], b ∈ [1.4, 1.6] | Cumulative damage from thermal cycling, vibration fatigue, radiation dose | **Moderate** — Arrhenius/Coffin-Manson (both discussed in §IV of the paper) are *the* standard electronics-reliability models, which strengthens the transfer |
| 6 fault modes / operating conditions (FD001–FD004) | Single- vs multi-condition, single- vs multi-fault requalification scenarios | **Strong** — gives you a built-in difficulty ladder |
| RUL prediction | "How many more missions can this box fly?" | **Strong** |
| PHM08 asymmetric score (late penalized harder) | Requalification risk asymmetry — clearing a box that later fails is far worse than scrapping a good one | **Strong** — cite this; it is the economic core of the whole problem statement |

**Where the analogy breaks — say this out loud in the document.** Gas-path degradation is smooth, monotone and thermodynamically constrained; avionics failures include abrupt latch-ups, single-event upsets, solder-joint cracking and connector intermittents that C-MAPSS cannot produce. Damage during a *single* flight is assumed negligible in C-MAPSS (§V.A of the paper) — that assumption is questionable for a re-entry/splashdown event, which is a high-stress single episode. State these as **limitations that motivate the synthetic avionics layer** (§3.2), not as fatal flaws.

### 3.2 The synthetic avionics layer (you chose this — it is the differentiator)

The problem statement asks for three data classes C-MAPSS simply does not contain. Synthesize them **as functions of the true degradation state**, so they are physically coherent rather than independent noise. Derive a normalized health proxy `h(t) ∈ [0,1]` per unit from the known RUL, then:

1. **Health bytes (fixed/discrete → pattern checking).** Per mission, emit a 16-byte telemetry health word per subsystem: BIT (built-in-test) pass/fail bits, mode flags, reset counters, checksum status. Flip-probability of each fault bit rises with `1 − h(t)`; add a small rate of transient, self-clearing flips. This gives the *pattern-checking* requirement real teeth — valid-state-machine transitions vs. impossible ones (e.g. `BIT_PASS` set while `SELFTEST_TIMEOUT` set).
2. **Vibration / acoustics / shock (continuous → bound checking).** Per mission, emit a short high-rate burst (e.g. 3-axis accelerometer). Base spectrum = sum of resonance peaks; degradation broadens and shifts peaks and raises the RMS floor as `h(t)` falls; inject occasional shock transients at re-entry/landing timestamps. This directly serves *"variable data like vibration, acoustics and shock shall be subjected to bound checks."*
3. **Image data (scope decision).** Propose an inspection-image channel (thermal/optical board imagery, e.g. solder-joint or conformal-coating defects) and **explicitly scope the implementation to future work**, with a named public surrogate dataset to be used later. Do not try to build this in 6 weeks.

**Framing note — C-MAPSS is itself simulated, and that is an asset, not a liability.** The data in `data/` is the batch output of a MATLAB/Simulink engine model with injected faults and superimposed noise (§III.A and §V of the paper); no real engine produced it. The problem statement explicitly anticipates exactly this kind of input — *"synthesised data from simulation test beds after fault injection can be provided as inputs."* **Say this early in the document.** It means:
- The base dataset's simulated nature is aligned with VSSC's own stated data strategy, not a compromise you're apologising for.
- Adding a synthetic avionics layer is *extending an existing simulation*, not contaminating real measurements — a much easier position to defend.
- The honest framing throughout is "physics-informed simulation surrogate," never "real flight data."

**Defensibility rules for synthetic data — state these explicitly or a reviewer will (rightly) attack the whole proposal:**
- Every synthetic generator is parameterized and its parameters are documented and justified against cited literature — mirroring how C-MAPSS grounded its own wear magnitudes in real-engine references (Table 3, from Chatterjee & Litt) and validated its response surfaces qualitatively against published real-engine behaviour (Goebel et al.). **Adopt the same two-step validation discipline for your generators; it is the single strongest defence available.**
- Synthetic channels are *never* used to claim ML accuracy on real degradation — only to demonstrate that the **pipeline mechanics** (pattern checks, bound checks, dashboard) function.
- Claims about degradation modelling rest on the C-MAPSS sensors alone (themselves simulated — label them as such in every results table).
- Provide a swap-in path: a documented schema contract so real VSSC channels replace synthetic ones without touching downstream code.

---

## 4. Proposed platform architecture (the design you are proposing)

Standard **medallion** layering, which maps cleanly onto the PDF's ETL ask:

```
Sources ──► BRONZE (raw, immutable, partitioned Parquet)
              │  space-separated telemetry, health-byte streams, vibration bursts
              ▼
            SILVER (typed, validated, conformed)
              │  unit/cycle keys, operating-condition clustering, sensor denoising,
              │  constant-sensor pruning, per-regime normalization
              ▼
            GOLD (analytics-ready marts)
              │  features per unit-cycle, RUL labels, health-index estimates,
              │  bound-check verdicts, decision registry
              ▼
   ┌──────────┴───────────┬──────────────────┐
 Deterministic         ML/AI              Dashboard
 engine (bound +       pipeline           (Streamlit)
 pattern checks)       (RUL, anomaly)
   └──────────┬───────────┴──────────────────┘
              ▼
       DECISION FUSION → verdict + human confirmation → decision registry
                                    │
                                    └──► feed-forward to next mission
```

**PoC stack (lightweight local, as chosen).** Python + Polars (or pandas) → Parquet on disk → **DuckDB** as the warehouse → Streamlit dashboard → scikit-learn / PyTorch for models. In the *proposal*, present this as the PoC tier and describe the production tier (object store + Iceberg/Delta, Airflow/Dagster orchestration, MLflow registry, Superset/Grafana) as the VSSC-deployment path. Explicitly argue **why DuckDB+Parquet is the right PoC choice** — columnar, zero-infra, larger-than-memory scans — and where it stops scaling.

### 4.1 The requalification decision framework (your novel contribution)

The PDF names three focus areas — *severe damage*, *performance deviations*, *minor anomalies*. Turn them into an explicit verdict ladder driven by **three independent evidence streams**:

| Evidence stream | Produces | Type |
|---|---|---|
| Bound checks + pattern checks | Hard violations, count & severity | Deterministic |
| RUL model | Predicted remaining missions + uncertainty band | ML |
| Anomaly detector | Novelty score vs. healthy-window baseline | ML |

Fuse into four verdicts: **REJECT** (severe damage) / **CONDITIONAL-CLEAR** (performance deviation, needs human sign-off) / **CLEAR-WITH-MONITORING** (minor anomaly) / **CLEAR**.

Two design points worth a section each:
- **Conditional clearance from precedent.** The PDF says *"conditional clearance based on decisions made during past missions… with human confirmation."* Design this as case-based retrieval: a **decision registry** table keyed by feature-space neighborhood; when a new unit lands near a past case, surface that precedent and its outcome to the reviewer. This is the "human-in-the-loop" requirement made concrete and is the most defensible AI-assist framing (recommend, never auto-decide).
- **Asymmetric cost.** Use the PHM08 scoring function (eq. 11 in the paper) as the model-selection criterion, not RMSE, and argue it mirrors the real requalification cost asymmetry. ⚠️ **Note the known discrepancy:** the paper states a₁ = 10, a₂ = 13 while asserting late predictions are penalized more heavily; the community-standard implementation uses **13 for early (d < 0), 10 for late (d ≥ 0)**. Flag this explicitly in the document, state which convention you use, and implement it as a single documented function. Reviewers notice this.
- Report the **correlation metric** alongside the score — §VII of the paper argues aggregate score alone can't distinguish consistently-good from mixed performance.

### 4.2 The 64 GB / terabyte claim — handle honestly

C-MAPSS train data is ~27 MB. Do not pretend otherwise. Make the volume argument from **first principles on the real data shape**:

> 3-axis vibration @ 20 kHz × float32 = 240 KB/s. A 10-second burst per mission ≈ 2.4 MB. 100 units × 200 missions = 20,000 bursts ≈ **48 GB** — before acoustics, before imagery, before multiple recovered stages.

That single calculation justifies the 64 GB figure and shows the volume lives in the *continuous* channels, not the cycle-summary telemetry. Then demonstrate the architecture handles it: partitioned Parquet + predicate pushdown + a DuckDB out-of-core scan benchmark on a generated multi-GB vibration table. **Run this benchmark at ~5–10 GB** (enough to prove the mechanism, small enough to not eat your laptop) and extrapolate with measured throughput.

---

## 5. PoC scope — the minimum that backs the document

Each item maps to a claim in D1. Build nothing else.

| PoC item | Supports which claim | Note |
|---|---|---|
| Loader + schema for FD001–FD004 | "ETL ingests multi-source telemetry" | 26 space-separated cols; **note the trailing-whitespace quirk** → 2 phantom empty columns if parsed naively |
| Operating-condition clustering (FD002/FD004) | "Platform normalizes out flight regime" | 6 conditions, cleanly separable by k-means on the 3 settings — a strong, cheap figure |
| Constant-sensor detection | "Automated data-quality profiling" | Several of the 21 sensors are constant in FD001 — show it, don't assert it |
| RUL baselines: linear → Random Forest → 1D-CNN/LSTM | "Deterministic + ML complement each other" | Use the standard **piecewise-linear RUL cap (≈125 cycles)** and justify why (early life carries no degradation signal) |
| Evaluation: RMSE **+ PHM08 score + correlation** | "Evaluation reflects operational cost asymmetry" | Compare against published FD001 benchmarks |
| Anomaly detection on healthy window | "Minor-anomaly detection without failure labels" | Autoencoder or Mahalanobis fit on first-N cycles |
| Synthetic health-byte + vibration generator | "Pattern & bound checks are implementable" | Per §3.2 |
| Bound/pattern check engine + verdict fusion | "The decision ladder works end-to-end" | Rule config in YAML, not hardcoded |
| Streamlit dashboard: fleet overview → unit drill-down | "Dashboarding, slicing & dicing, exec overview" | Screenshots go straight into D1 |
| DuckDB scale benchmark | "Architecture scales to 64 GB" | Per §4.2 |

**Scope out:** image processing (propose only), streaming/Kafka, real orchestration, MLOps, authentication, deployment.

---

## 6. Week-by-week schedule (6 weeks)

**Week 1 — Ground truth & framing**
- Read the Saxena paper closely; write `docs/dataset-analysis.md` summarizing the generation process (exponential wear model, H(t) = min of margins, noise model, 6 conditions, scoring function).
- Literature survey: 20–30 papers. Anchors: Ramasso & Saxena's CMAPSS benchmarking review (for *what counts as a good score* — essential so your numbers are contextualized), Saxena et al.'s prognostics-metrics paper, and the canonical deep-RUL line (Babu et al. CNN 2016; Zheng et al. LSTM 2017; Li et al. RESS 2018). ⚠️ Verify every citation against the actual paper before it enters `references.bib` — do not cite from memory.
- Draft the §3.1 mapping table. This is the spine of the document.
- **Milestone:** mapping table + annotated bibliography skeleton.

**Week 2 — Data reality check**
- Load all four datasets, EDA, condition clustering, constant-sensor profiling, degradation-trend plots.
- Write the loader/ETL bronze→silver skeleton.
- **Milestone:** 4–5 finished figures + `docs/dataset-analysis.md` complete.

**Week 3 — Architecture design**
- Write `docs/architecture.md`: medallion layers, schema contracts, warehouse choice rationale, PoC-vs-production tiers, the 64 GB argument.
- Design the decision framework (§4.1) and the decision-registry schema.
- **Milestone:** all architecture diagrams done; D2 draft complete.

**Week 4 — Modeling evidence**
- RUL baselines + PHM08 scoring + correlation metric; anomaly detector; comparison table vs. published benchmarks.
- **Milestone:** results table + 3 figures.

**Week 5 — Avionics layer + dashboard**
- Synthetic health-byte/vibration generators; bound- and pattern-check engine; verdict fusion; Streamlit dashboard; DuckDB scale benchmark.
- **Milestone:** end-to-end demo runs on one command; screenshots captured.

**Week 6 — Write and harden**
- Assemble D1 from the section drafts; limitations & risk analysis; future work (image processing, real-data swap-in, feed-forward across missions); proofread; verify every citation and every number in the document traces to a PoC output.
- **Milestone:** submittable PDF + tagged repo.

**Buffer note:** if you are at 4 weeks not 6, cut Week 5's dashboard to static matplotlib figures and drop the scale benchmark to a paper argument only. Do not cut Week 1 or Week 3 — the framing and the architecture *are* the deliverable.

---

## 7. Risks and how the document addresses each

| Risk | Mitigation in the proposal |
|---|---|
| "Turbofan data ≠ avionics" — the obvious attack | §3.1 mapping table + an explicit *Limitations of the Surrogate* section. Pre-empt it; do not hide it. |
| Synthetic data seen as circular / self-fulfilling | §3.2 defensibility rules: synthetic channels validate *mechanism*, never *accuracy*. |
| Overclaiming scale | §4.2 first-principles calculation + a measured benchmark with stated extrapolation. |
| ML numbers look unimpressive vs. SOTA | Frame as *baselines in a platform context*, cite the benchmarking review for realistic FD001 ranges, and note FD002/FD004 are substantially harder. |
| Label leakage in test evaluation | Use only `RUL_FD00x.txt` for test truth; never derive test labels from trajectory length. Document the test-set RUL range difference noted in §VI of the paper. |
| Scope creep into building a real platform | The deliverable is a document. The PoC exists only to produce cited figures. |

---

## 8. Repo layout

```
ETL-Avionics/
├── docs/
│   ├── 00-project-plan.md        # this plan, copied in as step 0
│   ├── proposal.md               # D1 — the main deliverable
│   ├── architecture.md           # D2
│   ├── dataset-analysis.md       # week 1–2 output
│   └── references.bib            # D6
├── data/                         # as-is (C-MAPSS + papers) — never modified
├── src/
│   ├── etl/          # bronze→silver→gold
│   ├── synth/        # health-byte + vibration generators
│   ├── checks/       # bound + pattern check engine (YAML-driven rules)
│   ├── models/       # RUL + anomaly
│   ├── decision/     # verdict fusion + registry
│   └── metrics.py    # RMSE, PHM08 score, correlation — one canonical impl
├── notebooks/        # EDA only; nothing load-bearing lives here
├── warehouse/        # DuckDB file + Parquet partitions (gitignored)
└── figures/          # every figure cited in docs/
```

---

## 9. Verification

The PoC is only useful if its numbers are trustworthy. Before any figure enters the document:

1. **Parse check** — `train_FD001.txt` must yield exactly 20,631 rows × 26 columns, 100 units. (Trailing spaces produce phantom columns; assert the shape.) Row counts to match: FD001 20,631 / FD002 53,759 / FD003 24,720 / FD004 61,249.
2. **Label check** — max cycle per training unit equals that unit's failure point; test RUL comes *only* from `RUL_FD00x.txt`. Assert train-unit counts: 100 / 260 / 100 / **249**; test-unit counts: 100 / 259 / 100 / **248**. ⚠️ `readme.txt` states FD004 has 248 train / 249 test — this is **wrong**, the files are the other way round (verified 2026-09-14). Worth a footnote in the document; it shows you actually opened the data.
3. **Condition clustering** — must recover exactly 6 clusters on FD002/FD004 and 1 on FD001/FD003. If it doesn't, the normalization is wrong.
4. **Metric check** — validate the PHM08 score function on hand-computed cases: d = 0 → 0; a late error must score higher than an early error of equal magnitude. Confirm against the paper's Fig. 9 shape.
5. **Benchmark sanity** — FD001 RMSE far outside published ranges (either direction) means a bug, most likely leakage or an unclipped RUL target. Investigate before publishing the number.
6. **End-to-end** — one command runs ingest → checks → model → verdict → dashboard on FD001; the dashboard renders a fleet view and a unit drill-down.
7. **Document trace** — every number and figure in `proposal.md` must be regenerable from a named script in `src/`. No orphan claims.

---

## 10. First actions on approval

1. Copy this file to `docs/00-project-plan.md` and `git init` the project.
2. Verify the `data/` extraction integrity against `CMAPSSData.zip` (the zip is still unextracted alongside it).
3. Write the loader + the shape assertions in §9.1–9.2 — everything downstream depends on parsing being right.
4. Start the Week 1 literature survey in parallel.
