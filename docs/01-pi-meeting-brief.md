# PI Meeting Brief — 2026-09-15

*Project: AI-Powered Analytics Platform for Requalification of Avionics from Recovered Vehicle Stages (ISRO/VSSC problem statement) · Day 4 of a 4–6 week solo effort · Deliverable = research proposal + evidence PoC*

---

## 1. The 30-second pitch (lead with this)

> The problem statement is about recovered launch-vehicle avionics. The only data available is NASA C-MAPSS — simulated turbofan gas-path sensors. Nobody has run-to-failure data for recovered avionics boards, so the contribution of this project is **not** RUL accuracy on FD001. It is: defend C-MAPSS as a *surrogate*, define the mapping precisely, state where the analogy breaks, and design an architecture that survives the swap when real VSSC data arrives.

If the PI hears one thing, it should be **"surrogate argument + architecture, not a leaderboard score."**

---

## 2. Status — what physically exists today

| Item | State |
|---|---|
| `docs/00-project-plan.md` | Complete — 6-week plan, deliverables D1–D6, risk table |
| `src/cmapss.py` | Loader for FD001–FD004 with unit-count assertions, RUL capping, constant-sensor detection |
| `src/metrics.py` | RMSE + PHM'08 asymmetric score (one canonical implementation) |
| `notebooks/01-preprocessing.ipynb` | Full EDA + first modelling result (§4 below) |
| `explore.py` | Reproduces the dataset fact table (§3) in one command |
| Proposal document (D1) | **Not started** — Week 6 assembly, sections drafted Weeks 1–5 |
| Architecture doc, synthetic layer, dashboard | Not started (Weeks 3, 5) |

Not yet a git repo. Worth doing before the next checkpoint.

---

## 3. Dataset facts — have these ready

| | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| Train rows | 20,631 | 53,759 | 24,720 | 61,249 |
| Train / test units | 100 / 100 | 260 / 259 | 100 / 100 | **249 / 248** |
| Operating conditions | 1 | 6 | 1 | 6 |
| Fault modes | 1 (HPC) | 1 (HPC) | 2 (HPC+Fan) | 2 (HPC+Fan) |
| Lifetime min/median/max | 128/199/362 | 128/199/378 | 145/220/525 | 128/234/543 |
| Test true RUL median | 86 | 80 | 77 | 88 |
| Constant sensors | 6/21 | 0 | 5/21 | 0 |

Total: ~160k rows, ~27 MB. **The problem statement asks for 64 GB.** See §6.

**Point worth making out loud:** `data/readme.txt` says FD004 has 248 train / 249 test. The files are the reverse (verified). Small thing, but it signals you actually opened the data rather than trusting documentation.

---

## 4. The one real result so far

Question asked: *the data is simulated — does it still need preprocessing?* Tested rather than assumed.

FD001, random forest, scored on last cycle of each test unit:

| Configuration | RMSE | PHM'08 |
|---|---|---|
| Raw sensors, raw target | 32.9 | 34,632 |
| Raw sensors, **target capped at 125** | 18.3 | 1,016 |
| Dead sensors dropped, capped | 18.3 | 986 |

FD002, six operating conditions:

| Configuration | RMSE | PHM'08 |
|---|---|---|
| Forest, raw | 29.2 | 11,778 |
| Forest, condition-normalised | 29.6 | 13,646 |
| Ridge, raw | 32.6 | 19,674 |
| Ridge, condition-normalised | 32.5 | 19,073 |

**Verdict to present:** preprocessing splits in two. The *janitorial* half (missing values, outliers, unit fixes) is genuinely skippable — the data is simulator output, nothing is malformed. The half that matters is *representation*: assuming an engine degrades from cycle one is a modelling error the simulator cannot save you from, and it costs more than every feature decision combined (34× on the asymmetric score).

Condition normalisation made the forest slightly **worse** — because the operating settings are themselves feature columns, so a tree reconstructs the regimes internally and pre-normalising only discards information. Published gains are reported for *sequence models over sliding windows*; that is the honest place to revisit it.

*This is a good thing to walk the PI through — it's a negative result, tested, with a mechanism.* FD001 RMSE 18.3 is in the normal published band for a per-row tree baseline; deep sequence models sit lower. Frame as a platform baseline, not a SOTA attempt.

---

## 5. The four decisions to get confirmed

| # | Decision | Rationale to give | What confirmation looks like |
|---|---|---|---|
| D-1 | Deliverable is a **proposal + design document**, not a working platform | No avionics run-to-failure data exists; a 6-week solo build of a TB-scale VSSC platform is not credible | PI agrees the written argument is the graded artifact |
| D-2 | **Synthesize** the avionics-specific channels (health bytes, vibration/shock) on top of C-MAPSS, as functions of the true degradation state | The problem statement needs pattern checks and bound checks; C-MAPSS contains neither. Synthetic channels demonstrate *mechanism*, never *accuracy* | PI accepts synthetic layer with stated defensibility rules |
| D-3 | **Image processing scoped to future work**, with a named public surrogate dataset proposed | Explicitly in the problem statement, but not buildable in 6 weeks alongside everything else | PI confirms this is an acceptable omission |
| D-4 | PoC stack stays lightweight local — Python + Parquet + DuckDB + Streamlit | Zero-infra, columnar, larger-than-memory scans. Production tier (object store + Iceberg, Airflow, MLflow) is *described*, not built | PI doesn't expect Spark/cloud |

---

## 6. Likely PI questions — prepared answers

| Question | Answer |
|---|---|
| **"Turbofan data isn't avionics. Why is this valid?"** | It's a surrogate, defended explicitly in a mapping table. The strongest transferable idea is the health-index construction: H(t) = min over operability margins, failure at H=0. That maps directly onto a requalification margin = min over thermal / timing / power / signal-integrity margins. Also strong: fleet of same-type units with unknown initial wear, exogenous operating regime that must be normalised out, and the asymmetric cost of a late prediction. |
| **"Where does the analogy break?"** | Gas-path degradation is smooth, monotone, thermodynamically constrained. Avionics fails abruptly — latch-ups, single-event upsets, solder-joint cracking, connector intermittents. Also, C-MAPSS assumes damage within a single flight is negligible; that assumption is questionable for a re-entry/splashdown event. Say this in the document; it motivates the synthetic layer rather than undermining the work. |
| **"Isn't C-MAPSS itself fake data?"** | Yes — MATLAB/Simulink output with injected faults and superimposed noise. That's an **asset**: the problem statement explicitly says *"synthesised data from simulation test beds after fault injection can be provided as inputs."* So the base data matches VSSC's own stated data strategy, and the synthetic avionics layer extends an existing simulation rather than contaminating real measurements. Framing throughout is "physics-informed simulation surrogate," never "real flight data." |
| **"27 MB vs the 64 GB requirement?"** | First-principles, from the real data shape: 3-axis vibration @ 20 kHz × float32 = 240 KB/s → a 10 s burst per mission ≈ 2.4 MB → 100 units × 200 missions ≈ **48 GB**, before acoustics, imagery, or multiple recovered stages. The volume lives in the *continuous* channels, not the cycle-summary telemetry. Plan is to benchmark a DuckDB out-of-core scan at 5–10 GB and extrapolate with measured throughput. |
| **"Isn't synthetic data circular?"** | Guarded by explicit rules: every generator is parameterized with literature-justified parameters (mirroring how C-MAPSS grounded its own wear magnitudes against real-engine references and validated response surfaces qualitatively); synthetic channels validate pipeline *mechanics* only; all degradation-modelling claims rest on the C-MAPSS sensors alone; a documented schema contract lets real VSSC channels swap in without touching downstream code. |
| **"Why is your RMSE worse than published deep models?"** | These are deliberate baselines in a platform context, not a SOTA attempt. FD002/FD004 are substantially harder than FD001. Model selection uses the PHM'08 asymmetric score, not RMSE, because clearing a box that later fails costs far more than scrapping a good one. |
| **"What's actually novel here?"** | The requalification decision framework: three independent evidence streams (deterministic bound+pattern checks / RUL model with uncertainty / anomaly novelty score) fused into a four-level verdict ladder — REJECT, CONDITIONAL-CLEAR, CLEAR-WITH-MONITORING, CLEAR. Plus conditional clearance as **case-based retrieval**: a decision registry keyed by feature-space neighbourhood surfaces the precedent case and its outcome to a human reviewer. Recommend, never auto-decide. That's the problem statement's human-in-the-loop clause made concrete. |

---

## 7. Two technical footnotes worth knowing cold

1. **PHM'08 score convention discrepancy.** The Saxena et al. paper gives a₁ = 10, a₂ = 13 while asserting late predictions are penalised more heavily. The community-standard implementation uses **13 for early (d < 0), 10 for late (d ≥ 0)**. This repo uses the community convention, documented in `src/metrics.py`. Flag it in the proposal — reviewers notice.
2. **Report correlation alongside the score.** §VII of the paper argues an aggregate score alone cannot distinguish consistently-good performance from mixed performance. Not yet implemented.

---

## 8. What to ask the PI for

1. **Is the surrogate framing acceptable, or is there VSSC/ISRO data that could actually be obtained?** — this is the single highest-leverage question; a real dataset changes the whole project.
2. **Who is the audience for the proposal** — internal evaluation, an ISRO submission, or a paper? Changes length, tone, and citation depth.
3. **Deadline and page budget.** Plan assumes 20–30 pages at ~6 weeks (≈ 24 Oct). Confirm or compress.
4. **Is there a domain contact at VSSC** who could sanity-check the avionics side of the mapping table? The weakest part of the argument is the analogue column, and a 20-minute conversation with someone who works on recovered hardware would strengthen it more than a week of reading.
5. **Bibliography expectations** — 20–30 refs planned. Enough?

---

## 9. Next two weeks (state this as the commitment)

| Week | Output |
|---|---|
| Week 1 (now → 19 Sep) | Close reading of Saxena et al. → `docs/dataset-analysis.md`; the §3.1 mapping table; bibliography skeleton (20–30 refs, each verified against the actual paper) |
| Week 2 (22–26 Sep) | Condition clustering + constant-sensor profiling figures; bronze→silver ETL skeleton; 4–5 finished figures |
| Week 3 | `docs/architecture.md` — medallion layers, schema contracts, 64 GB argument, decision-registry schema |

Fallback if the timeline compresses to 4 weeks: cut the Streamlit dashboard to static figures and reduce the scale benchmark to a paper argument. **Do not cut Weeks 1 and 3** — the framing and the architecture *are* the deliverable.
