# ETL-Avionics

Proof of concept for the ISRO/VSSC problem *AI-powered analytics platform for requalification of avionics from recovered vehicle stages*. NASA C-MAPSS turbofan data stands in for recovered-stage telemetry, which does not exist publicly. The deliverable is a research proposal (`docs/00-project-plan.md`); this code backs it.

| Part | Package | What it is |
|---|---|---|
| Backend ETL | `src/avionics_etl` | Raw → bronze → validated → silver → labels → gold Parquet layers, channel contracts with quarantine, lineage on every table, DuckDB warehouse. Single machine, no Spark; does not yet scale to 64 GB (see overview, *Status*) |
| Anomaly detection | `src/avionics_ad` | 8 detectors fitted on clean telemetry and scored on a copy with injected faults (level shift, pattern, time shift), judged recall-first (F2), in a Streamlit app |

**Full overview of the project, the pipeline layers and their status: [`docs/04-platform-overview.md`](docs/04-platform-overview.md)** (also in the app as the *Project overview* page).

## Setup and run

| Step | Command |
|---|---|
| Install | `python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt` (CPU torch: see comment in `requirements.txt`) |
| Run the app | `./run_demo.sh` (CPU only; `PORT=8599 THREADS=4 ./run_demo.sh` to override) |
| Tests | `PYTHONPATH=src .venv/bin/python -m pytest -q` |
| Build C-MAPSS (optional) | `make all` (about 2 min) |
| Full backend (optional) | `make warehouse` (about 6 min), then `PYTHONPATH=src .venv/bin/python -m avionics_etl.load.warehouse --demo` |

## Datasets

| Dataset | Status | How to get it |
|---|---|---|
| `lv_synthetic` | Always ready | Generated on the fly |
| `cmapss` (FD001) | Needs `build/silver/` | `make all`. Runs `raw`, `bronze`, `validate`, `silver` from `src/avionics_etl`. Reads `data/*.txt`, `configs/pipeline.yaml`, `configs/channels.yaml` |

The sidebar labels an unbuilt dataset "(not built)" and disables Run, with the reason.

`configs/pipeline.yaml` and `configs/channels.yaml` were reconstructed from what the ETL code reads (the originals were missing). Values the code needs but the repo does not define are my choices:

| Item | Choice |
|---|---|
| Paths | `build/{raw,bronze,silver,gold,quarantine,artifacts}` |
| `labels.rul_clip` | 125 |
| Outlier / feature / parquet settings | Conventional defaults (they do not affect anomaly detection) |
| Channel bounds | Observed min/max over the 8 data files, padded 10 percent; no `max_rate`; no units (not in `data/readme.txt`) |

Full backend: `make warehouse` (about 6 min; gold is the slow stage) builds labels, gold features and the DuckDB warehouse. The app does not need them. `configs/sources.yaml` was also reconstructed, from `data/readme.txt`.

C-MAPSS has no labelled anomalies, so faults are injected. With a long window some fault kinds fall outside the short test trajectories; the Breakdown tab lists the kinds that are missing. A window with no anomalous test windows is rejected with an error.

## Models

| Key | Family | Speed (CPU) | Channel attribution |
|---|---|---|---|
| `lof`, `ocsvm` | classical | seconds | no |
| `lstm`, `cnn_lstm`, `cnn_gru` | forecast | slower (trains a network) | yes |
| `lstm_ae`, `cnn_vae` | reconstruct | slower | yes |
| `tfa_cnn_ae_hdbscan` | cluster | slower | yes |

The UI trains deep models for 15 epochs by default. The **Model docs** page in the app (sidebar navigation) explains each model, its parameters and its score.

## Threshold rules

| Rule | Controls |
|---|---|
| `max_plus_sigma` | none |
| `mean_plus_k_sigma`, `median_plus_k_mad` | k |
| `percentile` | percentile |

## Layout

| Path | Role |
|---|---|
| `src/avionics_ad/ui/app.py` | Streamlit layer (rendering only) |
| `src/avionics_ad/ui/pages/1_Model_docs.py` | Model reference page (parameter tables read from the registry) |
| `src/avionics_ad/ui/pages/2_Project_overview.py` | Renders `docs/04-platform-overview.md` |
| `src/avionics_ad/ui/logic.py` | Decisions, staleness check, presentation tables (no Streamlit) |
| `src/avionics_ad/{datasets,experiment,threshold,metrics,attribution}.py`, `models/` | Core |
| `src/avionics_etl/` | Synthetic generator and C-MAPSS ETL |
| `tests/` | Unit tests plus Streamlit `AppTest` end-to-end tests |
