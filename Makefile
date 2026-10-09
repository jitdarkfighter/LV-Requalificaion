# Builds the C-MAPSS layers the `cmapss` dataset in the demo reads (silver/measurements).
# Needs configs/pipeline.yaml and configs/channels.yaml.
#   make all            (uses .venv/bin/python; override with PY=python)
PY ?= .venv/bin/python
export PYTHONPATH := src

.PHONY: all raw bronze validate labels silver gold warehouse clean-build

all: raw bronze validate silver

raw:
	$(PY) -m avionics_etl.extract.raw
bronze: raw
	$(PY) -m avionics_etl.extract.ingest
validate: bronze
	$(PY) -m avionics_etl.transform.validate
labels: validate
	$(PY) -m avionics_etl.transform.labels
silver: validate
	$(PY) -m avionics_etl.transform.silver
gold: silver labels
	$(PY) -m avionics_etl.transform.gold
warehouse: gold
	$(PY) -m avionics_etl.load.warehouse

clean-build:
	rm -rf build
