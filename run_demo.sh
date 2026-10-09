#!/usr/bin/env bash
# Launch the demo on CPU only, so it does not compete with GPU jobs.
# Override with: THREADS=8 ./run_demo.sh
cd "$(dirname "$0")"
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS="${THREADS:-2}" MKL_NUM_THREADS="${THREADS:-2}" OPENBLAS_NUM_THREADS="${THREADS:-2}"
exec nice -n 10 .venv/bin/streamlit run src/avionics_ad/ui/app.py \
  --server.headless true --server.address 127.0.0.1 --server.port "${PORT:-8501}"
