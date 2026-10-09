"""Anomaly detection over launch-vehicle telemetry.

Sits on top of :mod:`avionics_etl`: that package turns raw sources into clean,
normalised, windowed tensors; this one detects anomalies in them.
"""
