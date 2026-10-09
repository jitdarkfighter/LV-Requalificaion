"""Forecasting detectors.

The premise: train a model to predict what telemetry does next, using only data
where nothing was wrong. It learns normal dynamics. When something abnormal
happens the prediction misses, and the size of the miss is the anomaly score.

This is the family the reference study found most effective, and the reason is
structural — a forecaster is forced to learn *temporal* behaviour, so it can
notice a signal arriving at the wrong time even though its amplitude statistics
are unchanged. Nothing that looks only at the distribution of values can do
that.

The three architectures differ in how they read the input window:

``lstm``
    Recurrence alone. Processes the window step by step, carrying state. Strong
    on long-range temporal dependence; sees each timestep's channels only as a
    vector, so cross-channel structure has to be learned entirely through the
    recurrent state.

``cnn_lstm``
    Convolution first, then recurrence. The 1-D convolutions slide over time
    across all channels at once, so they pick up *local* joint patterns — a
    current step and the voltage sag that accompanies it — and hand a shorter
    sequence of richer features to the LSTM. The reference study's proposed
    model, and the one it reports as best.

``cnn_gru``
    The same idea with GRU cells, which have two gates rather than three. Fewer
    parameters and faster to train, at some cost in capacity.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from avionics_ad.models.base import ParamSpec
from avionics_ad.models.registry import register
from avionics_ad.models.torch_base import COMMON_PARAMS, TorchModel

FORECAST_PARAMS: tuple[ParamSpec, ...] = COMMON_PARAMS + (
    ParamSpec("horizon", "int", 1, "Steps ahead to predict", 1, 50),
    ParamSpec("num_layers", "int", 2, "Recurrent layers", 1, 4),
    ParamSpec("dropout", "float", 0.1, "Dropout between recurrent layers", 0.0, 0.8),
)

CONV_PARAMS: tuple[ParamSpec, ...] = FORECAST_PARAMS + (
    ParamSpec("conv_channels", "int", 64, "Convolution filters", 8, 256),
    ParamSpec("kernel_size", "int", 5, "Convolution kernel width in timesteps", 2, 15),
)


class _Forecaster(TorchModel):
    """Shared behaviour: split a window into history and the steps that follow."""

    family = "forecast"
    params_spec = FORECAST_PARAMS

    def _prepare(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = int(self.params["horizon"])
        if X.shape[1] <= h:
            raise ValueError(
                f"{self.name}: window of {X.shape[1]} steps is too short to predict "
                f"{h} ahead; use a longer window or a shorter horizon"
            )
        return X[:, :-h, :], X[:, -h:, :]


def _recurrent_head(torch_nn, kind: str, input_size: int, hidden: int, layers: int, dropout: float):
    cell = {"lstm": torch_nn.LSTM, "gru": torch_nn.GRU}[kind]
    return cell(
        input_size=input_size,
        hidden_size=hidden,
        num_layers=layers,
        batch_first=True,
        dropout=dropout if layers > 1 else 0.0,
    )


def _build_module(model: _Forecaster, n_channels: int, *, conv: bool, cell: str):
    """Assemble the network. Kept as a closure so torch imports stay lazy."""
    import torch
    import torch.nn as nn

    p = model.params
    hidden = int(p["hidden_size"])
    horizon = int(p["horizon"])
    layers = int(p["num_layers"])
    dropout = float(p["dropout"])

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm_in = nn.LayerNorm(n_channels)
            if conv:
                k = int(p["kernel_size"])
                filters = int(p["conv_channels"])
                self.conv = nn.Sequential(
                    # padding='same' keeps the sequence length, so the recurrent
                    # layer still sees one feature vector per original timestep
                    # and the window length does not have to be recomputed.
                    nn.Conv1d(n_channels, filters, kernel_size=k, padding="same"),
                    nn.ReLU(),
                    nn.Conv1d(filters, filters, kernel_size=k, padding="same"),
                    nn.ReLU(),
                )
                rnn_input = filters
            else:
                self.conv = None
                rnn_input = n_channels

            self.rnn = _recurrent_head(nn, cell, rnn_input, hidden, layers, dropout)
            self.norm_out = nn.LayerNorm(hidden)
            self.head = nn.Linear(hidden, horizon * n_channels)

        def forward(self, x):                      # (B, L, P)
            x = self.norm_in(x)
            if self.conv is not None:
                x = self.conv(x.transpose(1, 2)).transpose(1, 2)   # Conv1d wants (B, C, L)
            out, _ = self.rnn(x)
            last = self.norm_out(out[:, -1, :])    # the state after the whole window
            return self.head(last).reshape(-1, horizon, n_channels)

    return Net()


@register
class LSTMForecaster(_Forecaster):
    name = "lstm"
    label = "LSTM (forecasting)"
    description = """
    Recurrence only. Reads the window one timestep at a time, carrying state
    forward, and predicts what comes next. The simplest of the deep models and
    the natural baseline for the hybrids: if adding convolutions does not beat
    this, the extra machinery is not earning its place.
    """

    def _build_net(self, n_channels: int, window: int):
        return _build_module(self, n_channels, conv=False, cell="lstm")


@register
class CNNLSTMForecaster(_Forecaster):
    name = "cnn_lstm"
    label = "CNN + LSTM (forecasting)"
    params_spec = CONV_PARAMS
    description = """
    Convolutions extract local patterns across channels and time, then an LSTM
    models how those patterns evolve over the window. The combination is aimed
    squarely at spatio-temporal anomalies, where a fault shows up as a
    *relationship* between parameters breaking rather than any one of them
    leaving its limits. The reference study's proposed architecture.
    """

    def _build_net(self, n_channels: int, window: int):
        return _build_module(self, n_channels, conv=True, cell="lstm")


@register
class CNNGRUForecaster(_Forecaster):
    name = "cnn_gru"
    label = "CNN + GRU (forecasting)"
    params_spec = CONV_PARAMS
    description = """
    The same convolutional front end feeding GRU cells instead of LSTM. Two
    gates rather than three means roughly a quarter fewer recurrent parameters
    and faster training, which matters when a model has to be retrained per test
    campaign. The reference study found it close behind CNN+LSTM with lower
    complexity.
    """

    def _build_net(self, n_channels: int, window: int):
        return _build_module(self, n_channels, conv=True, cell="gru")
