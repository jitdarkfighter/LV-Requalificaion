"""Reconstruction detectors.

A different premise from forecasting. Instead of predicting what comes next, the
model is trained to compress a window into a narrow latent representation and
rebuild it. Trained only on normal data, the bottleneck learns to encode normal
behaviour efficiently. Abnormal behaviour has no efficient encoding, so it comes
back distorted, and the reconstruction error is the anomaly score.

Forecasting versus reconstruction
---------------------------------
Forecasting asks "is the next value surprising?"; reconstruction asks "does this
window look like the kind of thing I was trained on?" The second needs no
assumption that the signal is predictable, which helps where normal operation is
genuinely noisy — a forecaster penalises unpredictability it cannot avoid.

The cost is the mirror image: an autoencoder with enough capacity learns to copy
its input, anomalies included, and the score collapses toward zero everywhere.

How exposed a model is to that depends on its architecture, and the two here
differ. ``lstm_ae`` compresses the whole window into a single hidden state and
repeats that one vector across the decoder, so there is no path for per-timestep
detail to leak through — measured on sinusoidal test data, widening the latent
from 2 to 512 did not degrade detection at all. ``cnn_vae`` has such a path, and
guards it with the divergence term instead.

``lstm_ae``
    Recurrent encoder compresses the window into a single hidden state; that
    state is repeated and a recurrent decoder expands it back. The reference
    study's reconstruction baseline.

``cnn_vae``
    Convolutional, and *variational*: the encoder emits a distribution over the
    latent rather than a point, and the loss adds a divergence term pulling that
    distribution toward a standard normal. The regularisation is what stops the
    network simply memorising, which is the plain autoencoder's failure mode.
"""

from __future__ import annotations

import numpy as np

from avionics_ad.models.base import ParamSpec
from avionics_ad.models.registry import register
from avionics_ad.models.torch_base import COMMON_PARAMS, TorchModel

RECON_PARAMS: tuple[ParamSpec, ...] = COMMON_PARAMS + (
    ParamSpec("latent_size", "int", 16, "Bottleneck width", 2, 512),
    ParamSpec("num_layers", "int", 1, "Encoder/decoder layers", 1, 3),
)

VAE_PARAMS: tuple[ParamSpec, ...] = RECON_PARAMS + (
    ParamSpec("conv_channels", "int", 32, "Convolution filters", 8, 256),
    ParamSpec("kernel_size", "int", 5, "Convolution kernel width in timesteps", 2, 15),
    ParamSpec("kl_weight", "float", 0.001, "Weight on the divergence term", 0.0, 1.0),
    ParamSpec("score_includes_kl", "bool", False, "Add the divergence to the anomaly score"),
)


class _Reconstructor(TorchModel):
    """Shared behaviour: the target is the input itself."""

    family = "reconstruct"
    params_spec = RECON_PARAMS

    def _prepare(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return X, X


@register
class LSTMAutoencoder(_Reconstructor):
    name = "lstm_ae"
    label = "LSTM Autoencoder"
    description = """
    A recurrent encoder reads the window and compresses it into one hidden
    state; that state is repeated across the window length and a recurrent
    decoder expands it back out. Everything the window contained has to pass
    through the bottleneck, so only patterns seen during training survive the
    round trip intact.

    Strong on level and pattern anomalies. Weaker on time shifts, because a
    delayed-but-otherwise-correct window still contains familiar shapes and can
    be reconstructed well — which is what the reference study observed.

    The repeat-vector decoder is itself a hard bottleneck: one vector has to
    regenerate every timestep, so the usual autoencoder failure of learning to
    copy the input cannot happen here whatever the latent width.
    """

    def _build_net(self, n_channels: int, window: int):
        import torch
        import torch.nn as nn

        latent = int(self.params["latent_size"])
        hidden = int(self.params["hidden_size"])
        layers = int(self.params["num_layers"])

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm_in = nn.LayerNorm(n_channels)
                self.encoder = nn.LSTM(n_channels, hidden, num_layers=layers, batch_first=True)
                self.to_latent = nn.Linear(hidden, latent)
                self.from_latent = nn.Linear(latent, hidden)
                self.decoder = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True)
                self.head = nn.Linear(hidden, n_channels)
                self.window = window

            def forward(self, x):
                x = self.norm_in(x)
                _, (h, _) = self.encoder(x)
                z = self.to_latent(h[-1])                        # the bottleneck
                seed = self.from_latent(z).unsqueeze(1)
                repeated = seed.repeat(1, self.window, 1)        # one latent, many steps
                out, _ = self.decoder(repeated)
                return self.head(out)

        return Net()


@register
class CNNVAE(_Reconstructor):
    name = "cnn_vae"
    label = "CNN + Variational Autoencoder"
    params_spec = VAE_PARAMS
    description = """
    Convolutions compress the window across time and channels into a *distribution*
    over the latent space rather than a single point, and the loss adds a
    Kullback-Leibler term pulling that distribution toward a standard normal.

    That regularisation is the reason to prefer it over a plain autoencoder: it
    stops the network allocating a private corner of latent space to every
    training window, which is how autoencoders quietly learn to copy their input
    and lose all detection power. The trade-off is that the sampling makes
    scores slightly noisy, which the implementation removes at inference by
    using the distribution's mean rather than drawing from it.
    """

    def _build_net(self, n_channels: int, window: int):
        import torch
        import torch.nn as nn

        latent = int(self.params["latent_size"])
        filters = int(self.params["conv_channels"])
        k = int(self.params["kernel_size"])

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm_in = nn.LayerNorm(n_channels)
                self.encoder = nn.Sequential(
                    nn.Conv1d(n_channels, filters, k, padding="same"), nn.ReLU(),
                    nn.Conv1d(filters, filters, k, stride=2, padding=k // 2), nn.ReLU(),
                )
                self.reduced = (window + 1) // 2
                flat = filters * self.reduced
                self.to_mu = nn.Linear(flat, latent)
                self.to_logvar = nn.Linear(flat, latent)
                self.from_latent = nn.Linear(latent, flat)
                self.decoder = nn.Sequential(
                    nn.ConvTranspose1d(filters, filters, k, stride=2,
                                       padding=k // 2, output_padding=1), nn.ReLU(),
                    nn.Conv1d(filters, n_channels, k, padding="same"),
                )
                self.filters = filters
                self.window = window

            def forward(self, x):
                x = self.norm_in(x).transpose(1, 2)              # Conv1d wants (B, C, L)
                h = self.encoder(x).flatten(1)
                mu, logvar = self.to_mu(h), self.to_logvar(h).clamp(-10, 10)

                if self.training:
                    # Reparameterisation: sample, but keep the gradient path.
                    z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
                else:
                    # At inference use the mean, so a window always scores the
                    # same. A sampled score would make the same data land on
                    # different sides of the threshold between runs.
                    z = mu

                d = self.from_latent(z).view(-1, self.filters, self.reduced)
                out = self.decoder(d)[:, :, : self.window]
                return out.transpose(1, 2), mu, logvar

        return Net()

    def _loss(self, output, target):
        import torch
        import torch.nn.functional as F

        recon, mu, logvar = output
        reconstruction = F.mse_loss(recon, target)
        divergence = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return reconstruction + float(self.params["kl_weight"]) * divergence

    def _score(self, X: np.ndarray) -> np.ndarray:
        base = super()._score(X)
        if not self.params["score_includes_kl"]:
            return base
        return base + float(self.params["kl_weight"]) * self._latent_divergence(X)

    def _latent_divergence(self, X: np.ndarray) -> np.ndarray:
        """Per-window KL from the prior — how unusual the latent code itself is.

        A window can reconstruct well while sitting somewhere the model has
        never placed normal data. Including this catches that; it is off by
        default because it also makes the score harder to explain.
        """
        import torch

        out = []
        batch = max(1, int(self.params["batch_size"]) * 8)
        with torch.no_grad():
            xs = torch.from_numpy(np.ascontiguousarray(X, dtype="float32"))
            for i in range(0, len(xs), batch):
                _, mu, logvar = self._net(xs[i : i + batch])
                kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                out.append(kl.numpy())
        return np.concatenate(out)
