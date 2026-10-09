"""Time-frequency autoencoder with density clustering.

A different question from the other families. Forecasting asks whether the next
value is surprising; reconstruction asks whether a window resembles the training
data. This asks: **does this window fall into any of the modes of normal
operation that were observed at all?**

The pipeline
------------
1. **Time-frequency analysis.** Each channel's window is transformed to a
   spectrogram via a short-time Fourier transform. A level shift and an
   oscillation can share a mean and a variance while looking completely
   different in frequency, so this makes pattern anomalies — changes in
   *character* rather than magnitude — directly visible as energy moving between
   frequency bands.

2. **Convolutional autoencoder.** Trained to reconstruct the spectrogram,
   producing a compact embedding per window. Convolution is the natural choice
   here because a spectrogram is an image, and the features that matter are
   local in both time and frequency.

3. **HDBSCAN.** Clusters the embeddings by density. Why density clustering
   rather than k-means: normal operation on a launch vehicle is genuinely
   multi-modal — idle, powered, actuating are different regimes with different
   signatures — and nobody knows in advance how many modes there are. HDBSCAN
   finds that itself, handles clusters of different shapes and densities, and
   explicitly labels points that belong to no cluster rather than forcing every
   point into one.

4. **Scoring.** Distance from a window's embedding to the nearest cluster
   centre. A window that lands inside a known mode is normal; one that lands in
   empty space is not.

Two details that decide whether this works
------------------------------------------
**The embedding is standardised before clustering.** The autoencoder's latent
has no inherent scale — measured on homogeneous training data it came out with a
per-dimension standard deviation of about 0.001 — and HDBSCAN is a
distance-based method whose density estimates depend on that scale entirely.
Standardising makes the clustering behave the same whatever scale the network
happened to settle on.

**A single mode is a valid outcome.** When normal operation really is one
regime, the embeddings form one blob of near-uniform density, and HDBSCAN's
excess-of-mass criterion rejects that as noise rather than calling it a cluster.
That is HDBSCAN working as designed, not a failure, so the model treats it as
one mode centred on the training data. Abandoning clustering there would throw
away the thing being measured — distance from normal — over a question about how
many modes normal has.

Honest limitation
-----------------
scikit-learn's HDBSCAN does not predict cluster membership for unseen points, so
the fitted cluster centres are retained and new windows scored by distance to
them. That approximates what HDBSCAN does internally and loses the density
information at cluster edges. Only if the embedding is degenerate — every window
mapping to literally the same point — does the model fall back to reconstruction
error, and it records that it did.
"""

from __future__ import annotations

import numpy as np

from avionics_ad.models.base import ParamSpec
from avionics_ad.models.registry import register
from avionics_ad.models.torch_base import COMMON_PARAMS, TorchModel

TFA_PARAMS: tuple[ParamSpec, ...] = COMMON_PARAMS + (
    ParamSpec("latent_size", "int", 16, "Embedding width handed to the clusterer", 2, 256),
    ParamSpec("conv_channels", "int", 16, "Convolution filters", 4, 128),
    ParamSpec("nperseg", "int", 16, "STFT segment length in samples", 4, 256),
    ParamSpec("min_cluster_size", "int", 15, "Smallest group HDBSCAN will call a cluster", 2, 200),
    ParamSpec("min_samples", "int", 5, "How conservative the density estimate is", 1, 100),
    ParamSpec("blend_reconstruction", "float", 0.0,
              "Weight on reconstruction error added to the cluster distance", 0.0, 1.0),
)


def spectrogram(X: np.ndarray, *, nperseg: int) -> np.ndarray:
    """Magnitude spectrogram per channel: ``(B, W, P)`` -> ``(B, P, F, T)``.

    Log-scaled, because spectral energy spans orders of magnitude and a linear
    scale would let the strongest band dominate the convolution entirely.
    """
    from scipy.signal import stft

    X = np.asarray(X, dtype="float64")
    b, w, p = X.shape
    nperseg = int(min(nperseg, w))
    noverlap = nperseg // 2

    _, _, Z = stft(
        X.transpose(0, 2, 1),          # (B, P, W) — stft works on the last axis
        nperseg=nperseg,
        noverlap=noverlap,
        axis=-1,
        padded=True,
        boundary="zeros",
    )
    return np.log1p(np.abs(Z)).astype("float32")


def spectrogram_shape(window: int, n_channels: int, *, nperseg: int) -> tuple[int, int, int]:
    """Shape a window transforms to, without running the transform."""
    probe = spectrogram(np.zeros((1, window, n_channels), dtype="float32"), nperseg=nperseg)
    return probe.shape[1:]


@register
class TFACNNAutoencoderHDBSCAN(TorchModel):
    name = "tfa_cnn_ae_hdbscan"
    label = "TFA CNN-Autoencoder + HDBSCAN"
    family = "cluster"
    params_spec = TFA_PARAMS
    description = """
    Transforms each window to a spectrogram, learns a compact embedding with a
    convolutional autoencoder, then clusters those embeddings by density with
    HDBSCAN. A window is anomalous when it falls far from every mode of normal
    operation that was observed.

    The frequency view is what makes it distinctive: an oscillation and a level
    shift can share a mean and a standard deviation while looking entirely
    different in the spectrum. Density clustering suits the problem because
    normal operation really is multi-modal — idle, powered and actuating are
    different regimes — and HDBSCAN discovers how many modes there are instead
    of being told.
    """

    def __init__(self, **params):
        super().__init__(**params)
        self._centroids: np.ndarray | None = None
        self._embed_mean: np.ndarray | None = None
        self._embed_std: np.ndarray | None = None
        self._distance_scale: float = 1.0
        self.cluster_report: dict[str, float] = {}
        self.used_fallback: bool = False

    # -- the autoencoder ----------------------------------------------------

    def _prepare(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = spectrogram(X, nperseg=int(self.params["nperseg"]))
        return s, s

    def _build_net(self, n_channels: int, window: int):
        import torch
        import torch.nn as nn

        p, f, t = spectrogram_shape(window, n_channels, nperseg=int(self.params["nperseg"]))
        filters = int(self.params["conv_channels"])
        latent = int(self.params["latent_size"])

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(p, filters, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(filters, filters, 3, padding=1), nn.ReLU(),
                )
                flat = filters * f * t
                self.to_latent = nn.Linear(flat, latent)
                self.from_latent = nn.Linear(latent, flat)
                self.decoder = nn.Sequential(
                    nn.Conv2d(filters, filters, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(filters, p, 3, padding=1),
                )
                self.shape = (filters, f, t)

            def embed(self, x):
                return self.to_latent(self.encoder(x).flatten(1))

            def forward(self, x):
                z = self.embed(x)
                d = self.from_latent(z).view(-1, *self.shape)
                return self.decoder(d)

        return Net()

    # -- clustering ---------------------------------------------------------

    def _embed(self, X: np.ndarray, *, standardise: bool = True) -> np.ndarray:
        import torch

        spec, _ = self._prepare(X)
        out = []
        batch = max(1, int(self.params["batch_size"]) * 8)
        with torch.no_grad():
            xs = torch.from_numpy(np.ascontiguousarray(spec, dtype="float32"))
            for i in range(0, len(xs), batch):
                out.append(self._net.embed(xs[i : i + batch]).numpy())
        embeddings = np.concatenate(out).astype("float64")

        if standardise and self._embed_mean is not None:
            return (embeddings - self._embed_mean) / self._embed_std
        return embeddings

    def _fit(self, X: np.ndarray) -> None:
        from sklearn.cluster import HDBSCAN

        super()._fit(X)                       # train the autoencoder first

        raw = self._embed(X, standardise=False)

        # The latent has no inherent scale, and HDBSCAN's density estimate
        # depends on it entirely. Standardise before clustering, and keep the
        # parameters so unseen windows land in the same space.
        self._embed_mean = raw.mean(axis=0)
        self._embed_std = raw.std(axis=0)
        degenerate = self._embed_std < 1e-12
        if degenerate.all():
            self._embed_mean = self._embed_std = None
            self._centroids = None
            self.used_fallback = True
            self.cluster_report = {"n_clusters": 0, "noise_fraction": 1.0,
                                   "n_embeddings": int(len(raw)), "degenerate_embedding": 1.0}
            return
        self._embed_std = np.where(degenerate, 1.0, self._embed_std)
        embeddings = (raw - self._embed_mean) / self._embed_std

        labels = HDBSCAN(
            min_cluster_size=int(self.params["min_cluster_size"]),
            min_samples=int(self.params["min_samples"]),
            copy=True,
        ).fit_predict(embeddings)
        real = sorted({int(c) for c in labels if c >= 0})

        self.used_fallback = False
        if real:
            self._centroids = np.stack([embeddings[labels == c].mean(axis=0) for c in real])
        else:
            # One blob of near-uniform density. HDBSCAN rejects that as noise by
            # design; here it means normal operation has a single mode.
            self._centroids = embeddings.mean(axis=0)[None, :]

        self.cluster_report = {
            "n_clusters": float(len(real)),
            "modes_used": float(len(self._centroids)),
            "noise_fraction": float(np.mean(labels < 0)),
            "n_embeddings": float(len(embeddings)),
            "single_mode": float(not real),
        }

        # Normalise by how far normal data typically sits from its own mode, so
        # scores are comparable across runs and datasets.
        self._distance_scale = float(np.median(self._nearest_distance(embeddings))) or 1.0

    def _nearest_distance(self, embeddings: np.ndarray) -> np.ndarray:
        diff = embeddings[:, None, :] - self._centroids[None, :, :]
        return np.sqrt((diff**2).sum(axis=2)).min(axis=1)

    # -- scoring ------------------------------------------------------------

    def _score(self, X: np.ndarray) -> np.ndarray:
        reconstruction = super()._score(X)
        if self._centroids is None:
            return reconstruction

        distance = self._nearest_distance(self._embed(X)) / self._distance_scale
        blend = float(self.params["blend_reconstruction"])
        if blend <= 0:
            return distance
        return (1 - blend) * distance + blend * reconstruction

    def score_features(self, X: np.ndarray) -> np.ndarray | None:
        """Per-channel error, recovered from the spectrogram reconstruction.

        Each channel occupies its own plane of the spectrogram, so the error can
        still be decomposed per channel — attribution survives the transform,
        which it would not if the channels had been mixed together first.
        """
        if not self.is_fitted:
            raise RuntimeError(f"{self.name}: call fit() before score_features()")
        X = self._check(X, "score_features")
        target, prediction = self._predict(X)
        return np.abs(target - prediction).reshape(len(X), target.shape[1], -1).mean(axis=2)
