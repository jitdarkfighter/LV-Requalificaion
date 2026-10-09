"""Shared PyTorch plumbing: training loop, early stopping, determinism.

Every deep model here — forecasting, reconstruction, the autoencoder behind the
clustering method — needs the same things: a reproducible training loop, a way
to stop before it starts memorising, and bounded memory. Writing that once means
each model file contains only its architecture, which is the part worth reading.

Choices worth stating
---------------------
**Layer normalisation, not batch normalisation.** Batch norm normalises across
a mini-batch, so a window's representation depends on which other windows
happened to be batched with it. Layer norm normalises within each sample, which
keeps behaviour consistent across sequence lengths and makes inference on a
single window identical to inference in a batch. The reference study makes the
same choice for the same reason.

**Early stopping on a held-out split.** Training loss falls indefinitely; what
matters is when the model stops generalising. A slice of the training windows is
held back for that judgement alone, and never contributes a gradient.

**Determinism by default.** Seeds are set per model instance. Two runs with the
same configuration produce the same scores, which is what lets the benchmark be
compared across code changes at all.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from avionics_ad.models.base import AnomalyModel, ParamSpec


@dataclass
class TrainHistory:
    """What happened during training. Surfaced in the UI and the results store."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = 0
    stopped_early: bool = False

    @property
    def epochs_run(self) -> int:
        return len(self.train_loss)

    @property
    def best_val_loss(self) -> float:
        return min(self.val_loss) if self.val_loss else float("nan")


#: Parameters every deep model shares. Individual models extend this.
COMMON_PARAMS: tuple[ParamSpec, ...] = (
    ParamSpec("epochs", "int", 50, "Maximum training epochs", 1, 500),
    ParamSpec("batch_size", "int", 32, "Mini-batch size", 1, 1024),
    ParamSpec("learning_rate", "float", 1e-3, "Adam learning rate", 1e-5, 1.0),
    ParamSpec("hidden_size", "int", 64, "Recurrent/latent width", 4, 512),
    ParamSpec("patience", "int", 8, "Epochs without improvement before stopping", 1, 100),
    ParamSpec("validation_fraction", "float", 0.15, "Training windows held back for early stopping", 0.0, 0.5),
    ParamSpec("seed", "int", 0, "Random seed", 0, 2**31 - 1),
)


class TorchModel(AnomalyModel):
    """Base for deep models. Subclasses supply an architecture and a loss target."""

    params_spec = COMMON_PARAMS

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.history = TrainHistory()
        self._net = None
        self._n_channels: int | None = None

    # -- architecture hooks -------------------------------------------------

    @abc.abstractmethod
    def _build_net(self, n_channels: int, window: int):
        """Construct the network for this input shape."""

    @abc.abstractmethod
    def _prepare(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Split windows into (network input, target).

        Forecasters predict the next step from the preceding ones;
        reconstructors predict the window back from itself. That one method is
        the whole difference between the two families.
        """

    def _loss(self, output, target):
        """Training objective. Mean squared error unless a model needs more.

        A variational autoencoder does: its objective adds a divergence term to
        the reconstruction error, and training it on reconstruction alone
        collapses the latent distribution and makes it an ordinary autoencoder.
        """
        import torch.nn.functional as F

        return F.mse_loss(self._to_prediction(output), target)

    def _to_prediction(self, output):
        """Extract the reconstruction/forecast from whatever the net returned.

        Most networks return it directly. A VAE returns it alongside the latent
        parameters its loss needs, so this unwraps the tuple.
        """
        return output[0] if isinstance(output, tuple) else output

    # -- training -----------------------------------------------------------

    def _torch(self):
        import torch

        return torch

    def _seed_everything(self) -> None:
        torch = self._torch()
        seed = int(self.params["seed"])
        torch.manual_seed(seed)
        np.random.seed(seed)

    def _fit(self, X: np.ndarray) -> None:
        torch = self._torch()
        import torch.nn as nn

        self._seed_everything()
        torch.set_num_threads(max(1, min(8, (torch.get_num_threads() or 4))))

        inputs, targets = self._prepare(X)
        self._n_channels = X.shape[2]
        self._net = self._build_net(X.shape[2], X.shape[1])

        xs = torch.from_numpy(np.ascontiguousarray(inputs, dtype="float32"))
        ys = torch.from_numpy(np.ascontiguousarray(targets, dtype="float32"))

        n_val = int(len(xs) * float(self.params["validation_fraction"]))
        if n_val > 0 and len(xs) - n_val >= 1:
            # Held-out windows are taken from the END of the series. Normal
            # operation drifts, so a random split would let the model validate
            # against conditions it already trained on.
            x_tr, y_tr, x_va, y_va = xs[:-n_val], ys[:-n_val], xs[-n_val:], ys[-n_val:]
        else:
            x_tr, y_tr, x_va, y_va = xs, ys, xs, ys

        optimiser = torch.optim.Adam(self._net.parameters(), lr=float(self.params["learning_rate"]))
        batch = max(1, int(self.params["batch_size"]))
        patience = int(self.params["patience"])

        best_state, best_loss, since_improved = None, float("inf"), 0
        self.history = TrainHistory()

        for epoch in range(int(self.params["epochs"])):
            self._net.train()
            order = torch.randperm(len(x_tr))
            running, seen = 0.0, 0
            for i in range(0, len(order), batch):
                idx = order[i : i + batch]
                optimiser.zero_grad()
                loss = self._loss(self._net(x_tr[idx]), y_tr[idx])
                loss.backward()
                optimiser.step()
                running += float(loss.detach()) * len(idx)
                seen += len(idx)
            train_loss = running / max(seen, 1)

            self._net.eval()
            with torch.no_grad():
                val_loss = float(self._loss(self._net(x_va), y_va))

            self.history.train_loss.append(train_loss)
            self.history.val_loss.append(val_loss)

            if val_loss < best_loss - 1e-9:
                best_loss, since_improved = val_loss, 0
                best_state = {k: v.clone() for k, v in self._net.state_dict().items()}
                self.history.best_epoch = epoch
            else:
                since_improved += 1
                if since_improved >= patience:
                    self.history.stopped_early = True
                    break

        if best_state is not None:
            self._net.load_state_dict(best_state)
        self._net.eval()

    # -- scoring ------------------------------------------------------------

    def _predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (target, prediction) for the supplied windows."""
        torch = self._torch()
        inputs, targets = self._prepare(X)
        out = []
        batch = max(1, int(self.params["batch_size"]) * 8)
        with torch.no_grad():
            xs = torch.from_numpy(np.ascontiguousarray(inputs, dtype="float32"))
            for i in range(0, len(xs), batch):
                out.append(self._to_prediction(self._net(xs[i : i + batch])).numpy())
        return targets, np.concatenate(out) if out else np.empty_like(targets)

    def _score(self, X: np.ndarray) -> np.ndarray:
        target, prediction = self._predict(X)
        err = (target - prediction) ** 2
        return err.reshape(len(X), -1).mean(axis=1)

    def score_features(self, X: np.ndarray) -> np.ndarray | None:
        """Per-channel absolute error — the basis for root-cause attribution."""
        if not self.is_fitted:
            raise RuntimeError(f"{self.name}: call fit() before score_features()")
        X = self._check(X, "score_features")
        target, prediction = self._predict(X)
        err = np.abs(target - prediction)
        if err.ndim == 3:          # (B, horizon, P)
            return err.mean(axis=1)
        return err                 # (B, P)

    @property
    def n_parameters(self) -> int:
        if self._net is None:
            return 0
        return sum(p.numel() for p in self._net.parameters())
