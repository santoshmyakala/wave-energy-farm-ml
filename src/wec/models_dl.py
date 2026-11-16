"""Deep-learning models (PyTorch).

* ``mlp``: fully-connected network on canonically ordered coordinates.
* ``set_transformer``: permutation-equivariant self-attention over the 16 buoys with a learned
  pairwise attention bias from relative position (dx, dy). It predicts each buoy's power and
  sums them, so it is trained on the per-buoy labels as well as the farm total.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from wec.features import canonical_coordinates

COORD_SCALE = 566.0  # maximum coordinate in the published data (metres)


class MLP(nn.Module):
    def __init__(self, hidden: int = 512, depth: int = 4, dropout: float = 0.05):
        super().__init__()
        layers, width = [], 32
        for _ in range(depth):
            layers += [nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout)]
            width = hidden
        self.net = nn.Sequential(*layers, nn.Linear(width, 1))

    def forward(self, coords: torch.Tensor) -> tuple[torch.Tensor, None]:
        return self.net(coords.flatten(1)).squeeze(-1), None


class RelativeBiasAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads, self.scale = heads, (dim // heads) ** -0.5
        self.qkv, self.out = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.bias = nn.Sequential(nn.Linear(3, 64), nn.GELU(), nn.Linear(64, heads))

    def forward(self, h: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
        b, n, d = h.shape
        q, k, v = self.qkv(h).view(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-1, -2)) * self.scale + self.bias(rel).permute(0, 3, 1, 2)
        return self.out((attn.softmax(-1) @ v).transpose(1, 2).reshape(b, n, d))


class SetTransformer(nn.Module):
    def __init__(self, dim: int = 128, heads: int = 4, layers: int = 3):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.attn = nn.ModuleList(RelativeBiasAttention(dim, heads) for _ in range(layers))
        self.ff = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
            for _ in range(layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(layers))
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def forward(self, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        delta = coords[:, :, None, :] - coords[:, None, :, :]
        rel = torch.cat([delta, delta.norm(dim=-1, keepdim=True)], dim=-1)
        h = self.embed(coords)
        for attn, ff, norm in zip(self.attn, self.ff, self.norms, strict=True):
            h = h + attn(norm(h), rel)
            h = h + ff(h)
        per_buoy = self.head(h).squeeze(-1)
        return per_buoy.sum(-1), per_buoy


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    seed: int = 42
    threads: int = 4


class TorchRegressor:
    """Fit/predict wrapper with target scaling, early stopping and per-buoy auxiliary loss."""

    def __init__(self, kind: str, config: TrainConfig | None = None):
        self.kind, self.cfg = kind, config or TrainConfig()
        self.history: list[dict[str, float]] = []

    def _inputs(self, coords: np.ndarray) -> torch.Tensor:
        if self.kind == "mlp":
            x = canonical_coordinates(coords).reshape(-1, 16, 2)
        else:
            x = coords
        return torch.tensor(x / COORD_SCALE, dtype=torch.float32)

    def fit(self, coords, total, buoys, val_coords, val_total) -> TorchRegressor:
        torch.manual_seed(self.cfg.seed)
        torch.set_num_threads(self.cfg.threads)
        self.mu, self.sigma = float(total.mean()), float(total.std())
        self.buoy_mu = self.mu / 16
        self.model = MLP() if self.kind == "mlp" else SetTransformer()
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=self.cfg.lr, total_steps=self.cfg.epochs * int(np.ceil(len(total) / self.cfg.batch_size))
        )
        x = self._inputs(coords)
        y = torch.tensor((total - self.mu) / self.sigma, dtype=torch.float32)
        yb = torch.tensor((buoys - self.buoy_mu) / self.sigma, dtype=torch.float32)
        best, best_state, stale = np.inf, None, 0
        gen = torch.Generator().manual_seed(self.cfg.seed)
        for epoch in range(self.cfg.epochs):
            self.model.train()
            start = time.perf_counter()
            for batch in torch.randperm(len(y), generator=gen).split(self.cfg.batch_size):
                pred_total, pred_buoys = self._forward(x[batch])
                loss = nn.functional.mse_loss(pred_total, y[batch])
                if pred_buoys is not None:  # auxiliary per-buoy supervision
                    loss = loss + nn.functional.mse_loss(pred_buoys, yb[batch]) * 16
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
            val_rmse = float(np.sqrt(np.mean((self.predict(val_coords) - val_total) ** 2)))
            self.history.append({"epoch": epoch, "val_rmse": val_rmse, "seconds": time.perf_counter() - start})
            if val_rmse < best:
                best, best_state, stale = val_rmse, copy.deepcopy(self.model.state_dict()), 0
            else:
                stale += 1
                if stale >= self.cfg.patience:
                    break
        self.model.load_state_dict(best_state)
        return self

    def _forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        # For the set model each output is a buoy's (power - mean buoy power) / sigma, so their
        # sum is exactly the standardised farm total.
        return self.model(x)

    @torch.no_grad()
    def predict(self, coords: np.ndarray) -> np.ndarray:
        self.model.eval()
        out = torch.cat([self._forward(b)[0] for b in self._inputs(coords).split(4096)])
        return out.numpy() * self.sigma + self.mu
