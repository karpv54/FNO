"""Shared 1D FNO architecture and evaluation for the Burgers equation."""

import torch
from torch import nn
from torch.nn import functional as F
from spectral_layer import SpectralConv1d


class FNOBlock(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.local = nn.Conv1d(width, width, kernel_size=1)
        self.spectral = SpectralConv1d(width, modes)

    def forward(self, x):
        return F.gelu(self.local(x) + self.spectral(x))


class FNO1d(nn.Module):
    def __init__(self, width=16, modes=12, depth=3):
        super().__init__()
        self.lift = nn.Linear(2, width)
        self.blocks = nn.ModuleList([FNOBlock(width, modes) for _ in range(depth)])
        self.project = nn.Sequential(nn.Linear(width, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, u0):
        # u0: (B, N, 1). Rebuild the grid for whatever uniform resolution is given.
        if u0.ndim != 3 or u0.size(-1) != 1:
            raise ValueError('Expected u0 shape (B, N, 1)')
        batch, n, _ = u0.shape
        grid = torch.arange(n, dtype=u0.dtype, device=u0.device) / n
        grid = grid.view(1, n, 1).expand(batch, n, 1)
        features = torch.cat((u0, grid), dim=-1)   # (B, N, 2)
        hidden = self.lift(features).permute(0, 2, 1)  # (B, width, N)
        for block in self.blocks:
            hidden = block(hidden)
        return self.project(hidden.permute(0, 2, 1))  # (B, N, 1)


def relative_l2(prediction, target):
    """One relative error per function; equal grid weights cancel in the ratio."""
    difference = (prediction - target).flatten(1)
    target = target.flatten(1)
    return difference.norm(dim=1) / target.norm(dim=1).clamp_min(1e-8)


@torch.no_grad()
def evaluate(model, u0, target):
    model.eval()
    prediction = model(u0)
    errors = relative_l2(prediction, target)
    return {
        'mse': F.mse_loss(prediction, target).item(),
        'mean_relative_l2': errors.mean().item(),
        'max_relative_l2': errors.max().item(),
        'mean_absolute_mass_error': (prediction.mean(dim=1) - target.mean(dim=1)).abs().mean().item(),
    }
