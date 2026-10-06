"""An affine baseline with exactly the same trainable weights as FNO1d.

Only the GELU activations are removed. For a fixed spatial grid, the result
is an affine map of the initial profile. It retains the Fourier layers.
"""

from torch import nn

from fno_1d_model import FNO1d


class LinearBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.local = block.local
        self.spectral = block.spectral

    def forward(self, x):
        return self.local(x) + self.spectral(x)


class LinearFNO1d(FNO1d):
    def __init__(self, width=16, modes=12, depth=3):
        super().__init__(width=width, modes=modes, depth=depth)
        self.blocks = nn.ModuleList([LinearBlock(block) for block in self.blocks])
        self.project[1] = nn.Identity()


def parameter_count(model):
    """Count real scalar weights; a complex weight contributes two scalars."""
    return sum(p.numel() * (2 if p.is_complex() else 1)
               for p in model.parameters() if p.requires_grad)
