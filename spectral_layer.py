"""Learn channel mixing at each retained Fourier mode.

Input and output: (B, C, N), real float32 tensors.
Weights: (C_in, C_out, M), complex64. Here C_in = C_out = width.
"""

import torch
from torch import nn


class SpectralConv1d(nn.Module):
    def __init__(self, width: int, modes: int):
        super().__init__()
        if width < 1 or modes < 1:
            raise ValueError('width and modes must be positive')
        self.width = width
        self.modes = modes
        scale = 1 / width

        # Register a learnable complex tensor of shape (width, width, modes).
        self.weights = nn.Parameter(scale * torch.randn(width, width, modes, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(1) != self.width:
            raise ValueError('Expected input shape (B, width, N)')

        # Transform along the spatial axis.
        x_fourier = torch.fft.rfft(x, dim=-1)
        if self.modes > x_fourier.size(-1):
            raise ValueError('modes exceeds the available rFFT coefficients')

        # Allocate the entire one-sided output spectrum.
        output_fourier = torch.zeros_like(x_fourier)

        # Select the retained modes. Shape: (B, width, modes).
        low_modes = x_fourier[..., :self.modes]

        # Sum over input channels, separately for each batch and mode.
        output_fourier[..., :self.modes] = torch.einsum('bik,iok->bok', low_modes, self.weights)

        # Return a real signal with the original number of spatial points.
        return torch.fft.irfft(output_fourier, n=x.size(-1), dim=-1)
