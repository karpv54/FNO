"""Reference data for u_t + u*u_x = nu*u_xx on the periodic unit interval.

Uses the Cole-Hopf transformation with a Galilean shift for nonzero means.
The formula is continuous; its FFT evaluation has spatial discretization error.
This code generates labels only. It is never called inside the FNO forward pass.
"""

import numpy as np


def sample_coefficients(count, seed, harmonics=8):
    rng = np.random.default_rng(seed)
    k = np.arange(1, harmonics + 1, dtype=np.float64)
    coefficients = rng.normal(size=(count, 2, harmonics)) / k**2
    amplitude = rng.uniform(0.2, 1.0, size=(count, 1, 1))
    coefficients *= amplitude / np.abs(coefficients).sum(axis=(1, 2), keepdims=True)
    mean = rng.uniform(-0.3, 0.3, size=(count, 1))
    return coefficients, mean


def initial_profile(coefficients, mean, n):
    k = np.arange(1, coefficients.shape[-1] + 1, dtype=np.float64)
    angles = 2 * np.pi * k[:, None] * np.arange(n)[None, :] / n
    return mean + coefficients[:, 0] @ np.sin(angles) + coefficients[:, 1] @ np.cos(angles)


def solve_burgers(coefficients, mean, n=512, viscosity=0.01, final_time=1.0):
    """Return float64 solutions for finite Fourier-series initial conditions.

    Let m = mean(u0), v0 = u0-m, and P_x = v0.
    phi0 = exp(-P/(2*nu)); phi_t = nu*phi_xx.
    Then u(x,t) = m - 2*nu * phi_x(x-m*t,t) / phi(x-m*t,t).
    The Galilean shift is essential when m != 0.
    """
    if viscosity <= 0 or final_time < 0 or n <= 2 * coefficients.shape[-1]:
        raise ValueError('Require nu > 0, T >= 0, and a grid resolving all input harmonics')
    if final_time == 0:
        return initial_profile(coefficients, mean, n)
    k = np.arange(1, coefficients.shape[-1] + 1, dtype=np.float64)
    omega = 2 * np.pi * k
    angles = omega[:, None] * np.arange(n)[None, :] / n
    primitive = -(coefficients[:, 0] / omega) @ np.cos(angles)
    primitive += (coefficients[:, 1] / omega) @ np.sin(angles)
    exponent = -primitive / (2 * viscosity)
    # Multiplying phi by a positive constant does not change phi_x/phi.
    exponent -= exponent.max(axis=-1, keepdims=True)
    phi_initial = np.exp(exponent)
    frequencies = 2 * np.pi * np.fft.rfftfreq(n, d=1 / n)
    evolution = np.exp(-viscosity * frequencies**2 * final_time)
    shift = np.exp(-1j * mean * frequencies[None, :] * final_time)
    phi_hat = np.fft.rfft(phi_initial, axis=-1) * evolution * shift
    phi = np.fft.irfft(phi_hat, n=n, axis=-1)
    phi_x = np.fft.irfft(1j * frequencies * phi_hat, n=n, axis=-1)
    if not np.isfinite(phi).all() or np.any(phi <= 1e-14 * phi.max(axis=-1, keepdims=True)):
        raise FloatingPointError('Cole-Hopf reconstruction is poorly conditioned for these settings')
    return mean - 2 * viscosity * phi_x / phi


def sample_burgers_pairs(count, n, seed, solver_n=512):
    """Fixed nu=0.01, T=1. Return (B,N,1) float32 training arrays."""
    if solver_n % n != 0:
        raise ValueError('The reference grid must be a multiple of the model grid')
    coefficients, mean = sample_coefficients(count, seed)
    u0 = initial_profile(coefficients, mean, n)
    target_fine = solve_burgers(coefficients, mean, n=solver_n)
    target = target_fine[:, ::solver_n // n]
    return u0[..., None].astype(np.float32), target[..., None].astype(np.float32)
