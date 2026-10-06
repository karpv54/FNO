"""Forward Euler for periodic viscous Burgers, using NumPy only.

The conservative advection stencil uses a simple global Rusanov flux.
Diffusion uses centered second differences. Time accuracy is first order;
the stabilizing advection stencil is also first order in space.
"""

import math

import numpy as np


def stable_euler_steps(u0, viscosity=0.01, final_time=1.0, safety=0.8):
    """Return a safe number of steps for dx=1/N on the periodic unit interval."""
    values = np.asarray(u0, dtype=np.float64)
    if (values.ndim < 1 or values.shape[-1] < 3 or not np.isfinite(values).all()
            or not np.isfinite(viscosity) or viscosity <= 0
            or not np.isfinite(final_time) or final_time <= 0 or not 0 < safety < 1):
        raise ValueError('Require finite profiles, N >= 3, nu > 0, T > 0, and 0 < safety < 1')
    dx = 1 / values.shape[-1]
    speed = float(np.abs(values).max())
    max_dt = safety / (speed / dx + 2 * viscosity / dx**2)
    return max(1, math.ceil(final_time / max_dt))


def euler_burgers(u0, steps, viscosity=0.01, final_time=1.0):
    """Solve from t=0 to T; u0 has shape (N,) or (...,N), with no repeated endpoint."""
    minimum = stable_euler_steps(u0, viscosity, final_time, safety=0.999)
    if not isinstance(steps, (int, np.integer)) or steps < minimum:
        raise ValueError(f'Use at least {minimum} time steps for this profile and grid')
    u = np.array(u0, dtype=np.float64, copy=True)
    dx, dt = 1 / u.shape[-1], final_time / steps
    # Each profile has its own speed bound, independent of other batch members.
    speed = np.max(np.abs(u), axis=-1, keepdims=True)
    # Rusanov adds speed*dx/2 of numerical diffusion to stabilize advection.
    diffusion = viscosity / dx**2 + speed / (2 * dx)
    for _ in range(steps):
        left = np.roll(u, 1, axis=-1)
        right = np.roll(u, -1, axis=-1)
        advection = -(right**2 - left**2) / (4 * dx)
        u += dt * (advection + diffusion * (left - 2 * u + right))
    if not np.isfinite(u).all():
        raise FloatingPointError('Euler produced non-finite values')
    return u
