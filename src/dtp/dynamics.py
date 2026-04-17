"""Double-integrator dynamics for a 2D drone.

State: x = [p_x, p_y, v_x, v_y]^T in R^4
Input: u = [a_x, a_y]^T in R^2 (acceleration)

Discretized with zero-order hold over sampling period dt:
    x_{k+1} = A x_k + B u_k
"""

from __future__ import annotations

import numpy as np


def double_integrator_2d(dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the discrete-time (A, B) matrices for a 2D double integrator.

    Parameters
    ----------
    dt : float
        Sampling period in seconds.

    Returns
    -------
    A : ndarray, shape (4, 4)
    B : ndarray, shape (4, 2)
    """
    A = np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    B = np.array(
        [
            [0.5 * dt * dt, 0.0],
            [0.0, 0.5 * dt * dt],
            [dt, 0.0],
            [0.0, dt],
        ]
    )
    return A, B
