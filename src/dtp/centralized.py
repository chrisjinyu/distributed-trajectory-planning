"""Centralized joint QP reference solver.

Owner (long term): Yuyang Wu.
Owner (temporary stub): Christian Yu.

Solves the full coupled MPC problem over all N drones simultaneously using
the SAME linearized collision constraints as the distributed methods.
Provides the ground-truth optimum to benchmark Consensus ADMM and the
penalty-method baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from dtp.dynamics import double_integrator_2d
from dtp.scenario import Scenario


@dataclass
class CentralizedSolution:
    x: np.ndarray  # shape (N, H+1, 4)
    u: np.ndarray  # shape (N, H, 2)
    objective: float
    status: str


def solve_centralized(
    scenario: Scenario,
    n_ij: dict[tuple[int, int], np.ndarray] | None = None,
    solver: str = cp.OSQP,
) -> CentralizedSolution:
    """Solve the joint multi-agent QP.

    Parameters
    ----------
    scenario : Scenario
        Shared problem configuration.
    n_ij : dict, optional
        Collision normals. If None, collision constraints are omitted.
        Otherwise n_ij[(i, j)] is a (H+1, 2) array of unit vectors giving
        the separating-hyperplane normal for the ordered pair (i, j). We
        only need one normal per undirected edge because the constraints
        n^T (p_i - p_j) >= d and -n^T (p_i - p_j) >= d differ by sign.
        Convention: the key (i, j) with i < j is the canonical one.
    solver : str
        CVXPY solver identifier (OSQP, CLARABEL, SCS, etc.).

    Returns
    -------
    CentralizedSolution
    """
    N = scenario.N
    H = scenario.H
    A, B = double_integrator_2d(scenario.dt)

    x = [cp.Variable((H + 1, 4), name=f"x_{i}") for i in range(N)]
    u = [cp.Variable((H, 2), name=f"u_{i}") for i in range(N)]

    cost = 0
    constraints: list = []

    for i in range(N):
        x_ref = scenario.x_ref[i]
        cost = cost + sum(
            cp.quad_form(x[i][k] - x_ref[k], cp.psd_wrap(scenario.Q)) for k in range(H + 1)
        )
        cost = cost + sum(cp.quad_form(u[i][k], cp.psd_wrap(scenario.R)) for k in range(H))

        constraints.append(x[i][0] == scenario.x0[i])
        for k in range(H):
            constraints.append(x[i][k + 1] == A @ x[i][k] + B @ u[i][k])
            constraints.append(cp.norm_inf(u[i][k]) <= scenario.u_max)

    if n_ij is not None:
        for (i, j), n in n_ij.items():
            p_i = x[i][:, 0:2]
            p_j = x[j][:, 0:2]
            inner = cp.sum(cp.multiply(n, p_i - p_j), axis=1)
            constraints.append(inner >= scenario.d_min)

    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve(solver=solver)

    x_val = np.stack([xi.value for xi in x], axis=0)
    u_val = np.stack([ui.value for ui in u], axis=0)

    return CentralizedSolution(
        x=x_val, u=u_val, objective=float(problem.value), status=problem.status
    )
