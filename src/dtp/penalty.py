"""Distributed penalty-method baseline for multi-agent trajectory planning.

Owner: Yen-Ru Chin.

Formulation
-----------
Each agent i solves its own MPC QP with a quadratic penalty on collision
constraint violations. Unlike Consensus ADMM, there are no consensus copies
of neighbor trajectories and no dual variables: agents simply broadcast
their own trajectories and treat the latest broadcast from each neighbor
as fixed external data.

The nonconvex collision constraint ||p_i - p_j|| >= d_min is linearized
around the current iterate as a half-space with unit normal
n_ij[k] = (p_i[k] - p_j[k]) / ||...||. A slack s_ij >= 0 captures the
violation and is penalized quadratically:

    min  tracking + input cost + w * sum_{j in N_i} ||s_ij||^2
    s.t. dynamics, input bounds, and
         n_ij^T p_i[k] + s_ij[k] >= d_min + n_ij^T p_j_hat[k]

Small w trades feasibility for conditioning (agents may collide); large w
trades conditioning for feasibility (QPs become ill-scaled).

Per-iteration flow
------------------
    1. Each agent solves its local QP with current n_ij and p_j_hat fixed.
    2. Agents broadcast their own theta_i to neighbors.
    3. Collision normals and RHS Parameters are re-linearized from the
       latest positions.
    4. Primal residual ||theta - theta_prev|| check for convergence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np

from dtp.admm import _straight_line_warm_start
from dtp.mpc import PenaltyLocalQP, build_penalty_qp
from dtp.scenario import Scenario


@dataclass
class PenaltyHistory:
    """Per-iteration diagnostic log."""

    primal_residual: list[float] = field(default_factory=list)
    objective: list[float] = field(default_factory=list)
    constraint_violation: list[float] = field(default_factory=list)
    max_slack: list[float] = field(default_factory=list)


@dataclass
class PenaltyResult:
    x: np.ndarray  # (N, H+1, 4) each agent's own trajectory
    u: np.ndarray  # (N, H, 2)   each agent's controls
    iterations: int
    converged: bool
    history: PenaltyHistory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_objective(scenario: Scenario, x: np.ndarray, u: np.ndarray) -> float:
    Q, R = scenario.Q, scenario.R
    total = 0.0
    for i in range(scenario.N):
        dx = x[i] - scenario.x_ref[i]
        total += float(np.einsum("ka,ab,kb->", dx, Q, dx))
        total += float(np.einsum("ka,ab,kb->", u[i], R, u[i]))
    return total


def _compute_worst_violation(
    scenario: Scenario, x: np.ndarray, edge_normals: dict[tuple[int, int], np.ndarray]
) -> float:
    worst = 0.0
    for (i, j), n in edge_normals.items():
        p_i = x[i, :, 0:2]
        p_j = x[j, :, 0:2]
        inner = np.sum(n * (p_i - p_j), axis=1)
        violation = scenario.d_min - inner
        worst = max(worst, float(np.max(violation)))
    return worst


def _update_collision_data(
    scenario: Scenario, qps: dict[int, PenaltyLocalQP], theta_own: np.ndarray
) -> dict[tuple[int, int], np.ndarray]:
    """Re-linearize normals and refresh collision_rhs Parameters for all QPs."""
    edge_normals: dict[tuple[int, int], np.ndarray] = {}
    for i, j in scenario.graph.edges():
        a, b = (i, j) if i < j else (j, i)
        p_a = theta_own[a, :, 0:2]
        p_b = theta_own[b, :, 0:2]
        diff = p_a - p_b
        norms = np.linalg.norm(diff, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-6)
        n_ab = diff / norms  # (H+1, 2)
        edge_normals[(a, b)] = n_ab

        # Agent a sees neighbor b: normal points from a -> b, use p_b as p_j_hat.
        if b in qps[a].n_ij:
            qps[a].n_ij[b].value = n_ab
            qps[a].collision_rhs[b].value = scenario.d_min + np.sum(n_ab * p_b, axis=1)
        # Agent b sees neighbor a: normal is negated, use p_a as p_j_hat.
        if a in qps[b].n_ij:
            qps[b].n_ij[a].value = -n_ab
            qps[b].collision_rhs[a].value = scenario.d_min + np.sum(-n_ab * p_a, axis=1)
    return edge_normals


def _build_qps(
    scenario: Scenario,
    collision_penalty_weight: float,
    include_collision: bool,
) -> dict[int, PenaltyLocalQP]:
    return {
        i: build_penalty_qp(
            scenario,
            i,
            collision_penalty_weight=collision_penalty_weight,
            include_collision=include_collision,
        )
        for i in range(scenario.N)
    }


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def solve_penalty_method(
    scenario: Scenario,
    w: float = 1e3,
    max_iter: int = 100,
    eps_abs: float = 1e-4,
    eps_rel: float = 1e-3,
    solver: str = cp.CLARABEL,
    solver_opts: dict | None = None,
    initial_guess: np.ndarray | None = None,
    include_collision: bool = True,
    verbose: bool = False,
) -> PenaltyResult:
    """Run the distributed penalty-method baseline.

    Parameters
    ----------
    scenario : Scenario
    w : float
        Quadratic penalty weight on collision slack. This is the knob the
        proposal highlights: low w admits collisions, high w ill-conditions
        the QPs.
    max_iter : int
    eps_abs, eps_rel : float
        Boyd-style primal-residual tolerance on the change in own
        trajectories between iterations.
    solver : str
        CVXPY solver identifier.
    initial_guess : ndarray, optional
        Shape (N, H+1, 4). If None, a straight-line warm-start is used.
    include_collision : bool
    verbose : bool

    Returns
    -------
    PenaltyResult
    """
    if solver_opts is None:
        if solver == cp.OSQP:
            solver_opts = {
                "max_iter": 50000,
                "eps_abs": 1e-7,
                "eps_rel": 1e-7,
                "polish": True,
            }
        else:
            solver_opts = {}

    N = scenario.N
    H = scenario.H

    qps = _build_qps(scenario, w, include_collision)

    if initial_guess is None:
        theta_own = _straight_line_warm_start(scenario)
    else:
        theta_own = initial_guess.copy()

    u_store = np.zeros((N, H, 2))

    # Initial collision linearization from the warm-start.
    edge_normals = _update_collision_data(scenario, qps, theta_own)

    history = PenaltyHistory()
    converged = False
    final_iter = 0

    for k in range(max_iter):
        theta_own_prev = theta_own.copy()

        # --- Step 1: primal update (parallel in principle) ---
        for i in range(N):
            try:
                qps[i].problem.solve(solver=solver, warm_start=True, **solver_opts)
            except cp.error.SolverError as e:
                raise RuntimeError(
                    f"Agent {i} subproblem solver failed at iteration {k}: {e}"
                ) from e
            status = qps[i].problem.status
            if status not in ("optimal", "optimal_inaccurate"):
                raise RuntimeError(
                    f"Agent {i} subproblem returned status '{status}' at iteration {k}."
                )
            theta_own[i] = qps[i].theta.value
            u_store[i] = qps[i].u.value

        # --- Step 2: broadcast via re-linearization + RHS refresh ---
        if include_collision:
            edge_normals = _update_collision_data(scenario, qps, theta_own)

        # --- Step 3: diagnostics ---
        primal_res = float(np.linalg.norm(theta_own - theta_own_prev))
        obj = _compute_objective(scenario, theta_own, u_store)
        viol = (
            _compute_worst_violation(scenario, theta_own, edge_normals)
            if include_collision
            else 0.0
        )
        max_slack = 0.0
        if include_collision:
            for i in range(N):
                for s in qps[i].slack.values():
                    if s.value is not None:
                        max_slack = max(max_slack, float(np.max(s.value)))

        history.primal_residual.append(primal_res)
        history.objective.append(obj)
        history.constraint_violation.append(viol)
        history.max_slack.append(max_slack)

        if verbose:
            print(
                f"iter {k:3d} | r={primal_res:.3e} | J={obj:.4e} "
                f"| viol={viol:.3e} | slack={max_slack:.3e}"
            )

        # --- Step 4: convergence check ---
        n_elem = N * (H + 1) * 4
        traj_norm = float(np.linalg.norm(theta_own))
        eps_pri = np.sqrt(n_elem) * eps_abs + eps_rel * traj_norm
        final_iter = k + 1
        if primal_res < eps_pri:
            converged = True
            break

    return PenaltyResult(
        x=theta_own,
        u=u_store,
        iterations=final_iter,
        converged=converged,
        history=history,
    )
