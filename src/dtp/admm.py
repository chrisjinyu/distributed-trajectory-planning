"""Consensus ADMM driver for distributed multi-agent trajectory planning.

Owner: Christian Yu

Formulation
-----------
Each agent i maintains its own trajectory theta_i plus, for each neighbor
j in N_i, a local belief tilde_theta_i^(j). The consensus constraint
tilde_theta_i^(j) = theta_j is enforced via scaled dual variables u_dual_ij
that are updated each iteration.

DPP-compliance
--------------
The per-agent subproblem is built with rho as a *constant* (not a CVXPY
Parameter) so the problem stays DPP and CVXPY can cache canonicalization.
When rho changes (adaptive residual balancing), the QPs are rebuilt.

Per-iteration flow:
    1. Each agent solves its local QP.
    2. Agents broadcast their own theta_i to neighbors.
    3. Dual update per directed edge: u_dual_ij += tilde_theta_i^(j) - theta_j.
    4. Re-linearize collision normals from own-theta positions.
    5. Residuals + stopping check.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np

from dtp.mpc import LocalQP, build_local_qp
from dtp.scenario import Scenario


@dataclass
class ADMMHistory:
    """Per-iteration diagnostic log for convergence analysis."""

    primal_residual: list[float] = field(default_factory=list)
    dual_residual: list[float] = field(default_factory=list)
    objective: list[float] = field(default_factory=list)
    constraint_violation: list[float] = field(default_factory=list)
    rho: list[float] = field(default_factory=list)
    max_slack: list[float] = field(default_factory=list)


@dataclass
class ADMMResult:
    x: np.ndarray  # (N, H+1, 4) each agent's own trajectory
    u: np.ndarray  # (N, H, 2)   each agent's controls
    iterations: int
    converged: bool
    history: ADMMHistory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_local_objective(scenario: Scenario, x: np.ndarray, u: np.ndarray) -> float:
    Q, R = scenario.Q, scenario.R
    total = 0.0
    for i in range(scenario.N):
        dx = x[i] - scenario.x_ref[i]
        total += float(np.einsum("ka,ab,kb->", dx, Q, dx))
        total += float(np.einsum("ka,ab,kb->", u[i], R, u[i]))
    return total


def _compute_constraint_violation(
    scenario: Scenario, x: np.ndarray, n_ij: dict[tuple[int, int], np.ndarray]
) -> float:
    worst = 0.0
    for (i, j), n in n_ij.items():
        p_i = x[i, :, 0:2]
        p_j = x[j, :, 0:2]
        inner = np.sum(n * (p_i - p_j), axis=1)
        violation = scenario.d_min - inner
        worst = max(worst, float(np.max(violation)))
    return worst


def _update_collision_normals(
    scenario: Scenario, qps: dict[int, LocalQP], theta_own: np.ndarray
) -> dict[tuple[int, int], np.ndarray]:
    edge_normals: dict[tuple[int, int], np.ndarray] = {}
    for i, j in scenario.graph.edges():
        a, b = (i, j) if i < j else (j, i)
        p_a = theta_own[a, :, 0:2]
        p_b = theta_own[b, :, 0:2]
        diff = p_a - p_b
        norms = np.linalg.norm(diff, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-6)
        n_ab = diff / norms
        edge_normals[(a, b)] = n_ab
        if b in qps[a].n_ij:
            qps[a].n_ij[b].value = n_ab
        if a in qps[b].n_ij:
            qps[b].n_ij[a].value = -n_ab
    return edge_normals


def _straight_line_warm_start(scenario: Scenario) -> np.ndarray:
    N, H = scenario.N, scenario.H
    lines = np.zeros((N, H + 1, 4))
    alphas = np.linspace(0.0, 1.0, H + 1)
    for k in range(N):
        x_start = scenario.x0[k]
        x_end = scenario.x_ref[k, -1]
        offset = np.zeros(4)
        offset[0] = scenario.d_min * 0.25 * np.cos(2 * np.pi * k / max(N, 1))
        offset[1] = scenario.d_min * 0.25 * np.sin(2 * np.pi * k / max(N, 1))
        for kk, a in enumerate(alphas):
            bump = 4 * a * (1 - a) * offset
            lines[k, kk] = (1 - a) * x_start + a * x_end + bump
    return lines


def _capture_state(qps: dict[int, LocalQP], scenario: Scenario) -> dict:
    """Snapshot all Parameter values so they can be restored after rebuild."""
    state = {}
    for i in range(scenario.N):
        state[i] = {
            "x0": qps[i].x0.value.copy(),
            "theta_j_hat": {j: qps[i].theta_j_hat[j].value.copy() for j in scenario.neighbors(i)},
            "u_dual": {j: qps[i].u_dual[j].value.copy() for j in scenario.neighbors(i)},
            "n_ij": {j: qps[i].n_ij[j].value.copy() for j in scenario.neighbors(i)},
        }
    return state


def _restore_state(
    qps: dict[int, LocalQP], state: dict, scenario: Scenario, dual_scale: float = 1.0
) -> None:
    for i in range(scenario.N):
        qps[i].x0.value = state[i]["x0"]
        for j in scenario.neighbors(i):
            qps[i].theta_j_hat[j].value = state[i]["theta_j_hat"][j]
            qps[i].u_dual[j].value = state[i]["u_dual"][j] * dual_scale
            qps[i].n_ij[j].value = state[i]["n_ij"][j]


def _build_qps(
    scenario: Scenario,
    rho: float,
    include_collision: bool,
    collision_slack_weight: float,
) -> dict[int, LocalQP]:
    return {
        i: build_local_qp(
            scenario,
            i,
            rho=rho,
            include_collision=include_collision,
            collision_slack_weight=collision_slack_weight,
        )
        for i in range(scenario.N)
    }


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def solve_consensus_admm(
    scenario: Scenario,
    rho: float = 1.0,
    max_iter: int = 200,
    eps_abs: float = 1e-4,
    eps_rel: float = 1e-3,
    solver: str = cp.CLARABEL,
    solver_opts: dict | None = None,
    initial_guess: np.ndarray | None = None,
    include_collision: bool = True,
    collision_slack_weight: float = 1e4,
    adaptive_rho: bool = False,
    tau_incr: float = 2.0,
    tau_decr: float = 2.0,
    mu: float = 10.0,
    verbose: bool = False,
) -> ADMMResult:
    """Run distributed Consensus ADMM (DPP-compliant formulation).

    See module docstring for the algorithm. Key user-facing parameters:
      rho                      : initial ADMM penalty.
      adaptive_rho             : if True, applies residual balancing (rebuilds QPs).
      collision_slack_weight   : penalty weight on linearized-collision slack.

    Returns an ADMMResult with the converged per-agent own trajectory.
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

    # --- Build per-agent QPs with the initial rho baked in ---
    qps = _build_qps(scenario, rho, include_collision, collision_slack_weight)

    # --- Own-trajectory warm-start ---
    if initial_guess is None:
        theta_own = _straight_line_warm_start(scenario)
    else:
        theta_own = initial_guess.copy()

    u_store = np.zeros((N, H, 2))

    # Seed belief Parameters and duals.
    for i in range(N):
        for j in scenario.neighbors(i):
            qps[i].theta_j_hat[j].value = theta_own[j].copy()
            qps[i].u_dual[j].value = np.zeros((H + 1, 4))

    edge_normals = _update_collision_normals(scenario, qps, theta_own)

    history = ADMMHistory()
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
                    f"Agent {i} subproblem returned status '{status}' "
                    f"at iteration {k}. Try a different solver or raise "
                    f"solver_opts['max_iter']."
                )
            theta_own[i] = qps[i].theta.value
            u_store[i] = qps[i].u.value

        # --- Step 2: broadcast own theta_i; each agent updates theta_j_hat ---
        for i in range(N):
            for j in scenario.neighbors(i):
                qps[i].theta_j_hat[j].value = theta_own[j].copy()

        # --- Step 3: dual update on each directed edge (i, j) ---
        for i in range(N):
            for j in scenario.neighbors(i):
                tilde_val = qps[i].tilde[j].value
                qps[i].u_dual[j].value = qps[i].u_dual[j].value + (tilde_val - theta_own[j])

        # --- Step 4: re-linearize collision normals ---
        if include_collision:
            edge_normals = _update_collision_normals(scenario, qps, theta_own)

        # --- Step 5: residuals ---
        primal_sq = 0.0
        for i in range(N):
            for j in scenario.neighbors(i):
                tilde_val = qps[i].tilde[j].value
                primal_sq += float(np.sum((tilde_val - theta_own[j]) ** 2))
        primal_res = np.sqrt(primal_sq)

        dual_res = rho * np.sqrt(float(np.sum((theta_own - theta_own_prev) ** 2)))

        # --- Diagnostics ---
        obj = _compute_local_objective(scenario, theta_own, u_store)
        viol = (
            _compute_constraint_violation(scenario, theta_own, edge_normals)
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
        history.dual_residual.append(dual_res)
        history.objective.append(obj)
        history.constraint_violation.append(viol)
        history.rho.append(rho)
        history.max_slack.append(max_slack)

        if verbose:
            print(
                f"iter {k:3d} | r={primal_res:.3e} | s={dual_res:.3e} "
                f"| J={obj:.4e} | viol={viol:.3e} | slack={max_slack:.3e} "
                f"| rho={rho:.2f}"
            )

        # --- Combined Boyd-style tolerance check ---
        p = 2 * scenario.graph.number_of_edges() * (H + 1) * 4  # directed edges
        n = N * (H + 1) * 4
        belief_norm = np.sqrt(
            sum(float(np.sum(theta_own[j] ** 2)) for i in range(N) for j in scenario.neighbors(i))
        )
        eps_pri = np.sqrt(p) * eps_abs + eps_rel * belief_norm
        eps_dual = np.sqrt(n) * eps_abs + eps_rel * rho * np.sqrt(
            sum(
                float(np.sum(qps[i].u_dual[j].value ** 2))
                for i in range(N)
                for j in scenario.neighbors(i)
            )
        )
        final_iter = k + 1
        if primal_res < eps_pri and dual_res < eps_dual:
            converged = True
            break

        # --- Adaptive rho (He-Yang-Wang residual balancing) ---
        # Because rho is baked into the QP as a constant for DPP, changing
        # it requires REBUILDING the QPs. We snapshot Parameter values,
        # rebuild, then restore (with the scaled-form dual rescaling).
        if adaptive_rho:
            new_rho = None
            dual_scale = 1.0
            if primal_res > mu * dual_res:
                new_rho = rho * tau_incr
                dual_scale = 1.0 / tau_incr
            elif dual_res > mu * primal_res:
                new_rho = rho / tau_decr
                dual_scale = tau_decr

            if new_rho is not None:
                state = _capture_state(qps, scenario)
                qps = _build_qps(scenario, new_rho, include_collision, collision_slack_weight)
                _restore_state(qps, state, scenario, dual_scale=dual_scale)
                rho = new_rho

    return ADMMResult(
        x=theta_own,
        u=u_store,
        iterations=final_iter,
        converged=converged,
        history=history,
    )
