"""Consensus ADMM driver for distributed multi-agent trajectory planning.

Owner: Christian Yu.

Formulation
-----------
Each agent i maintains its own trajectory theta_i plus, for each neighbor
j in N_i, a local belief tilde_theta_i^(j). The consensus constraint
tilde_theta_i^(j) = theta_j is enforced via scaled dual variables u_dual_ij
that are updated each iteration.

Scaled ADMM form throughout. Per-iteration flow:

    1. Each agent solves its local QP (parallel in principle; serial here).
       -- Output: theta_i (own) and tilde_theta_i^(j) (belief of each j).
    2. Agents BROADCAST their own theta_i to neighbors.
       -- Each agent i then knows theta_j for j in N_i.
    3. Dual update on each directed edge (i, j):
           u_dual_ij <- u_dual_ij + (tilde_theta_i^(j) - theta_j)
       This drives agent i's BELIEF toward neighbor j's ACTUAL trajectory.
    4. Collision normals are re-linearized from own-theta positions.
    5. Primal / dual residuals computed; loop terminates on tolerance.

At convergence, tilde_theta_i^(j) = theta_j for every edge, so agent i's
plan (which uses its belief in collision constraints) is consistent with
j's actual trajectory.

Returns
-------
ADMMResult.x stacks each agent's own theta_i, shape (N, H+1, 4).
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
    """Max violation of the linearized collision half-spaces over all edges,
    evaluated on each agent's OWN positions (not beliefs)."""
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
    """Recompute linearization normals from the agents' OWN trajectories.

    theta_own : (N, H+1, 4)
    """
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
    """Per-agent straight-line interpolation with a small symmetry-breaking
    bump to avoid degenerate collision normals at mid-horizon crossings.

    Returns : (N, H+1, 4)
    """
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
    """Run distributed Consensus ADMM (tilde-belief formulation).

    Parameters
    ----------
    scenario : Scenario
    rho : float
        Initial ADMM penalty parameter.
    max_iter : int
    eps_abs, eps_rel : float
        Tolerances for the combined stopping criterion (Boyd 2011 §3.3.1).
    solver : str
        CVXPY solver. Default Clarabel for stability under parameter changes.
    solver_opts : dict, optional
        Extra kwargs forwarded to Problem.solve.
    initial_guess : ndarray of shape (N, H+1, 4), optional
        Per-agent trajectory warm-start. Each agent's belief of j is
        seeded from the j-th entry.
    include_collision : bool
    collision_slack_weight : float
    adaptive_rho : bool
        He-Yang-Wang residual balancing.
    tau_incr, tau_decr, mu : float
        Adaptive-rho hyperparameters.
    verbose : bool

    Returns
    -------
    ADMMResult
        .x is (N, H+1, 4) stack of each agent's own theta_i.
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

    # --- Build per-agent QPs once; reuse across iterations ---
    qps: dict[int, LocalQP] = {
        i: build_local_qp(
            scenario,
            i,
            include_collision=include_collision,
            collision_slack_weight=collision_slack_weight,
        )
        for i in range(N)
    }
    for qp in qps.values():
        qp.rho.value = rho

    # --- Initial own-trajectory warm-start ---
    if initial_guess is None:
        theta_own = _straight_line_warm_start(scenario)
    else:
        theta_own = initial_guess.copy()

    u_store = np.zeros((N, H, 2))

    # --- Seed belief parameters: each agent believes its neighbors follow the
    #     straight-line warm-start as well. Seed duals to zero.
    for i in range(N):
        for j in scenario.neighbors(i):
            qps[i].theta_j_hat[j].value = theta_own[j].copy()
            qps[i].u_dual[j].value = np.zeros((H + 1, 4))

    # --- Seed collision normals from own-trajectory warm-start ---
    edge_normals = _update_collision_normals(scenario, qps, theta_own)

    history = ADMMHistory()
    converged = False
    final_iter = 0

    for k in range(max_iter):
        theta_own_prev = theta_own.copy()

        # --- Step 1: local QP solves (primal update) ---
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
                    f"at iteration {k}. Try a different solver or pass "
                    f"solver_opts with higher max_iter."
                )
            theta_own[i] = qps[i].theta.value
            u_store[i] = qps[i].u.value

        # --- Step 2: broadcast own theta_i; each agent updates theta_j_hat ---
        for i in range(N):
            for j in scenario.neighbors(i):
                # Agent i receives neighbor j's just-computed theta_j.
                qps[i].theta_j_hat[j].value = theta_own[j].copy()

        # --- Step 3: dual update on each directed edge (i, j) ---
        # u_dual_ij <- u_dual_ij + (tilde_theta_i^(j) - theta_j)
        # After step 2, theta_j_hat[j] == theta_j (the just-computed value),
        # so this is symmetric to the scaled-ADMM convention.
        for i in range(N):
            for j in scenario.neighbors(i):
                tilde_val = qps[i].tilde[j].value
                qps[i].u_dual[j].value = qps[i].u_dual[j].value + (tilde_val - theta_own[j])

        # --- Step 4: re-linearize collision normals ---
        if include_collision:
            edge_normals = _update_collision_normals(scenario, qps, theta_own)

        # --- Step 5: residuals ---
        # Primal residual = sum over directed edges of ||tilde_i^(j) - theta_j||^2
        primal_sq = 0.0
        for i in range(N):
            for j in scenario.neighbors(i):
                tilde_val = qps[i].tilde[j].value
                primal_sq += float(np.sum((tilde_val - theta_own[j]) ** 2))
        primal_res = np.sqrt(primal_sq)

        # Dual residual proxy: change in own theta scaled by rho.
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
        # Count scalars in the primal residual: sum_i |N_i| * (H+1) * 4
        p = 2 * scenario.graph.number_of_edges() * (H + 1) * 4  # directed edges
        n = N * (H + 1) * 4
        # norm of the primal "A z" side = sqrt(sum_i sum_j ||theta_j||^2)
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

        # --- Adaptive rho ---
        if adaptive_rho:
            if primal_res > mu * dual_res:
                rho = rho * tau_incr
                for qp in qps.values():
                    qp.rho.value = rho
                    for j in qp.u_dual:
                        qp.u_dual[j].value = qp.u_dual[j].value / tau_incr
            elif dual_res > mu * primal_res:
                rho = rho / tau_decr
                for qp in qps.values():
                    qp.rho.value = rho
                    for j in qp.u_dual:
                        qp.u_dual[j].value = qp.u_dual[j].value * tau_decr

    return ADMMResult(
        x=theta_own,
        u=u_store,
        iterations=final_iter,
        converged=converged,
        history=history,
    )
