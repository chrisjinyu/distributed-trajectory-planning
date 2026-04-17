"""Per-agent MPC QP builder (distributed Consensus ADMM formulation).

Owner: Yuyang Wu.
temporary stub for unblocking ADMM dev

Formulation
-----------
Each agent i maintains:
  - theta_i in R^{(H+1) x 4}  -- its OWN trajectory (dynamics + tracking).
  - tilde_theta_i^(j) for each neighbor j in N_i -- i's local BELIEF of j's
    trajectory. These appear in collision constraints and in the consensus
    penalty that drives them toward agreement with j's own theta_j.

Consensus constraint (ONE per directed edge (i, j) with j in N_i):
    tilde_theta_i^(j) = theta_j

The dual variable lambda_ij (in scaled form, call it u_ij) enforces this
equality. Note it is NOT theta_i = theta_j. Agent i's own trajectory is
never constrained to equal any neighbor's; only its belief of j must match
j's actual theta_j.

Local QP (per agent i, per ADMM iteration):

    minimize     sum_k [ ||theta_i[k] - x_ref_i[k]||_Q^2 ]    # tracking
               + sum_k [ ||u_i[k]||_R^2 ]                      # input
               + sum_{j in N_i} (rho/2) * ||tilde_theta_i^(j) - theta_j_hat + u_ij||^2
               + w * sum_{j in N_i} ||s_ij||^2                # collision slack
    subject to
        theta_i[0] = x0_i
        theta_i[k+1] = A theta_i[k] + B u_i[k]
        ||u_i[k]||_inf <= u_max
        n_ij[k]^T (p_i[k] - tilde_p_i^(j)[k]) + s_ij[k] >= d_min,   s_ij >= 0

The crucial point: COLLISION USES tilde_p_i^(j), NOT p_j directly. Agent i
plans against its own belief of where j is; consensus drives that belief
toward the truth.

Public interface (STABLE)
-------------------------
    build_local_qp(scenario, agent_id,
                   include_collision=True,
                   collision_slack_weight=1e4) -> LocalQP
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np

from dtp.dynamics import double_integrator_2d
from dtp.scenario import Scenario


@dataclass
class LocalQP:
    """Container for the per-agent ADMM subproblem."""

    problem: cp.Problem
    theta: cp.Variable  # own trajectory (H+1, 4)
    u: cp.Variable  # own controls   (H, 2)
    tilde: dict[int, cp.Variable] = field(default_factory=dict)  # belief of each neighbor
    x0: cp.Parameter = None  # type: ignore[assignment]
    rho: cp.Parameter = None  # type: ignore[assignment]
    theta_j_hat: dict[int, cp.Parameter] = field(default_factory=dict)  # broadcast from j
    u_dual: dict[int, cp.Parameter] = field(default_factory=dict)  # scaled dual per edge
    n_ij: dict[int, cp.Parameter] = field(default_factory=dict)  # collision normals
    slack: dict[int, cp.Variable] = field(default_factory=dict)  # collision slack


def build_local_qp(
    scenario: Scenario,
    agent_id: int,
    include_collision: bool = True,
    collision_slack_weight: float = 1e4,
) -> LocalQP:
    """Construct agent i's local ADMM subproblem."""
    i = agent_id
    H = scenario.H
    A, B = double_integrator_2d(scenario.dt)
    neighbors = scenario.neighbors(i)

    # --- Decision variables ---
    theta = cp.Variable((H + 1, 4), name=f"theta_{i}")
    u = cp.Variable((H, 2), name=f"u_{i}")
    tilde: dict[int, cp.Variable] = {
        j: cp.Variable((H + 1, 4), name=f"tilde_{i}_of_{j}") for j in neighbors
    }

    # --- Parameters ADMM mutates ---
    x0 = cp.Parameter(4, name=f"x0_{i}", value=scenario.x0[i])
    rho = cp.Parameter(nonneg=True, name="rho", value=1.0)

    theta_j_hat: dict[int, cp.Parameter] = {}
    u_dual: dict[int, cp.Parameter] = {}
    n_ij: dict[int, cp.Parameter] = {}

    for j in neighbors:
        theta_j_hat[j] = cp.Parameter(
            (H + 1, 4), name=f"theta_j_hat_{i}_{j}", value=np.zeros((H + 1, 4))
        )
        u_dual[j] = cp.Parameter((H + 1, 4), name=f"u_dual_{i}{j}", value=np.zeros((H + 1, 4)))
        n_ij[j] = cp.Parameter(
            (H + 1, 2),
            name=f"n_{i}{j}",
            value=np.tile([1.0, 0.0], (H + 1, 1)),
        )

    # --- Costs ---
    Q = scenario.Q
    R = scenario.R
    x_ref = scenario.x_ref[i]

    tracking_cost = sum(cp.quad_form(theta[k] - x_ref[k], cp.psd_wrap(Q)) for k in range(H + 1))
    input_cost = sum(cp.quad_form(u[k], cp.psd_wrap(R)) for k in range(H))
    local_cost = tracking_cost + input_cost

    # Consensus penalty (scaled form) on the BELIEF of each neighbor:
    #   sum_{j in N_i} (rho/2) || tilde_theta_i^(j) - theta_j_hat + u_dual[j] ||^2
    consensus_cost = 0
    for j in neighbors:
        diff = tilde[j] - theta_j_hat[j] + u_dual[j]
        consensus_cost = consensus_cost + (rho / 2) * cp.sum_squares(diff)

    # --- Constraints on own trajectory ---
    constraints = [theta[0] == x0]
    for k in range(H):
        constraints.append(theta[k + 1] == A @ theta[k] + B @ u[k])
        constraints.append(cp.norm_inf(u[k]) <= scenario.u_max)

    # --- Linearized collision half-spaces using i's own theta and i's BELIEFS ---
    # n_ij[k]^T (p_i[k] - tilde_p_i^(j)[k]) + s_ij[k] >= d_min
    slack: dict[int, cp.Variable] = {}
    slack_cost = 0
    if include_collision:
        for j in neighbors:
            p_i = theta[:, 0:2]
            p_tilde_j = tilde[j][:, 0:2]
            inner = cp.sum(cp.multiply(n_ij[j], p_i - p_tilde_j), axis=1)
            s = cp.Variable(H + 1, nonneg=True, name=f"slack_{i}{j}")
            slack[j] = s
            constraints.append(inner + s >= scenario.d_min)
            slack_cost = slack_cost + collision_slack_weight * cp.sum_squares(s)

    objective = cp.Minimize(local_cost + consensus_cost + slack_cost)
    problem = cp.Problem(objective, constraints)

    return LocalQP(
        problem=problem,
        theta=theta,
        u=u,
        tilde=tilde,
        x0=x0,
        rho=rho,
        theta_j_hat=theta_j_hat,
        u_dual=u_dual,
        n_ij=n_ij,
        slack=slack,
    )
