"""Per-agent MPC QP builder (distributed Consensus ADMM, DPP-compliant).

Owner: Yuyang Wu
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

Local QP (per agent i, per ADMM iteration):

    minimize     sum_k [ ||theta_i[k] - x_ref_i[k]||_Q^2 ]    # tracking
               + sum_k [ ||u_i[k]||_R^2 ]                      # input
               + sum_{j in N_i} consensus_j                    # DPP-expanded, see below
               + w * sum_{j in N_i} ||s_ij||^2                 # collision slack
    subject to
        theta_i[0] = x0_i
        theta_i[k+1] = A theta_i[k] + B u_i[k]
        ||u_i[k]||_inf <= u_max
        n_ij[k]^T (p_i[k] - tilde_p_i^(j)[k]) + s_ij[k] >= d_min,   s_ij >= 0

DPP-compliant consensus expansion
---------------------------------
The scaled-ADMM consensus penalty (rho/2)*||tilde - theta_j_hat + u_dual||^2
expands (dropping the variable-free constant term) to

    (rho/2) * sum_squares(tilde)                       # constant * sum_squares(Variable)
  - rho      * sum(multiply(tilde, theta_j_hat - u_dual))  # constant * Variable * Parameter

Both terms are DPP provided rho is a *constant*, not a Parameter. That is
why rho enters build_local_qp as a float. When the ADMM driver changes rho
(e.g., adaptive residual balancing), it rebuilds the QPs.

Public interface (STABLE)
-------------------------
    build_local_qp(scenario, agent_id,
                   rho=1.0,
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
    theta: cp.Variable  # own trajectory
    u: cp.Variable  # own controls
    tilde: dict[int, cp.Variable] = field(default_factory=dict)  # beliefs per neighbor
    x0: cp.Parameter = None  # type: ignore[assignment]
    theta_j_hat: dict[int, cp.Parameter] = field(default_factory=dict)  # broadcast from j
    u_dual: dict[int, cp.Parameter] = field(default_factory=dict)  # scaled dual per edge
    n_ij: dict[int, cp.Parameter] = field(default_factory=dict)  # collision normals
    slack: dict[int, cp.Variable] = field(default_factory=dict)  # collision slack
    rho: float = 1.0  # baked-in at build time


def build_local_qp(
    scenario: Scenario,
    agent_id: int,
    rho: float = 1.0,
    include_collision: bool = True,
    collision_slack_weight: float = 1e4,
) -> LocalQP:
    """Construct agent i's local ADMM subproblem (DPP-compliant).

    Parameters
    ----------
    scenario : Scenario
    agent_id : int
    rho : float
        ADMM penalty parameter, baked into the objective as a constant to
        preserve DPP. Changing rho requires rebuilding the QP.
    include_collision : bool
    collision_slack_weight : float

    Returns
    -------
    LocalQP
    """
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

    # --- Parameters ADMM mutates between iterations ---
    x0 = cp.Parameter(4, name=f"x0_{i}", value=scenario.x0[i])

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

    # Tracking cost: quad_form(Variable - const, const_PSD). DPP.
    tracking_cost = sum(cp.quad_form(theta[k] - x_ref[k], cp.psd_wrap(Q)) for k in range(H + 1))
    # Input cost: quad_form(Variable, const_PSD). DPP.
    input_cost = sum(cp.quad_form(u[k], cp.psd_wrap(R)) for k in range(H))
    local_cost = tracking_cost + input_cost

    # Consensus penalty in DPP-expanded form. See module docstring for the algebra.
    #   (rho/2) * sum_squares(tilde)
    #   - rho * sum(multiply(tilde, theta_j_hat - u_dual))
    # The dropped constant (rho/2) * ||theta_j_hat - u_dual||^2 doesn't affect argmin.
    consensus_cost = 0
    for j in neighbors:
        # Quadratic term: constant coefficient * sum_squares(Variable). DPP.
        consensus_cost = consensus_cost + (rho / 2.0) * cp.sum_squares(tilde[j])
        # Linear term: constant * multiply(Variable, Parameter-expr). DPP.
        # Using cp.sum(cp.multiply(...)) instead of a dot product to handle the
        # (H+1, 4) shape cleanly.
        target = theta_j_hat[j] - u_dual[j]  # Parameter-affine expression
        consensus_cost = consensus_cost - rho * cp.sum(cp.multiply(tilde[j], target))

    # --- Constraints on own trajectory ---
    constraints = [theta[0] == x0]
    for k in range(H):
        constraints.append(theta[k + 1] == A @ theta[k] + B @ u[k])
        constraints.append(cp.norm_inf(u[k]) <= scenario.u_max)

    # --- Linearized collision half-spaces using i's theta and i's BELIEFS ---
    # n_ij[k]^T (p_i[k] - tilde_p_i^(j)[k]) + s_ij[k] >= d_min
    # multiply(Parameter, Variable) is DPP.
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
        theta_j_hat=theta_j_hat,
        u_dual=u_dual,
        n_ij=n_ij,
        slack=slack,
        rho=rho,
    )
