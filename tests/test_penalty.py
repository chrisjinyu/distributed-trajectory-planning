"""Smoke tests for the distributed penalty-method baseline.

These assert that solve_penalty_method returns correctly-shaped output,
reaches goal states on a simple swap, and drives collision slack to a
small residual when the penalty weight is large enough.
"""

from __future__ import annotations

import numpy as np

from dtp.mpc import PenaltyLocalQP, build_penalty_qp
from dtp.penalty import PenaltyResult, solve_penalty_method
from dtp.scenario import Scenario


def _two_drone_swap_scenario() -> Scenario:
    x0 = np.array([[-3.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]])
    goals = np.array([[3.0, 0.0], [-3.0, 0.0]])
    return Scenario.from_goals(x0, goals, H=20, dt=0.2, u_max=3.0, d_min=0.5)


def test_build_penalty_qp_shapes():
    """Variables and Parameters have the expected shapes for a 2-drone scenario."""
    scn = _two_drone_swap_scenario()
    qp = build_penalty_qp(scn, agent_id=0, collision_penalty_weight=1e3)

    assert isinstance(qp, PenaltyLocalQP)
    assert qp.theta.shape == (scn.H + 1, 4)
    assert qp.u.shape == (scn.H, 2)
    assert qp.x0.shape == (4,)
    # Only one neighbor in a 2-drone complete graph.
    neighbors = scn.neighbors(0)
    assert list(qp.n_ij.keys()) == neighbors
    for j in neighbors:
        assert qp.n_ij[j].shape == (scn.H + 1, 2)
        assert qp.collision_rhs[j].shape == (scn.H + 1,)
        assert qp.slack[j].shape == (scn.H + 1,)


def test_two_drone_swap_converges_and_avoids():
    """On the canonical 2-drone swap, penalty method reaches goals without collision."""
    scn = _two_drone_swap_scenario()
    result = solve_penalty_method(scn, w=1e3, max_iter=60)

    assert isinstance(result, PenaltyResult)
    assert result.x.shape == (scn.N, scn.H + 1, 4)
    assert result.u.shape == (scn.N, scn.H, 2)
    assert len(result.history.primal_residual) == result.iterations

    # Both drones within 0.05 of their goal at horizon end.
    for i in range(scn.N):
        goal = scn.x_ref[i, -1, 0:2]
        end_pos = result.x[i, -1, 0:2]
        assert np.linalg.norm(end_pos - goal) < 0.05

    # Min pairwise distance respects d_min up to small numerical slack.
    min_dist = np.inf
    for k in range(scn.H + 1):
        d = np.linalg.norm(result.x[0, k, 0:2] - result.x[1, k, 0:2])
        min_dist = min(min_dist, d)
    assert min_dist >= scn.d_min - 1e-2


def test_slack_shrinks_as_w_grows():
    """Increasing the penalty weight pushes the max constraint violation down."""
    scn = _two_drone_swap_scenario()
    low = solve_penalty_method(scn, w=1.0, max_iter=40)
    high = solve_penalty_method(scn, w=1e4, max_iter=40)

    assert (
        high.history.constraint_violation[-1] <= low.history.constraint_violation[-1] + 1e-6
    ), "Higher penalty weight should not produce a larger constraint violation."


def test_no_collision_mode_ignores_slack():
    """With include_collision=False, slack dict is empty and violation is zero."""
    scn = _two_drone_swap_scenario()
    qp = build_penalty_qp(scn, agent_id=0, include_collision=False)
    assert qp.slack == {}
    assert qp.n_ij == {}
    assert qp.collision_rhs == {}

    result = solve_penalty_method(scn, w=1e3, max_iter=20, include_collision=False)
    assert all(v == 0.0 for v in result.history.constraint_violation)
    assert all(s == 0.0 for s in result.history.max_slack)
