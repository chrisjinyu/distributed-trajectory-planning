"""Smoke tests: verify the environment is wired up correctly.

If these pass for all teammates, we know CVXPY, OSQP, and Clarabel are
installed, importable, and able to solve a small QP. These are the
minimum conditions needed before anyone starts implementing the real
algorithms.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from dtp.dynamics import double_integrator_2d
from dtp.utils import random_initial_positions, ring_graph


def test_imports_work():
    """All core algorithm modules import without error."""
    from dtp import admm, centralized, mpc, penalty, utils  # noqa: F401


def test_double_integrator_shapes():
    A, B = double_integrator_2d(dt=0.1)
    assert A.shape == (4, 4)
    assert B.shape == (4, 2)


def test_double_integrator_consistency():
    """Simulating one step should match the formula x' = Ax + Bu."""
    A, B = double_integrator_2d(dt=0.1)
    x0 = np.array([0.0, 0.0, 1.0, 0.0])  # at origin, moving +x at 1 m/s
    u0 = np.array([0.0, 0.0])            # no acceleration
    x1 = A @ x0 + B @ u0
    # After 0.1 s with no accel: should be at (0.1, 0) with same velocity.
    np.testing.assert_allclose(x1, [0.1, 0.0, 1.0, 0.0])


def test_cvxpy_osqp_solves_qp():
    """CVXPY + OSQP can solve a trivial QP."""
    x = cp.Variable(3)
    P = np.eye(3)
    q = np.array([1.0, -2.0, 0.5])
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, P) + q @ x))
    prob.solve(solver=cp.OSQP)
    assert prob.status == cp.OPTIMAL
    # Unconstrained solution: x* = -q
    np.testing.assert_allclose(x.value, -q, atol=1e-4)


def test_cvxpy_clarabel_solves_qp():
    """Clarabel is installed and usable as an alternative to OSQP."""
    x = cp.Variable(3)
    P = np.eye(3)
    q = np.array([1.0, -2.0, 0.5])
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, P) + q @ x))
    prob.solve(solver=cp.CLARABEL)
    assert prob.status == cp.OPTIMAL
    np.testing.assert_allclose(x.value, -q, atol=1e-4)


def test_ring_graph_structure():
    g = ring_graph(4)
    assert g.number_of_nodes() == 4
    assert g.number_of_edges() == 4  # cycle on 4 nodes


def test_initial_positions_shape():
    p = random_initial_positions(n=6, radius=5.0, seed=0)
    assert p.shape == (6, 2)
    # Roughly on the circle of radius 5 (with small jitter)
    dists = np.linalg.norm(p, axis=1)
    assert np.all(np.abs(dists - 5.0) < 1.0)
