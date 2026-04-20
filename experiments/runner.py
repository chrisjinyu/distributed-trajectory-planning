"""Experiment runner: load a scenario config and run all three methods.

Loads a YAML config (see experiments/configs/) into a Scenario, runs the
penalty method, Consensus ADMM, and the centralized QP on it, and returns
a structured MethodResult per method plus shared ground-truth normals.

Used by the verification notebook and future report plots. Keeps the
three algorithm modules untouched.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from dtp.admm import ADMMResult, _straight_line_warm_start, solve_consensus_admm
from dtp.centralized import CentralizedSolution, solve_centralized
from dtp.penalty import PenaltyResult, solve_penalty_method
from dtp.scenario import Scenario
from dtp.utils import complete_graph, ring_graph

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_scenario(path: str | Path) -> Scenario:
    """Load a YAML scenario config and construct a Scenario."""
    with open(path) as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    N = int(cfg["N"])
    x0 = np.asarray(cfg["x0"], dtype=float)
    goals = np.asarray(cfg["goals"], dtype=float)

    graph_kind = cfg.get("graph", "complete")
    if graph_kind == "complete":
        graph = complete_graph(N)
    elif graph_kind == "ring":
        graph = ring_graph(N)
    else:
        raise ValueError(f"Unknown graph kind: {graph_kind!r}")

    return Scenario.from_goals(
        x0=x0,
        goals=goals,
        H=int(cfg["H"]),
        dt=float(cfg["dt"]),
        u_max=float(cfg["u_max"]),
        d_min=float(cfg["d_min"]),
        graph=graph,
    )


# ---------------------------------------------------------------------------
# Metric collection
# ---------------------------------------------------------------------------


@dataclass
class MethodResult:
    """Unified per-method metrics for comparison plots."""

    name: str
    x: np.ndarray  # (N, H+1, 4)
    u: np.ndarray  # (N, H, 2)
    iterations: int
    wall_time: float
    final_objective: float
    max_collision_violation: float  # over scenario.graph edges only
    min_pairwise_distance: float  # across all pairs, whole horizon
    history: Any  # method-specific history object (may be None)


def _min_pairwise_distance(x: np.ndarray) -> float:
    """Min distance between any two drones over the full horizon."""
    N = x.shape[0]
    best = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(x[i, :, 0:2] - x[j, :, 0:2], axis=1)
            best = min(best, float(np.min(d)))
    return best


def _max_violation_on_edges(scenario: Scenario, x: np.ndarray) -> float:
    """Max positive (d_min - ||p_i - p_j||) over graph edges and horizon."""
    worst = 0.0
    for i, j in scenario.graph.edges():
        d = np.linalg.norm(x[i, :, 0:2] - x[j, :, 0:2], axis=1)
        worst = max(worst, float(np.max(scenario.d_min - d)))
    return worst


def _objective(scenario: Scenario, x: np.ndarray, u: np.ndarray) -> float:
    Q, R = scenario.Q, scenario.R
    total = 0.0
    for i in range(scenario.N):
        dx = x[i] - scenario.x_ref[i]
        total += float(np.einsum("ka,ab,kb->", dx, Q, dx))
        total += float(np.einsum("ka,ab,kb->", u[i], R, u[i]))
    return total


# ---------------------------------------------------------------------------
# Method drivers
# ---------------------------------------------------------------------------


def run_penalty(scenario: Scenario, w: float = 1e3, **kwargs: Any) -> MethodResult:
    start = time.perf_counter()
    res: PenaltyResult = solve_penalty_method(scenario, w=w, **kwargs)
    elapsed = time.perf_counter() - start
    return MethodResult(
        name=f"penalty (w={w:g})",
        x=res.x,
        u=res.u,
        iterations=res.iterations,
        wall_time=elapsed,
        final_objective=_objective(scenario, res.x, res.u),
        max_collision_violation=_max_violation_on_edges(scenario, res.x),
        min_pairwise_distance=_min_pairwise_distance(res.x),
        history=res.history,
    )


def run_admm(scenario: Scenario, rho: float = 1.0, **kwargs: Any) -> MethodResult:
    start = time.perf_counter()
    res: ADMMResult = solve_consensus_admm(scenario, rho=rho, **kwargs)
    elapsed = time.perf_counter() - start
    return MethodResult(
        name=f"ADMM (rho={rho:g})",
        x=res.x,
        u=res.u,
        iterations=res.iterations,
        wall_time=elapsed,
        final_objective=_objective(scenario, res.x, res.u),
        max_collision_violation=_max_violation_on_edges(scenario, res.x),
        min_pairwise_distance=_min_pairwise_distance(res.x),
        history=res.history,
    )


def run_centralized(
    scenario: Scenario,
    n_ij: dict | None = None,
    scp_iters: int = 5,
    linearization_from: np.ndarray | None = None,
    **kwargs: Any,
) -> MethodResult:
    """Run the centralized joint QP with a small SCP loop.

    The collision constraint is linearized around the current iterate's
    positions, the joint QP is solved, and normals are recomputed from
    the new solution. Running a handful of SCP iterations produces a
    much better reference than one-shot linearization from straight-line
    warm-start.

    Parameters
    ----------
    linearization_from : ndarray, optional
        Shape ``(N, H+1, 4)``. If supplied, the initial collision normals
        are computed from this trajectory instead of the straight-line
        warm-start. Useful for head-on antipodal-swap scenarios where the
        straight-line linearization is degenerate and SCP gets stuck on
        a 1D-pass solution; seed with the penalty or ADMM output to break
        the symmetry.

    If ``n_ij`` is None and ``scp_iters == 0``, collisions are ignored.
    """
    start = time.perf_counter()

    if n_ij is None and scp_iters > 0:
        n_ij = build_centralized_normals(scenario, linearization_point=linearization_from)

    res: CentralizedSolution = solve_centralized(scenario, n_ij=n_ij, **kwargs)
    # SCP can collapse to infeasibility on tight, clustered linearizations
    # (e.g., N=8 antipodal swap seeded from a bad penalty trajectory). In
    # that case solve_centralized returns NaN-filled arrays; fall back to
    # the last good iterate rather than propagating NaNs through plotting.
    last_good = res if not np.isnan(res.x).any() else None
    for _ in range(scp_iters - 1 if n_ij is not None else 0):
        if np.isnan(res.x).any():
            if last_good is not None:
                res = last_good
            break
        last_good = res
        n_ij = build_centralized_normals(scenario, linearization_point=res.x)
        res = solve_centralized(scenario, n_ij=n_ij, **kwargs)
    if np.isnan(res.x).any() and last_good is not None:
        res = last_good

    elapsed = time.perf_counter() - start
    return MethodResult(
        name="centralized",
        x=res.x,
        u=res.u,
        iterations=scp_iters if n_ij is not None else 1,
        wall_time=elapsed,
        final_objective=res.objective,
        max_collision_violation=_max_violation_on_edges(scenario, res.x),
        min_pairwise_distance=_min_pairwise_distance(res.x),
        history=None,
    )


def build_centralized_normals(
    scenario: Scenario, linearization_point: np.ndarray | None = None
) -> dict[tuple[int, int], np.ndarray]:
    """Build (i, j)-keyed collision normals for the centralized solver.

    Uses the given (N, H+1, 4) trajectory as the linearization point.
    If not supplied, uses the perturbed straight-line warm-start from
    admm._straight_line_warm_start so antipodal-swap cases don't produce
    degenerate zero-vector normals at horizon midpoint.

    Only includes edges in scenario.graph to match the distributed
    methods' coupling structure.
    """
    if linearization_point is None:
        linearization_point = _straight_line_warm_start(scenario)

    edges = [tuple(sorted(e)) for e in scenario.graph.edges()]
    n_ij: dict[tuple[int, int], np.ndarray] = {}
    for i, j in edges:
        p_i = linearization_point[i, :, 0:2]
        p_j = linearization_point[j, :, 0:2]
        diff = p_i - p_j
        norms = np.linalg.norm(diff, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-6)
        n_ij[(i, j)] = diff / norms
    return n_ij
