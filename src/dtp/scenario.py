"""Scenario configuration: shared problem parameters for the multi-agent MPC.

A single Scenario object is passed to mpc, admm, centralized, and penalty
modules so everyone works from the same cost weights, horizons, and bounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np


@dataclass
class Scenario:
    """Shared configuration for a multi-agent trajectory planning instance.

    Attributes
    ----------
    N : int
        Number of drones.
    H : int
        MPC horizon length (number of control intervals).
    dt : float
        Sampling period in seconds.
    x0 : ndarray, shape (N, 4)
        Initial states. Row i is drone i's state [px, py, vx, vy].
    x_ref : ndarray, shape (N, H+1, 4)
        Reference state trajectory. Element [i, k, :] is drone i's
        reference at time step k. If you only have goal positions, call
        Scenario.from_goals() to construct a stationary reference.
    Q : ndarray, shape (4, 4)
        State-tracking cost weight (symmetric PSD). Same for all agents.
    R : ndarray, shape (2, 2)
        Input cost weight (symmetric PD). Same for all agents.
    u_max : float
        Infinity-norm bound on the control input: ||u||_inf <= u_max.
    d_min : float
        Minimum safe separation between any two drones.
    graph : nx.Graph
        Communication graph G = (V, E). Nodes are drone indices 0..N-1.
        Edges represent bidirectional communication / collision coupling.
    """

    N: int
    H: int
    dt: float
    x0: np.ndarray
    x_ref: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    u_max: float
    d_min: float
    graph: nx.Graph = field(default_factory=nx.Graph)

    def __post_init__(self) -> None:
        # Light shape validation so bugs surface early.
        if self.x0.shape != (self.N, 4):
            raise ValueError(f"x0 shape {self.x0.shape}, expected ({self.N}, 4)")
        if self.x_ref.shape != (self.N, self.H + 1, 4):
            raise ValueError(
                f"x_ref shape {self.x_ref.shape}, expected ({self.N}, {self.H + 1}, 4)"
            )
        if self.Q.shape != (4, 4) or self.R.shape != (2, 2):
            raise ValueError("Q must be 4x4 and R must be 2x2")
        if self.graph.number_of_nodes() != self.N:
            # Auto-populate a complete graph if the caller didn't specify one.
            self.graph = nx.complete_graph(self.N)

    def neighbors(self, i: int) -> list[int]:
        """Sorted list of neighbor indices for agent i."""
        return sorted(self.graph.neighbors(i))

    @classmethod
    def from_goals(
        cls,
        x0: np.ndarray,
        goals: np.ndarray,
        H: int,
        dt: float,
        Q: np.ndarray | None = None,
        R: np.ndarray | None = None,
        u_max: float = 2.0,
        d_min: float = 0.5,
        graph: nx.Graph | None = None,
    ) -> Scenario:
        """Build a Scenario with a stationary goal reference.

        Each drone's reference trajectory is simply its goal position with
        zero velocity, repeated for all H+1 time steps.
        """
        N = x0.shape[0]
        if goals.shape != (N, 2):
            raise ValueError(f"goals shape {goals.shape}, expected ({N}, 2)")
        x_ref = np.zeros((N, H + 1, 4))
        x_ref[:, :, 0:2] = goals[:, None, :]  # broadcast goal position over time
        # velocity reference stays zero
        if Q is None:
            Q = np.diag([10.0, 10.0, 1.0, 1.0])  # prioritize position over velocity
        if R is None:
            R = 0.1 * np.eye(2)
        if graph is None:
            graph = nx.complete_graph(N)
        return cls(
            N=N,
            H=H,
            dt=dt,
            x0=x0,
            x_ref=x_ref,
            Q=Q,
            R=R,
            u_max=u_max,
            d_min=d_min,
            graph=graph,
        )
