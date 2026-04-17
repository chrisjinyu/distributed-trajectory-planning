"""Shared helpers: communication graph construction, logging, plotting."""

from __future__ import annotations

import networkx as nx
import numpy as np


def complete_graph(n: int) -> nx.Graph:
    """All-to-all communication graph on n nodes. Good starting point for debugging."""
    return nx.complete_graph(n)


def ring_graph(n: int) -> nx.Graph:
    """Ring communication topology: each drone talks to 2 neighbors."""
    return nx.cycle_graph(n)


def random_initial_positions(n: int, radius: float, seed: int | None = None) -> np.ndarray:
    """Place n drones on a circle of given radius. Goals are the antipodal point.

    This produces a classic 'swap through the middle' scenario that forces
    the collision-avoidance constraints to activate.
    """
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rng.uniform(0, 0.1, n)
    return np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
