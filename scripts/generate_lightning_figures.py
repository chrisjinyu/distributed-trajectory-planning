"""Generate the four figures used in the lightning-talk slides.

Outputs vector PDFs under docs/lightning_talk/figures/ so the Beamer source
can \\includegraphics them directly. Reuses the solvers in src/dtp and the
scenario YAMLs in experiments/configs/ -- no new algorithmic code.

Figures produced:
  fig_s2_trajectories.pdf   -- 3 xy-panels (penalty / ADMM / centralized), S2
  fig_w_sensitivity.pdf     -- two-panel w-sweep on S2 (violation + iterations)
  fig_rho_sensitivity.pdf   -- two-panel rho-sweep on S2 (violation + iterations) for ADMM
  fig_s3_trajectories.pdf   -- 3 xy-panels on S3 (N=8)

Invoke from the repo root:
    uv run python scripts/generate_lightning_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.runner import (  # noqa: E402
    load_scenario,
    run_admm,
    run_centralized,
    run_penalty,
)

CONFIG_DIR = REPO_ROOT / "experiments" / "configs"
FIG_DIR = REPO_ROOT / "docs" / "lightning_talk" / "figures"

PANEL_COLORS = {"penalty": "#d62728", "ADMM": "#1f77b4", "centralized": "#2ca02c"}


def _print_row(label: str, res) -> None:
    print(
        f"  {label:>14s} | iters={res.iterations:3d} "
        f"| J={res.final_objective:12.4e} "
        f"| viol={res.max_collision_violation:.3e} "
        f"| min_d={res.min_pairwise_distance:.4f}"
    )


def _plot_xy_panel(ax, scenario, x, title, tick_every=5):
    N = scenario.N
    H = scenario.H
    tick_idx = np.arange(tick_every, H, tick_every)
    for i in range(N):
        line, = ax.plot(x[i, :, 0], x[i, :, 1], linewidth=1.5, alpha=0.85)
        color = line.get_color()
        # Direction arrows at regular timesteps: use quiver so the
        # arrowhead points along the instantaneous velocity and the
        # time-ordering of the path is unambiguous.
        dx = x[i, tick_idx + 1, 0] - x[i, tick_idx, 0]
        dy = x[i, tick_idx + 1, 1] - x[i, tick_idx, 1]
        ax.quiver(
            x[i, tick_idx, 0],
            x[i, tick_idx, 1],
            dx,
            dy,
            color=color,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.005,
            headwidth=5.0,
            headlength=6.0,
            headaxislength=5.5,
            alpha=0.95,
            zorder=3,
        )
        ax.plot(x[i, 0, 0], x[i, 0, 1], "s", markersize=7, color="k", alpha=0.7)
        ax.plot(
            scenario.x_ref[i, -1, 0],
            scenario.x_ref[i, -1, 1],
            "*",
            markersize=11,
            color="k",
            alpha=0.7,
        )
    ax.set_title(title, fontsize=13)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x (m)", fontsize=11)


def figure_s2_trajectories():
    print("[fig_s2_trajectories] solving S2 ...")
    scenario = load_scenario(CONFIG_DIR / "four_drone_ring.yaml")
    penalty = run_penalty(scenario, w=1e3, max_iter=200)
    admm = run_admm(scenario, rho=5.0, max_iter=300, collision_slack_weight=1e7)
    # Seed centralized SCP from ADMM's output; the default straight-line
    # warm-start is degenerate on antipodal swaps (normals go to zero at
    # horizon midpoint) and drives OSQP to a suboptimal solution.
    cent = run_centralized(scenario, scp_iters=5, linearization_from=admm.x)

    _print_row("penalty", penalty)
    _print_row("ADMM", admm)
    _print_row("centralized", cent)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharex=True, sharey=True)
    _plot_xy_panel(axes[0], scenario, penalty.x, "Penalty  (w=10$^3$)")
    _plot_xy_panel(axes[1], scenario, admm.x, "ADMM  ($\\rho$=5)")
    _plot_xy_panel(axes[2], scenario, cent.x, "Centralized")
    axes[0].set_ylabel("y (m)", fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / "fig_s2_trajectories.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig_s2_trajectories] wrote {out}")


def _sensitivity_plot(ax_v, ax_i, xs, viols, iters, x_label, color, title_param):
    """Two-panel sensitivity plot: violation (linear y, [0, 0.5]) and iterations."""
    ax_v.semilogx(xs, viols, "o-", color=color, linewidth=1.8, markersize=7)
    ax_v.set_xlabel(x_label, fontsize=11)
    ax_v.set_ylabel("Max constraint violation (m)", fontsize=11)
    ax_v.set_ylim(0.0, 0.5)
    ax_v.grid(True, which="both", alpha=0.3)
    ax_v.set_title(f"(a) Violation vs ${title_param}$", fontsize=12)

    ax_i.semilogx(xs, iters, "s-", color=color, linewidth=1.8, markersize=7)
    ax_i.set_xlabel(x_label, fontsize=11)
    ax_i.set_ylabel("Iterations", fontsize=11)
    ax_i.grid(True, which="both", alpha=0.3)
    ax_i.set_title(f"(b) Iterations vs ${title_param}$", fontsize=12)


def figure_w_sensitivity():
    print("[fig_w_sensitivity] sweeping w on S2 ...")
    scenario = load_scenario(CONFIG_DIR / "four_drone_ring.yaml")
    ws = np.logspace(2, 4, 8).tolist()
    viols, iters = [], []
    for w in ws:
        res = run_penalty(scenario, w=w, max_iter=10000)
        viols.append(res.max_collision_violation)
        iters.append(res.iterations)
        print(f"  w={w:>8.1f} | iters={res.iterations:5d} | viol={res.max_collision_violation:.3e}")

    fig, (ax_v, ax_i) = plt.subplots(1, 2, figsize=(9.5, 3.4))
    _sensitivity_plot(ax_v, ax_i, ws, viols, iters,
                      "Penalty weight $w$", PANEL_COLORS["penalty"], "w")
    fig.tight_layout()
    out = FIG_DIR / "fig_w_sensitivity.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig_w_sensitivity] wrote {out}")


def figure_rho_sensitivity():
    print("[fig_rho_sensitivity] sweeping rho on S2 ...")
    scenario = load_scenario(CONFIG_DIR / "four_drone_ring.yaml")
    rhos = np.logspace(-1, 3, 9).tolist()  # 0.1 to 1000
    viols, iters = [], []
    for rho in rhos:
        res = run_admm(scenario, rho=rho, max_iter=10000, collision_slack_weight=1e7)
        viols.append(res.max_collision_violation)
        iters.append(res.iterations)
        print(
            f"  rho={rho:>8.2f} | iters={res.iterations:5d} "
            f"| viol={res.max_collision_violation:.3e}"
        )

    fig, (ax_v, ax_i) = plt.subplots(1, 2, figsize=(9.5, 3.4))
    _sensitivity_plot(ax_v, ax_i, rhos, viols, iters,
                      "ADMM penalty $\\rho$", PANEL_COLORS["ADMM"], "\\rho")
    fig.tight_layout()
    out = FIG_DIR / "fig_rho_sensitivity.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig_rho_sensitivity] wrote {out}")


def figure_s3_trajectories():
    print("[fig_s3_trajectories] solving S3 ...")
    scenario = load_scenario(CONFIG_DIR / "eight_drone_complete.yaml")
    penalty = run_penalty(scenario, w=2e3, max_iter=500)
    admm = run_admm(
        scenario,
        rho=300.0,
        max_iter=500,
        collision_slack_weight=1e7,
        eps_abs=1e-4,
        eps_rel=1e-3,
        initial_guess=penalty.x.copy(),
    )
    # Seed centralized SCP from ADMM's output (feasible), not penalty
    # (infeasible). Otherwise the first SCP iteration starts from a
    # collision-embedded linearization and produces oversized detours.
    cent = run_centralized(scenario, scp_iters=5, linearization_from=admm.x)

    _print_row("penalty", penalty)
    _print_row("ADMM", admm)
    _print_row("centralized", cent)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharex=True, sharey=True)
    _plot_xy_panel(axes[0], scenario, penalty.x, "Penalty  (w=2$\\cdot$10$^3$)")
    _plot_xy_panel(axes[1], scenario, admm.x, "ADMM  ($\\rho$=300, seeded)")
    _plot_xy_panel(axes[2], scenario, cent.x, "Centralized")
    axes[0].set_ylabel("y (m)", fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / "fig_s3_trajectories.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig_s3_trajectories] wrote {out}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figure_s2_trajectories()
    figure_w_sensitivity()
    figure_rho_sensitivity()
    figure_s3_trajectories()
    print("\nAll figures written to", FIG_DIR)


if __name__ == "__main__":
    main()
