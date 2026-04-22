"""Render drone-flight animation videos from scenario + optimizer configs.

Library API: :func:`animate_comparison`. CLI: ``python -m experiments.animate``.

Loads a scenario YAML (via :func:`experiments.runner.load_scenario`), runs one
or more optimizer recipes specified by optimizer-config YAMLs, and writes a
side-by-side MP4 with one panel per optimizer. Each panel shows:

    - colored dots at the current drone positions,
    - a trailing line from t=0 up to the current frame,
    - persistent start squares and goal stars,
    - dashed d_min/2 safety circles around each drone,
    - a scenario-level time counter (t = k*dt s).

MP4 only (no GIF fallback) per Yen-Ru's preference. Requires ffmpeg on PATH.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.animation as mpl_animation
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Circle

from dtp.scenario import Scenario
from experiments.runner import (
    MethodResult,
    load_scenario,
    run_admm,
    run_centralized,
    run_penalty,
)

# ---------------------------------------------------------------------------
# Solver dispatch
# ---------------------------------------------------------------------------


@dataclass
class OptimizerConfig:
    method: str
    params: dict[str, Any]
    # Either key seeds this optimizer from an earlier optimizer's result
    # in the same CLI run. `seed_from_method` is the unified name; the
    # older `linearization_from_method` is kept as an alias for configs
    # written before the unification.
    seed_from_method: str | None = None


def _load_optimizer_config(path: str | Path) -> OptimizerConfig:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    seed = cfg.get("seed_from_method") or cfg.get("linearization_from_method")
    return OptimizerConfig(
        method=cfg["method"],
        params=cfg.get("params", {}) or {},
        seed_from_method=seed,
    )


def _seed_x_from(prior: list[MethodResult], src: str | None) -> np.ndarray | None:
    if src is None:
        return None
    src_lower = src.lower()
    match = next((r.x for r in prior if r.name.lower().startswith(src_lower)), None)
    if match is None:
        print(
            f"[warn] requested seed_from_method={src!r} but no prior result "
            f"matches; falling back to built-in warm-start.",
            file=sys.stderr,
        )
    return match


def _run_optimizer(
    scenario: Scenario, cfg: OptimizerConfig, prior: list[MethodResult]
) -> MethodResult:
    if cfg.method == "penalty":
        return run_penalty(scenario, **cfg.params)
    if cfg.method == "admm":
        seed = _seed_x_from(prior, cfg.seed_from_method)
        kwargs = dict(cfg.params)
        if seed is not None:
            kwargs["initial_guess"] = seed
        return run_admm(scenario, **kwargs)
    if cfg.method == "centralized":
        seed = _seed_x_from(prior, cfg.seed_from_method)
        return run_centralized(scenario, linearization_from=seed, **cfg.params)
    raise ValueError(f"unknown optimizer method: {cfg.method!r}")


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------


def _axis_limits(results: list[MethodResult], margin: float = 0.5) -> tuple[float, float, float, float]:
    xs = np.concatenate([r.x[:, :, 0].ravel() for r in results])
    ys = np.concatenate([r.x[:, :, 1].ravel() for r in results])
    return xs.min() - margin, xs.max() + margin, ys.min() - margin, ys.max() + margin


def _min_pairwise_distance_over_time(x: np.ndarray, N: int) -> np.ndarray:
    """Return (T,) array of minimum pairwise distances at each time step."""
    T = x.shape[1]
    mind = np.full(T, np.inf)
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(x[i, :, 0:2] - x[j, :, 0:2], axis=1)
            mind = np.minimum(mind, d)
    return mind


def animate_comparison(
    scenario: Scenario,
    results: list[MethodResult],
    filename: str | Path,
    fps: int = 3,
    show_safety_circles: bool = True,
    show_time: bool = True,
    show_markers: bool = True,
    dpi: int = 110,
    scenario_name: str | None = None,
) -> None:
    """Render a side-by-side MP4 animation of drone flights.

    One panel per ``MethodResult`` in ``results``, drawn left-to-right in the
    given order. Each frame advances all panels by one time step; frames run
    from k=0 to k=H inclusive.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not on PATH -- install via 'sudo dnf install ffmpeg' "
            "(Fedora) or 'conda install -c conda-forge ffmpeg'."
        )
    if not results:
        raise ValueError("results must contain at least one MethodResult")

    N = scenario.N
    H = scenario.H
    dt = scenario.dt
    d_min = scenario.d_min
    n_panels = len(results)

    colors = plt.cm.tab10(np.arange(N) % 10)
    xmin, xmax, ymin, ymax = _axis_limits(results)

    # Precompute per-result min pairwise distance trace so we can draw a
    # stable time series under each trajectory panel.
    mind_traces = [_min_pairwise_distance_over_time(r.x, N) for r in results]
    mind_ylo = min(float(mind.min()) for mind in mind_traces)
    mind_yhi = max(float(mind.max()) for mind in mind_traces)
    # Pad so d_min reference line has breathing room on both sides.
    mind_ylo = min(mind_ylo, d_min) - 0.15 * max(mind_yhi - d_min, 0.1)
    mind_yhi = max(mind_yhi, d_min) + 0.15 * max(mind_yhi - d_min, 0.1)
    t_axis = np.arange(H + 1) * dt

    fig = plt.figure(figsize=(5 * n_panels, 6.5), dpi=dpi)
    gs = fig.add_gridspec(2, n_panels, height_ratios=[3, 1], hspace=0.35)
    traj_axes = [fig.add_subplot(gs[0, i]) for i in range(n_panels)]
    ts_axes = [fig.add_subplot(gs[1, i]) for i in range(n_panels)]

    dots_per_panel: list[list] = []
    trails_per_panel: list[list] = []
    circles_per_panel: list[list] = []
    now_lines: list = []  # moving vertical marker in each time-series subplot

    for ax, ts_ax, r, mind in zip(traj_axes, ts_axes, results, mind_traces):
        # --- trajectory panel (top) ---
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(r.name)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

        if show_markers:
            for i in range(N):
                ax.scatter(
                    r.x[i, 0, 0], r.x[i, 0, 1],
                    marker="s", s=70, facecolor=colors[i], edgecolor="k", zorder=4,
                )
                ax.scatter(
                    r.x[i, -1, 0], r.x[i, -1, 1],
                    marker="*", s=140, facecolor=colors[i], edgecolor="k", zorder=4,
                )

        trails = []
        for i in range(N):
            (line,) = ax.plot([], [], "-", color=colors[i], alpha=0.5, linewidth=1.5)
            trails.append(line)
        trails_per_panel.append(trails)

        dots = ax.scatter(
            r.x[:, 0, 0], r.x[:, 0, 1],
            c=colors[:N], s=80, edgecolor="k", zorder=5,
        )
        dots_per_panel.append([dots])

        circles = []
        if show_safety_circles:
            for i in range(N):
                c = Circle(
                    (r.x[i, 0, 0], r.x[i, 0, 1]),
                    radius=d_min / 2,
                    fill=False,
                    linestyle="--",
                    edgecolor=colors[i],
                    alpha=0.6,
                    linewidth=1.0,
                )
                ax.add_patch(c)
                circles.append(c)
        circles_per_panel.append(circles)

        # --- min-pairwise-distance time series (bottom) ---
        ts_ax.plot(t_axis, mind, color="tab:blue", linewidth=1.6)
        ts_ax.axhline(d_min, color="tab:red", linestyle="--", linewidth=1.0, label=f"d_min = {d_min}")
        # Shade violation band (below d_min) for quick visual "uh-oh" cue.
        ts_ax.fill_between(t_axis, mind_ylo, d_min, color="tab:red", alpha=0.07)
        ts_ax.set_xlim(t_axis[0], t_axis[-1])
        ts_ax.set_ylim(mind_ylo, mind_ylo if mind_yhi == mind_ylo else mind_yhi)
        ts_ax.set_xlabel("t (s)")
        ts_ax.set_ylabel("min pairwise d (m)")
        ts_ax.grid(True, alpha=0.3)
        ts_ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
        now_line = ts_ax.axvline(t_axis[0], color="k", alpha=0.7, linewidth=1.2)
        now_lines.append(now_line)

    title = scenario_name or "scenario"
    fig.suptitle(f"{title}  (N={N}, H={H}, d_min={d_min})", fontsize=12)

    time_text = None
    if show_time:
        time_text = fig.text(
            0.98, 0.975, "", ha="right", va="top", fontsize=11,
            family="monospace",
        )

    fig.tight_layout(rect=(0, 0, 1, 0.95))

    def init():
        artists = []
        for dots in dots_per_panel:
            artists.extend(dots)
        for trails in trails_per_panel:
            artists.extend(trails)
        for circles in circles_per_panel:
            artists.extend(circles)
        artists.extend(now_lines)
        if time_text is not None:
            time_text.set_text("")
            artists.append(time_text)
        return artists

    def animate(k: int):
        artists = []
        for r, (dots,), trails, circles in zip(
            results, dots_per_panel, trails_per_panel, circles_per_panel
        ):
            positions = r.x[:, k, 0:2]
            dots.set_offsets(positions)
            artists.append(dots)
            for i, line in enumerate(trails):
                line.set_data(r.x[i, : k + 1, 0], r.x[i, : k + 1, 1])
                artists.append(line)
            for i, c in enumerate(circles):
                c.center = (positions[i, 0], positions[i, 1])
                artists.append(c)
        for now_line in now_lines:
            now_line.set_xdata([k * dt, k * dt])
            artists.append(now_line)
        if time_text is not None:
            time_text.set_text(f"t = {k * dt:.2f} s")
            artists.append(time_text)
        return artists

    ani = mpl_animation.FuncAnimation(
        fig, animate, init_func=init, frames=H + 1, interval=1000 // fps, blit=False
    )

    writer = mpl_animation.FFMpegWriter(fps=fps, bitrate=1800)
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    ani.save(str(filename), writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"[ok] wrote {filename}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m experiments.animate",
        description="Render a side-by-side MP4 of drones flying, one panel per optimizer.",
    )
    p.add_argument("--scenario", required=True, help="path to scenario YAML")
    p.add_argument(
        "--optimizer", required=True, action="append",
        help="path to an optimizer YAML (repeat for more panels)",
    )
    p.add_argument("--output", default=None, help="output MP4 path (default: <scenario>_comparison.mp4)")
    p.add_argument("--fps", type=int, default=3, help="frames per second (default 3)")
    p.add_argument("--no-safety-circles", dest="safety_circles", action="store_false")
    p.add_argument("--no-time", dest="time", action="store_false")
    p.add_argument("--no-markers", dest="markers", action="store_false")
    p.set_defaults(safety_circles=True, time=True, markers=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    scenario = load_scenario(args.scenario)
    with open(args.scenario) as f:
        scenario_name = yaml.safe_load(f).get("name", Path(args.scenario).stem)
    optimizer_cfgs = [_load_optimizer_config(p) for p in args.optimizer]

    results: list[MethodResult] = []
    for cfg in optimizer_cfgs:
        print(f"[run] {cfg.method}  params={cfg.params}")
        res = _run_optimizer(scenario, cfg, prior=results)
        print(
            f"[done] {res.name}: iters={res.iterations}  t={res.wall_time:.2f}s  "
            f"viol={res.max_collision_violation:.3e}  min_d={res.min_pairwise_distance:.3f}"
        )
        results.append(res)

    output = args.output
    if output is None:
        stem = Path(args.scenario).stem
        output = f"{stem}_comparison.mp4"

    animate_comparison(
        scenario,
        results,
        output,
        fps=args.fps,
        show_safety_circles=args.safety_circles,
        show_time=args.time,
        show_markers=args.markers,
        scenario_name=scenario_name,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
