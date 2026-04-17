# Distributed Multi-Agent Trajectory Planning

Comparing **Consensus ADMM** and **distributed penalty methods** for multi-drone
trajectory planning with collision avoidance, using a centralized QP as the
optimal reference.

Course project for Optimization (Spring 2026). Team: Christian Yu, Yuyang Wu,
Yen-Ru Chin.

---

## Setup (first-time, per teammate)

### 1. Install `uv`

`uv` is a fast, reproducible Python package manager. It installs Python itself
and locks dependencies exactly, so everyone on the team runs identical code.

**Google Colab:**
Since this repository is structured around `uv` for local virtual environment management and uses a `src/` layout, it requires a specific setup to work with Google Colab's persistent global environment. 

To run the project in Colab, place the following two cells at the very top of your notebook.

#### 1. Fetch the repository

```python
!git clone https://github.com/chrisjinyu/distributed-trajectory-planning.git
%cd distributed-trajectory-planning

# Install uv in Colab
!pip install uv

# Use uv to install the pyproject.toml dependencies directly into Colab's global system environment. 
# The '-e .' flag performs an editable install, which tells Colab where to find the src/dtp module.
!uv pip install --system -e .
```

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify:
```bash
uv --version
```

### 2. Clone and sync

```bash
git clone <repo-url> distributed-trajectory-planning
cd distributed-trajectory-planning
uv sync
```

That's it. `uv sync` reads `pyproject.toml` and `uv.lock`, fetches the exact
pinned versions, installs Python 3.11 if you don't have it, and creates
`.venv/` in the repo. No `conda activate`, no manual pip installs.

### 3. Verify the environment works

```bash
uv run pytest
```

You should see the smoke test pass. This confirms CVXPY, OSQP, and Clarabel
are all wired up correctly.

### 4. (Optional) Register the Jupyter kernel

If you want to use notebooks in VS Code or JupyterLab:
```bash
uv run python -m ipykernel install --user --name dtp --display-name "Python (dtp)"
```

Then launch:
```bash
uv run jupyter lab
```

---

## Running things

Prefix any command with `uv run` and it will execute inside the project's
virtual environment. No activation needed.

```bash
uv run pytest                         # run all tests
uv run pytest tests/test_smoke.py -v  # run one file
uv run ruff check .                   # lint
uv run ruff format .                  # format
uv run python -m dtp.admm             # run a module
```

---

## Adding a dependency

**Don't** use `pip install`. Use `uv`:

```bash
uv add somepackage              # adds to [project.dependencies]
uv add --dev mypytest-plugin    # adds to [dev] optional deps
```

This updates `pyproject.toml` *and* `uv.lock`. Commit both files.

---

## Repository layout

```
distributed-trajectory-planning/
├── pyproject.toml           # Dependencies + tool config
├── uv.lock                  # Exact pinned versions (commit this!)
├── .python-version          # Pinned Python version
├── src/dtp/                 # Main package
│   ├── dynamics.py          # Double-integrator model, A/B matrices
│   ├── mpc.py               # Per-agent MPC QP formulation
│   ├── admm.py              # Consensus ADMM (Christian)
│   ├── penalty.py           # Distributed penalty baseline (Yen-Ru)
│   ├── centralized.py       # Joint QP reference (Yuyang)
│   └── utils.py             # Shared helpers
├── tests/                   # Unit tests + smoke test
├── notebooks/               # Exploration (strip outputs before commit)
├── experiments/configs/     # Scenario configs (YAML/JSON)
└── results/                 # Figures, logs (gitignored)
```

---

## Division of labor

| Area | Owner |
|------|-------|
| Consensus ADMM, convergence analysis, sensitivity experiments | Christian |
| Drone dynamics + MPC formulation, centralized baseline, trajectory visualization | Yuyang |
| Penalty method baseline, experiment design, report writing | Yen-Ru |

Everyone contributes to the literature review, presentation, and final report.

---

## Conventions

- **Math symbols in code**: we use capitalized single letters when they match
  the paper (`A`, `B`, `Q`, `R`, `H`). Ruff is configured to allow this.
- **Notebooks**: strip outputs before committing. Run
  `uv run jupyter nbconvert --clear-output --inplace notebooks/*.ipynb`
  or set up `nbstripout` locally.
- **Branching**: one branch per feature, PRs for anything non-trivial. The
  three algorithm implementations should stay independent so we can diff
  behavior cleanly.
