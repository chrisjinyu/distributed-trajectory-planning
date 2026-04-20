# Presentation slide outline — Yen-Ru's portion (Thu 2026-04-23)

Target length: ~5-7 minutes (roughly 7-9 slides). Owner: Yen-Ru Chin.

The intro (problem setup + shared MPC formulation) and ADMM presentation
are Christian and Yuyang's responsibility; this outline only covers the
penalty-method segment Yen-Ru leads.

---

## 1. Title / section header — "Distributed penalty-method baseline"
- One sentence framing: "a simpler distributed solver we compare against
  Consensus ADMM to see what the dual-variable machinery buys us."

## 2. Formulation slide
- The nonconvex collision constraint `||p_i - p_j|| ≥ d_min` and its
  linearization to the half-space `n_ij^T (p_i - p_j) ≥ d_min`.
- Per-agent QP: tracking + input + `w · sum ||s_ij||^2` with slack `s ≥ 0`
  absorbing violation. No consensus variables, no duals.
- Contrast box: ADMM's local QP has `tilde` + dual `u` per edge; penalty
  drops both.

## 3. Algorithm pseudocode
```
initialize theta_i from straight-line warm-start
repeat
    for each agent i (in parallel):
        solve local QP with n_ij, collision_rhs frozen as Parameters
    broadcast theta_i to ring-neighbors
    re-linearize n_ij and collision_rhs = d_min + n_ij^T p_j_hat
until ||theta - theta_prev|| < tol
```
- Key message: each round is N independent convex QPs; no central solver,
  no dual synchronization.

## 4. S2 trajectory comparison (4-drone ring)
- Three side-by-side panels from `notebooks/02_penalty_verification.ipynb`
  (experiment 2 cell): penalty / ADMM / centralized trajectories in the
  xy-plane with start squares and goal stars.
- Animated version in `results/videos/s2_comparison.mp4` — same three
  panels synchronized, with the min-pairwise-distance time series under
  each. Drop the MP4 onto the slide for visual impact.
- Takeaway bullet: all three plans reach goals; penalty tracks tightly to
  ADMM at `w=1e3`.

## 5. Convergence comparison on S2
- Two stacked log-y plots (also from the notebook): primal residual vs
  iteration, and max collision violation vs iteration, overlaying penalty
  and ADMM.
- Takeaway bullet: ADMM's dual updates drive violation below penalty's
  floor; penalty plateaus at the level the slack weight allows.

## 6. w-sensitivity teaser (the proposal's money slide)
- Two small log-x plots from notebook experiment 4:
  (a) max collision violation vs w — U-shape, sweet spot around 10^3
  (b) iterations-to-tolerance vs w — flat-then-rising, showing
      ill-conditioning above 10^3
- Narrate: "small w lets drones collide, large w lets the QP stiffen and
  stalls. ADMM has no such knob — the dual lets rho tune conditioning
  while the consensus equality still drives feasibility."

## 7. Scalability to N=8 (S3)
- Animated side-by-side from `results/videos/s3_comparison.mp4`: penalty
  / ADMM / centralized on the 8-drone antipodal swap.
- Key finding: ADMM needs non-default tuning at N=8. With `rho=300`,
  `collision_slack_weight=1e7`, and a penalty-seeded initial guess,
  ADMM reaches `viol ~ 1e-5` — matching centralized. With default
  `rho=5` it gets trapped at the symmetric warm-start.
- Narrate: at larger N the "dual-updates-will-drive-feasibility" story
  is still true, but the constants need scaling. Penalty with `w=2·10³`
  still converges out of the box — simpler but leaves a feasibility gap.

## 8. When to pick which method
| Criterion | Penalty | ADMM | Centralized |
|-----------|---------|------|-------------|
| Implementation complexity | low | medium | low |
| Per-iteration cost | 1 QP × N | 1 QP × N (bigger) | 1 joint QP |
| Feasibility guarantee | none (w-dependent) | in limit | hard |
| Tuning knobs | w | rho (+ adaptive) | solver only |
| Scales with N | yes (converges everywhere, gap grows) | yes (needs rho/w_coll retune past N=4) | no (N·H variables) |

## 9. Takeaways + what's next
- Penalty = quick-to-ship baseline that quantifies the value of ADMM's
  dual machinery. Matches ADMM closely on objective when `w` is tuned.
- ADMM matches the centralized reference across S1, S2, S3 — once tuned.
- For the final report we'll add a random-init robustness sweep and a
  fair wall-time comparison; deferred past this presentation.

---

## Assets already generated

**Notebook `notebooks/02_penalty_verification.ipynb`:**
- **Slide 4 figure:** "Trajectory + convergence plots for S2" cell.
- **Slide 5 figure:** "Convergence comparison on S2" cell.
- **Slide 6 figures:** "Plot violation vs w and iterations vs w" cell.
- **Slide 7 figures:** S3 trajectory + S3 convergence cells.

**Animated videos in `results/videos/`:**
- `s1_comparison.mp4` — N=2, 3 panels (penalty / ADMM / centralized).
- `s2_comparison.mp4` — N=4 ring, same 3 panels.
- `s3_comparison.mp4` — N=8 complete graph, same 3 panels (ADMM tuned).

Regenerate all three videos from scratch with:
```
./scripts/generate_videos.sh
```

Export static figures from the notebook via right-click → save image, or
run `uv run jupyter nbconvert --to html notebooks/02_penalty_verification.ipynb`
and screenshot from the rendered HTML.

## Things NOT to commit to on the slides
- Exact speedup numbers — wall-times from the MVP are not benchmarked
  carefully.
- Any "penalty method is better" conclusion — the proposal's thesis is
  ADMM's dual machinery is worth the complexity, so stay aligned with
  that.
- Claims of out-of-the-box ADMM at N=8 — acknowledge the tuning step.
