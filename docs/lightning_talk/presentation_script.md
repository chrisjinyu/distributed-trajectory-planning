# Lightning Talk Script — 2026-04-23

Target length: **5 minutes strict**.
Team: Christian Yu, Yuyang Wu, Yen-Ru Chin.

Roughly ~130 words per minute at presentation pace. Budget below is the
target duration per slide; it leaves ~30 s of cushion for transitions
and demoing the trajectory arrows.

---

## Slide 1 — Title  *(≈ 15 s)*

> "Hi, we're Christian, Yuyang, and Yen-Ru. Our project compares two
> distributed optimization methods — **penalty** and **consensus ADMM** —
> on a multi-drone trajectory planning problem, with a centralized solver
> as the ground-truth reference."

## Slide 2 — Background & Motivation  *(≈ 40 s)*

> "The setup: `N` drones, each wants to reach its goal without colliding.
> There is no central coordinator — each drone only talks to its
> neighbors. Every drone solves its own MPC QP: track the goal, bound
> control effort, maintain minimum separation from neighbors. We
> linearize the non-convex collision constraint into a half-space so the
> per-agent problem stays a convex QP.
>
> What differs between the methods is how each one handles the
> **coupling** between drones. Penalty adds a soft cost on violation;
> ADMM maintains dual variables that negotiate feasibility through a
> consensus update. The question is whether ADMM's extra machinery
> actually pays off."

## Slide 3 — Key Question  *(≈ 25 s)*

> "Three concrete sub-questions:
> do both methods reach the centralized optimum?
> how sensitive is each to its tuning knob — `w` for penalty, `ρ` for
> ADMM? and does the comparison still hold as we scale from 4 drones to
> 8?"

## Slide 4 — Take-home 1: all three reach goals, only penalty collides  *(≈ 45 s)*

> "S2: four drones swap antipodally on a ring communication graph.
> The three panels show penalty, ADMM, and centralized trajectories —
> the small arrows mark time direction. All three methods reach the
> goals and agree on objective within 0.04 percent.
>
> But look at the minimum pairwise distance. Penalty comes in at
> **0.483 meters — below the 0.5 m safety threshold**. That is a
> real collision, just absorbed silently by the quadratic slack cost.
> ADMM and centralized both respect the boundary exactly."

## Slide 5 — Take-home 2: penalty has a U-shape in `w`; ADMM doesn't  *(≈ 50 s)*

> "Now the sensitivity sweep. I ran the penalty method on S2 for eight
> values of `w` between 10² and 10⁴.
>
> Left plot: violation versus `w`.
> Right plot: iterations versus `w`.
>
> Small `w` — the collision cost is too weak, drones overlap by almost
> half a meter. Large `w` — the collision cost dominates the QP, the
> solver can't converge in 500 iterations, and we *still* get 24 cm of
> overlap. There's a narrow sweet spot around `w ≈ 10³`.
>
> ADMM doesn't have this tradeoff. The dual variable
> `λ_ij` is an **integrator** — it adds up consensus disagreements over
> iterations and keeps pushing until feasibility holds. That means
> `ρ` is free to focus on QP conditioning, and `λ` handles feasibility.
> Two separate levers, each doing one job."

## Slide 6 — Take-home 3: gap widens at N = 8  *(≈ 50 s)*

> "Does the story hold at 8 drones? Yes — more dramatically.
> With the complete communication graph, penalty leaves a 33-centimeter
> overlap. The drones literally intersect. ADMM matches centralized's
> objective to within 0.03 percent and achieves feasibility to solver
> tolerance. Note that ADMM did need a penalty-seeded warm-start here —
> the straight-line initialization is degenerate at N = 8 — but once
> seeded, it converges cleanly. The feasibility gap between penalty and
> ADMM grew roughly 20× from S2 to S3."

## Slide 7 — When to use which  *(≈ 35 s)*

> "Summary table. Penalty is the simplest, with just one knob — but
> that knob has to trade off feasibility against conditioning, which
> is the U-shape we just saw. ADMM costs a bit more code, but converges
> to the centralized optimum and scales distributedly in O(N) parallel
> work per iteration. Centralized is the ground truth, but a single
> joint QP of size O(N²) doesn't scale.
>
> Bottom line: **ADMM's dual-variable feedback pays off exactly where
> penalty's single `w` knob runs out of room — and that gap grows with
> the number of coupled agents.**"

## Slide 8 — Thank you  *(≈ 15 s, or drop if short on time)*

> "Code, configs, and animated comparison videos are on GitHub. Happy
> to take questions."

---

## Delivery notes

- **Cadence.** Slow down on slides 4, 5, 6 — let the numbers land.
  Speed up on slides 2, 3, 7.
- **Pointers.** On slide 4, point at the red "0.483" in the table as
  you say "below the safety threshold." On slide 5, physically trace
  the U-shape with your hand while saying "sweet spot around 10³".
- **Backup video.** If laptop/projector cooperates, play
  `results/videos/s3_comparison.mp4` silently while narrating slide 6 —
  drones colliding in the penalty panel sells the point instantly.
- **If running long.** Drop slide 8 and end on slide 7's thesis
  sentence — "the gap grows with the number of coupled agents."
- **If asked in Q&A about `w_coll`.** Not a feasibility knob — it's a
  numerical safety margin inside each ADMM subproblem so infeasible
  early linearizations don't crash the solver. We set it once to 10⁷
  and forget it. The feasibility work is done by the dual `λ`.

## Speaker hand-offs (if splitting three ways)

One natural division for ~5 min:
- **Christian** (≈1:45): slides 1–3 (title + background + question),
  and slide 6 (ADMM scalability + tuning story).
- **Yuyang** (≈0:45): slide 7 (summary table + thesis), short because
  the table does the talking.
- **Yen-Ru** (≈2:15): slides 4 and 5 — the penalty results and the
  w-sensitivity sweep she owns.

If one person is presenting solo, just read the whole script through.
