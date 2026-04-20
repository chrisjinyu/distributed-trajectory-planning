#!/usr/bin/env bash
# Render side-by-side drone-flight MP4s for all three benchmark scenarios
# (S1, S2, S3) into results/videos/. Uses the experiments/animate.py CLI.
#
# Usage (from repo root):
#   ./scripts/generate_videos.sh
#
# Requires ffmpeg on PATH. Requires `uv sync` already run.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="results/videos"
mkdir -p "$OUT_DIR"

CONFIGS="experiments/configs"
OPTS="$CONFIGS/optimizers"

echo "==> S1 (two-drone swap, N=2)"
uv run python -m experiments.animate \
    --scenario "$CONFIGS/two_drone_swap.yaml" \
    --optimizer "$OPTS/penalty_default.yaml" \
    --optimizer "$OPTS/admm_default.yaml" \
    --optimizer "$OPTS/centralized_default.yaml" \
    --output "$OUT_DIR/s1_comparison.mp4" \
    --no-markers

echo ""
echo "==> S2 (four-drone ring, N=4)"
uv run python -m experiments.animate \
    --scenario "$CONFIGS/four_drone_ring.yaml" \
    --optimizer "$OPTS/penalty_default.yaml" \
    --optimizer "$OPTS/admm_default.yaml" \
    --optimizer "$OPTS/centralized_default.yaml" \
    --output "$OUT_DIR/s2_comparison.mp4" \
    --no-markers

echo ""
echo "==> S3 (eight-drone complete, N=8) -- ADMM uses tuned rho/w_coll + penalty seed"
uv run python -m experiments.animate \
    --scenario "$CONFIGS/eight_drone_complete.yaml" \
    --optimizer "$OPTS/penalty_s3.yaml" \
    --optimizer "$OPTS/admm_s3.yaml" \
    --optimizer "$OPTS/centralized_default.yaml" \
    --output "$OUT_DIR/s3_comparison.mp4" \
    --no-markers

echo ""
echo "Done. Videos in $OUT_DIR/:"
ls -la "$OUT_DIR"/*.mp4
