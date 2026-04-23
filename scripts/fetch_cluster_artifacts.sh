#!/usr/bin/env bash
# Copy artifacts from the cluster into this local repo.
# Includes: NSGA outputs, baselines CSV/JSON, Plotly HTML plots, job logs, optional prediction cache.
#
# Default (no model weights, no venv):
#   results/   — nsga2 npy, summary, analysis/ (pareto_*.html, knee_*.html, knee_point.json, CSV, …)
#   logs/      — slurm .out / .err for jobs you ran
#
# Usage:
#   ./scripts/fetch_cluster_artifacts.sh
#   ./scripts/fetch_cluster_artifacts.sh user@host
#   ./scripts/fetch_cluster_artifacts.sh --with-cache
#   ./scripts/fetch_cluster_artifacts.sh user@host --with-cache
set -euo pipefail

REMOTE="dhish_s@10.13.1.162"
WITH_CACHE=0
for arg in "$@"; do
  case "$arg" in
    --with-cache) WITH_CACHE=1 ;;
    -h|--help)
      echo "Usage: $0 [user@host] [--with-cache]"
      exit 0
      ;;
    *@*) REMOTE="$arg" ;;
  esac
done

REMOTE_ROOT="${REMOTE_ROOT:-~/nsga2_medical_ensemble}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

mkdir -p "$LOCAL_ROOT/results" "$LOCAL_ROOT/logs"
if [[ "$WITH_CACHE" -eq 1 ]]; then
  mkdir -p "$LOCAL_ROOT/data/cache"
fi

echo "Remote: $REMOTE:$REMOTE_ROOT"
echo "Local:  $LOCAL_ROOT"
echo ""

rsync -avz --progress \
  "${REMOTE}:${REMOTE_ROOT}/results/" \
  "${LOCAL_ROOT}/results/"

rsync -avz --progress \
  "${REMOTE}:${REMOTE_ROOT}/logs/" \
  "${LOCAL_ROOT}/logs/"

if [[ "$WITH_CACHE" -eq 1 ]]; then
  echo ""
  echo "Syncing data/cache/ (P_cache.npy, labels, etc.) — can be large …"
  rsync -avz --progress \
    "${REMOTE}:${REMOTE_ROOT}/data/cache/" \
    "${LOCAL_ROOT}/data/cache/"
fi

echo ""
echo "Done."
echo "  Plots (open .html in a browser): $LOCAL_ROOT/results/analysis/"
if [[ "$WITH_CACHE" -eq 0 ]]; then
  echo "  Re-run with --with-cache to also copy data/cache/ (enables local re-runs of validation scripts)."
fi
