#!/bin/bash
#SBATCH --job-name=adablock-probe-train
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --requeue
#SBATCH --output=logs/adablock_probe_train_%j.out
#SBATCH --error=logs/adablock_probe_train_%j.err
#SBATCH --open-mode=append

# --requeue: if preempted the job re-enters the queue automatically.
# The probe has file-based resume logic (skips already-written sample_ids)
# so each requeued run picks up exactly where the previous run stopped.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p logs traces/adablock

echo "======================================================"
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $(hostname)"
echo "Started   : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "======================================================"

uv run python scripts/probe_adablock_llada.py \
    --gsm8k \
    --gsm8k-split train \
    --output-dir traces/adablock \
    --gen-length 256 \
    --init-block-length 16 \
    --delimiter-threshold 0.3 \
    --threshold 0.9 \
    --limit 5000

echo "======================================================"
echo "Finished  : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "======================================================"
