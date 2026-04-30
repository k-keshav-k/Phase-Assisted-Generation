#!/bin/bash
#SBATCH --job-name=adablock-conf-probe
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c24m170-a100-2
#SBATCH --gres=gpu:2
#SBATCH --time=23:59:00
#SBATCH --requeue
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append

# Can be submitted from any directory:
#   sbatch scripts/slurm/slurm_probe_adablock_conf.sh
# --requeue + file-based resume in the probe means preempted jobs restart safely.

set -euo pipefail

# Always cd to the repo root regardless of where sbatch was called from
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p traces/adablock

echo "======================================================"
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $(hostname)"
echo "Dir       : $SLURM_SUBMIT_DIR"
echo "Started   : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "======================================================"

uv run python scripts/probe_adablock_llada_conf.py \
    --gsm8k \
    --gsm8k-split train \
    --output-dir traces/adablock \
    --output-file gsm8k_train_conf_traces.jsonl \
    --gen-length 512 \
    --init-block-length 16 \
    --delimiter-threshold 0.3 \
    --threshold 0.9 \
    --limit 5000

uv run python scripts/probe_adablock_llada_conf.py \
    --gsm8k \
    --gsm8k-split test \
    --output-dir traces/adablock \
    --output-file gsm8k_test_conf_traces.jsonl \
    --gen-length 512 \
    --init-block-length 16 \
    --delimiter-threshold 0.3 \
    --threshold 0.9 \
    --limit 200

echo "======================================================"
echo "Finished  : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "======================================================"
