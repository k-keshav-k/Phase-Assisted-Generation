#!/bin/bash
#SBATCH --job-name=adablock-conf-probe
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --requeue
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append

# Submit from the repo root: sbatch scripts/slurm/slurm_probe_adablock_conf.sh
# --requeue + file-based resume in the probe means preempted jobs restart safely.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

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
    --gen-length 512 \
    --init-block-length 16 \
    --delimiter-threshold 0.3 \
    --threshold 0.9 \
    --limit 5000

uv run python scripts/probe_adablock_llada_conf.py \
    --gsm8k \
    --gsm8k-split test \
    --output-dir traces/adablock \
    --gen-length 512 \
    --init-block-length 16 \
    --delimiter-threshold 0.3 \
    --threshold 0.9 \
    --limit 200

echo "======================================================"
echo "Finished  : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "======================================================"
