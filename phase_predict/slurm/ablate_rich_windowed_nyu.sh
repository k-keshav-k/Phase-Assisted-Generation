#!/bin/bash
#SBATCH --job-name=pp_ablate_rich
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/%u/Phase-Assisted-Generation/logs/pp_ablate_%j.out
#SBATCH --error=/scratch/%u/Phase-Assisted-Generation/logs/pp_ablate_%j.err

module purge
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /scratch/$USER/Phase-Assisted-Generation
mkdir -p logs output/ablations_rich_win

uv run --group phase_predict python scripts/ablate_phase_predict.py \
    --train-jsonl traces/rich/stab_tuples_conf_train_rich.jsonl \
    --test-jsonl  traces/rich/stab_tuples_conf_test_rich.jsonl \
    --output-dir  output/ablations_rich_win \
    --preset      small \
    --epochs      60 \
    --dataset-mode windowed \
    --window-size 32 \
    --min-history 1 \
    --input-features block_size nfe mean_top1_confidence min_top1_confidence digit_fraction delimiter_fraction mean_gap max_gap mean_stab_step max_stab_step mean_ref_step max_ref_step
