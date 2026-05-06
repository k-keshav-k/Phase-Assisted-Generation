#!/bin/bash
#SBATCH --job-name=pp_eval_rich
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=00:45:00
#SBATCH --requeue
#SBATCH --output=./logs/pp_eval_%j.out
#SBATCH --error=./logs/pp_eval_%j.err

module purge
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /scratch/$USER/Phase-Assisted-Generation
mkdir -p logs output/ablations_rich_win/eval_reports

ABLATE_DIR=output/ablations_rich_win
TEST_JSONL=traces/rich/stab_tuples_conf_test_rich.jsonl
EVAL_MODE=teacher_forced
TOP_N=3

# Pick the TOP_N checkpoints with the lowest val_loss from the ablation CSV.
# CSV columns: run_id, window_size, d_model, n_heads, n_layers, dropout,
#              learning_rate, train_loss, val_loss, best_epoch,
#              training_time_sec, checkpoint_path
TOP_CKPTS=$(awk -F, 'NR>1 {print $9, $12}' "$ABLATE_DIR/ablation_results.csv" | sort -g | head -n "$TOP_N" | awk '{print $2}')

for ckpt in $TOP_CKPTS; do
    base=$(basename "${ckpt%.pt}")
    uv run --group phase_predict python scripts/eval_phase_predict.py \
        --checkpoint "$ckpt" \
        --jsonl      "$TEST_JSONL" \
        --features   block_size nfe mean_top1_confidence min_top1_confidence digit_fraction delimiter_fraction mean_gap max_gap mean_stab_step max_stab_step mean_ref_step max_ref_step \
        --mode       "$EVAL_MODE" \
        --report-json "$ABLATE_DIR/eval_reports/${base}_${EVAL_MODE}.json"
done
