#!/bin/bash
# Cost-aware eval pass over the top-N checkpoints from an ablation sweep.
#
# Reports block_size top-1/+/-1 acc, nfe MAE/RMSE, fraction of blocks that
# under-allocate NFE, and the headline metric: per-problem total NFE
# delta vs the AdaBlock baseline schedule recorded in the same JSONL.
#
# Defaults assume the ablation sweep landed in output/ablations_rich_win.
# Override via env vars:
#   ABLATE_DIR=output/ablations_rich_med TOP_N=5 EVAL_MODE=rollout \
#     sbatch phase_predict/slurm/eval_rich_windowed_nyu.sbatch
#
#SBATCH --job-name=pp_eval_rich
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=00:45:00
#SBATCH --requeue
#SBATCH --output=./logs/%j_%x.out
#SBATCH --error=./logs/%j_%x.err

set -euo pipefail
module purge

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/scratch/$USER/.uv-cache}

# --- project location ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROJECT_DIR=${PROJECT_DIR:-$DEFAULT_PROJECT_DIR}
cd "$PROJECT_DIR"

# --- experiment configuration ---
TEST_JSONL=${TEST_JSONL:-$PROJECT_DIR/traces/rich/stab_tuples_conf_test_rich.jsonl}
ABLATE_DIR=${ABLATE_DIR:-$PROJECT_DIR/output/ablations_rich_win}
TOP_N=${TOP_N:-3}
EVAL_MODE=${EVAL_MODE:-teacher_forced}
INPUT_FEATURES=${INPUT_FEATURES:-"block_size nfe mean_top1_confidence min_top1_confidence digit_fraction delimiter_fraction mean_gap max_gap mean_stab_step max_stab_step mean_ref_step max_ref_step"}
REPORT_DIR=${REPORT_DIR:-$ABLATE_DIR/eval_reports}

mkdir -p ./logs "$REPORT_DIR"

if [ ! -f "$TEST_JSONL" ]; then
  echo "ERROR: missing test JSONL: $TEST_JSONL"
  exit 1
fi
if [ ! -f "$ABLATE_DIR/ablation_results.csv" ]; then
  echo "ERROR: missing ablation results CSV: $ABLATE_DIR/ablation_results.csv"
  echo "Run the ablation sweep first, then resubmit this eval job."
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not available in PATH."
  exit 1
fi

echo "=== Environment ==="
echo "PROJECT_DIR=$PROJECT_DIR"
echo "ABLATE_DIR=$ABLATE_DIR"
echo "TEST_JSONL=$TEST_JSONL"
echo "TOP_N=$TOP_N   EVAL_MODE=$EVAL_MODE"
echo "REPORT_DIR=$REPORT_DIR"

uv sync --group phase_predict

echo "=== Picking top-$TOP_N checkpoints by val_loss (CSV col 9) ==="
# ablation_results.csv columns:
# run_id,window_size,d_model,n_heads,n_layers,dropout,learning_rate,train_loss,val_loss,best_epoch,training_time_sec,checkpoint_path
TOP_CKPTS=$(awk -F, 'NR>1 {print $9, $12}' "$ABLATE_DIR/ablation_results.csv" | sort -g | head -n "$TOP_N" | awk '{print $2}')

if [ -z "$TOP_CKPTS" ]; then
  echo "ERROR: could not parse top checkpoints from $ABLATE_DIR/ablation_results.csv"
  exit 1
fi
echo "$TOP_CKPTS"

i=0
for ckpt in $TOP_CKPTS; do
  i=$((i+1))
  echo ""
  echo "=== [$i/$TOP_N] Evaluating: $ckpt (mode=$EVAL_MODE) ==="
  base=$(basename "${ckpt%.pt}")
  # shellcheck disable=SC2086
  uv run --group phase_predict python scripts/eval_phase_predict.py \
      --checkpoint "$ckpt" \
      --jsonl      "$TEST_JSONL" \
      --features   $INPUT_FEATURES \
      --mode       "$EVAL_MODE" \
      --report-json "$REPORT_DIR/${base}_${EVAL_MODE}.json"
done

echo ""
echo "=== Eval reports written to $REPORT_DIR ==="
ls -1 "$REPORT_DIR" | head -20
