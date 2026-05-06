#!/bin/bash
# Ablation sweep for phase_predict on the rich-features dataset, windowed mode.
#
# Defaults run the "headline" experiment described in the branch plan:
#   * preset            : small  (32 configs after dedup)
#   * dataset mode      : windowed, window_size=32, min_history=1
#   * input features    : all 12 fields in the rich tuples
#   * epochs            : 60 (with early stopping, patience=10)
#   * output dir        : output/ablations_rich_win
#
# Override via env vars at submit time:
#   PRESET=medium EPOCHS=80 WINDOW_SIZE=48 \
#     OUTPUT_DIR=/scratch/$USER/pp_out/ablations_rich_med \
#     sbatch phase_predict/slurm/ablate_rich_windowed_nyu.sbatch
#
# Smoke variant (~5 min, sanity check before the real run):
#   EPOCHS=5 OUTPUT_DIR=/scratch/$USER/pp_out/ablations_rich_smoke \
#     sbatch --time=0:30:00 phase_predict/slurm/ablate_rich_windowed_nyu.sbatch
#
# Wall time on c12m85-a100-1 (1 x A100 40GB):
#   small  preset, 60 epochs : ~2-3 hours
#   medium preset, 80 epochs : ~5-6 hours
#
#SBATCH --job-name=pp_ablate_rich
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
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

# --- experiment configuration (override via env vars on sbatch) ---
TRAIN_JSONL=${TRAIN_JSONL:-$PROJECT_DIR/traces/rich/stab_tuples_conf_train_rich.jsonl}
TEST_JSONL=${TEST_JSONL:-$PROJECT_DIR/traces/rich/stab_tuples_conf_test_rich.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_DIR/output/ablations_rich_win}
PRESET=${PRESET:-small}
EPOCHS=${EPOCHS:-60}
DATASET_MODE=${DATASET_MODE:-windowed}
WINDOW_SIZE=${WINDOW_SIZE:-32}
MIN_HISTORY=${MIN_HISTORY:-1}
# Space-separated list. Quoted so the array form survives the export.
INPUT_FEATURES=${INPUT_FEATURES:-"block_size nfe mean_top1_confidence min_top1_confidence digit_fraction delimiter_fraction mean_gap max_gap mean_stab_step max_stab_step mean_ref_step max_ref_step"}

mkdir -p ./logs "$OUTPUT_DIR"

# --- sanity: input files must exist on the HPC. If only rich.zip was synced,
#     unzip it once before submitting the job.
if [ ! -f "$TRAIN_JSONL" ] || [ ! -f "$TEST_JSONL" ]; then
  echo "ERROR: missing input files"
  echo "  TRAIN_JSONL=$TRAIN_JSONL  exists=$( [ -f "$TRAIN_JSONL" ] && echo yes || echo no )"
  echo "  TEST_JSONL=$TEST_JSONL    exists=$( [ -f "$TEST_JSONL" ] && echo yes || echo no )"
  echo "If only traces/rich.zip is present, run once and resubmit:"
  echo "  unzip -j -o traces/rich.zip 'rich/*.jsonl' -d traces/rich/ -x '__MACOSX/*'"
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not available in PATH. Install with: pipx install uv  (or curl -LsSf https://astral.sh/uv/install.sh | sh)"
  exit 1
fi

echo "=== Environment ==="
echo "PROJECT_DIR=$PROJECT_DIR"
echo "PWD=$(pwd)"
echo "TRAIN_JSONL=$TRAIN_JSONL"
echo "TEST_JSONL=$TEST_JSONL"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "PRESET=$PRESET   EPOCHS=$EPOCHS"
echo "DATASET_MODE=$DATASET_MODE   WINDOW_SIZE=$WINDOW_SIZE   MIN_HISTORY=$MIN_HISTORY"
echo "INPUT_FEATURES=$INPUT_FEATURES"

echo "=== Sync uv environment (phase_predict group) ==="
uv sync --group phase_predict

echo "=== GPU visibility ==="
uv run --group phase_predict python -c 'import torch; print("cuda?", torch.cuda.is_available(), "devices:", torch.cuda.device_count(), "name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)'

echo "=== Launch ablation sweep ==="
# shellcheck disable=SC2086
uv run --group phase_predict python scripts/ablate_phase_predict.py \
    --train-jsonl "$TRAIN_JSONL" \
    --test-jsonl  "$TEST_JSONL" \
    --output-dir  "$OUTPUT_DIR" \
    --preset      "$PRESET" \
    --epochs      "$EPOCHS" \
    --dataset-mode "$DATASET_MODE" \
    --window-size "$WINDOW_SIZE" \
    --min-history "$MIN_HISTORY" \
    --input-features $INPUT_FEATURES

echo "=== Top-3 by val_loss (col 9 in CSV) ==="
sort -t, -k9,9g "$OUTPUT_DIR/ablation_results.csv" 2>/dev/null | head -4 || true

echo "=== Done. Run the eval sbatch next:"
echo "  ABLATE_DIR=$OUTPUT_DIR sbatch phase_predict/slurm/eval_rich_windowed_nyu.sbatch"
