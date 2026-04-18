#!/bin/bash
#SBATCH --job-name=phase_cpd_collect
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --output=./logs/%j_%x.out
#SBATCH --error=./logs/%j_%x.err
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --requeue

mkdir -p ./logs

singularity exec --nv \
    --overlay ${OVERLAY_PATH:-/scratch/kk6081/overlay-25GB-500K.ext3}:rw \
    ${SIF_PATH:-/scratch/kk6081/ubuntu-20.04.3.sif} \
    /bin/bash -c "
source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:\$PATH
conda activate llmr
set -euo pipefail

export PROJECT_DIR=\${PROJECT_DIR:-/scratch/kk6081/nyu-llm-reasoners-a3}
export PROMPTS_FILE=\${PROMPTS_FILE:-\$PROJECT_DIR/phase_cpd/data/prompts/research_prompts.jsonl}
export RAW_TRACE_ROOT=\${RAW_TRACE_ROOT:-\$PROJECT_DIR/outputs/phase_cpd_raw}
export TRACE_OUTPUT_DIR=\${TRACE_OUTPUT_DIR:-\$PROJECT_DIR/phase_cpd/data/traces_real}
export UV_PROJECT_ENVIRONMENT=\${UV_PROJECT_ENVIRONMENT:-\$PROJECT_DIR/.venv-phase-cpd-dream}
export UV_CACHE_DIR=\${UV_CACHE_DIR:-\$PROJECT_DIR/.uv-cache}
export UV_PYTHON=\${UV_PYTHON:-3.11}
export DREAM_MODEL_NAME=\${DREAM_MODEL_NAME:-Dream-org/Dream-v0-Instruct-7B}
export DREAM_MAX_NEW_TOKENS=\${DREAM_MAX_NEW_TOKENS:-256}
export DREAM_STEPS=\${DREAM_STEPS:-256}
export DREAM_TEMPERATURE=\${DREAM_TEMPERATURE:-0.2}
export DREAM_TOP_P=\${DREAM_TOP_P:-0.95}
export DREAM_ALG=\${DREAM_ALG:-entropy}
export DREAM_ALG_TEMP=\${DREAM_ALG_TEMP:-0.0}
export DREAM_TORCH_DTYPE=\${DREAM_TORCH_DTYPE:-bfloat16}
export HF_HOME=\${HF_HOME:-/scratch/kk6081/.cache/huggingface}
export DREAM_TRACE_CMD=\${DREAM_TRACE_CMD:-uv run --group phase_cpd_dream --python \"\$UV_PYTHON\" python -m phase_cpd.trace_jobs.run_dream_trace_dump --prompts \"\$PROMPTS_FILE\" --output-dir \"\$RAW_TRACE_ROOT/dream\" --model-name \"\$DREAM_MODEL_NAME\" --max-new-tokens \"\$DREAM_MAX_NEW_TOKENS\" --steps \"\$DREAM_STEPS\" --temperature \"\$DREAM_TEMPERATURE\" --top-p \"\$DREAM_TOP_P\" --alg \"\$DREAM_ALG\" --alg-temp \"\$DREAM_ALG_TEMP\" --torch-dtype \"\$DREAM_TORCH_DTYPE\"}
export PYTORCH_CUDA_ALLOC_CONF=\${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

cd \$PROJECT_DIR
mkdir -p ./logs \$RAW_TRACE_ROOT/dream \$TRACE_OUTPUT_DIR \$HF_HOME \$UV_CACHE_DIR

echo \"PROMPTS_FILE=\$PROMPTS_FILE\"
echo \"RAW_TRACE_ROOT=\$RAW_TRACE_ROOT\"
echo \"TRACE_OUTPUT_DIR=\$TRACE_OUTPUT_DIR\"
echo \"UV_PROJECT_ENVIRONMENT=\$UV_PROJECT_ENVIRONMENT\"
echo \"UV_CACHE_DIR=\$UV_CACHE_DIR\"
echo \"UV_PYTHON=\$UV_PYTHON\"
echo \"DREAM_MODEL_NAME=\$DREAM_MODEL_NAME\"
echo \"DREAM_STEPS=\$DREAM_STEPS\"
echo \"DREAM_MAX_NEW_TOKENS=\$DREAM_MAX_NEW_TOKENS\"
echo \"HF_HOME=\$HF_HOME\"

# DREAM_TRACE_CMD should write one raw JSON file per prompt into:
#   \$RAW_TRACE_ROOT/dream
#
# Expected raw JSON shape per file:
# {
#   \"trace_id\": \"optional-id\",
#   \"prompt\": \"...\",
#   \"model_name\": \"dream-7b\",
#   \"decoding_metadata\": {\"run_id\": \"...\", \"chunk_size\": 4, \"refinement_steps\": 8},
#   \"steps\": [
#     {
#       \"step_index\": 0,
#       \"tokens\": [
#         {\"token_index\": 0, \"token_text\": \"Hello\", \"top1_prob\": 0.62, \"selected_logit\": 2.1},
#         ...
#       ]
#     },
#     ...
#   ]
# }

if ! command -v uv >/dev/null 2>&1; then
  echo \"uv is not available in the current environment.\"
  echo \"Install uv once in your base container or conda env, then resubmit.\"
  exit 1
fi

echo \"Creating or updating Dream uv environment\"
uv sync --group phase_cpd_dream --python \"\$UV_PYTHON\"

echo \"Running Dream trace generation\"
eval \"\$DREAM_TRACE_CMD\"
uv run --group phase_cpd_dream --python \"\$UV_PYTHON\" \
    python -m phase_cpd.collect_traces \
    --backend dream \
    --source \$RAW_TRACE_ROOT/dream \
    --output-dir \$TRACE_OUTPUT_DIR
"
