#!/bin/bash
#SBATCH --job-name=phase_cpd_collect
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c24m170-a100-2
#SBATCH --output=./logs/%j_%x.out
#SBATCH --error=./logs/%j_%x.err
#SBATCH --time=24:00:00
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
export BACKENDS=\${BACKENDS:-dream llada}
export DREAM_MODEL_NAME=\${DREAM_MODEL_NAME:-dream-7b}
export LLADA_MODEL_NAME=\${LLADA_MODEL_NAME:-llada}
export PYTORCH_CUDA_ALLOC_CONF=\${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

cd \$PROJECT_DIR
mkdir -p ./logs \$RAW_TRACE_ROOT/dream \$RAW_TRACE_ROOT/llada \$TRACE_OUTPUT_DIR

echo \"PROMPTS_FILE=\$PROMPTS_FILE\"
echo \"RAW_TRACE_ROOT=\$RAW_TRACE_ROOT\"
echo \"TRACE_OUTPUT_DIR=\$TRACE_OUTPUT_DIR\"
echo \"BACKENDS=\$BACKENDS\"

# DREAM_TRACE_CMD and LLADA_TRACE_CMD should each write one raw JSON file per prompt into:
#   \$RAW_TRACE_ROOT/dream
#   \$RAW_TRACE_ROOT/llada
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

if [[ \" \$BACKENDS \" == *\" dream \"* ]]; then
  : \"\${DREAM_TRACE_CMD:?Set DREAM_TRACE_CMD to an instrumented Dream command that writes raw JSON traces into \$RAW_TRACE_ROOT/dream}\"
  echo \"Running Dream trace generation\"
  eval \"\$DREAM_TRACE_CMD\"
  python phase_cpd/collect_traces.py \
      --backend dream \
      --source \$RAW_TRACE_ROOT/dream \
      --output-dir \$TRACE_OUTPUT_DIR
fi

if [[ \" \$BACKENDS \" == *\" llada \"* ]]; then
  : \"\${LLADA_TRACE_CMD:?Set LLADA_TRACE_CMD to an instrumented LLaDA command that writes raw JSON traces into \$RAW_TRACE_ROOT/llada}\"
  echo \"Running LLaDA trace generation\"
  eval \"\$LLADA_TRACE_CMD\"
  python phase_cpd/collect_traces.py \
      --backend llada \
      --source \$RAW_TRACE_ROOT/llada \
      --output-dir \$TRACE_OUTPUT_DIR
fi
"
