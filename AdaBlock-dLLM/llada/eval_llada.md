# LLaDA Model Evaluation Guide

This document provides instructions for evaluating the LLaDA model using baseline (LLaDA and Fast-dLLM) and AdaBlock-dLLM.

## Environment Setup

Before running any evaluation, set the following environment variables:
```bash
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
```



## Baseline Evaluation (`eval_llada_baseline.py`)

### Common Parameters
```bash
MODEL_PATH="GSAI-ML/LLaDA-8B-Instruct"  # or "GSAI-ML/LLaDA-1.5-8B-Instruct"
GEN_LENGTH=512
BLOCK_LENGTH=32
THRESHOLD=0.9
```

### Methods

#### 1. Top-K Decoding
- `steps = gen_length` (one token per step)

```bash
# GSM8K
accelerate launch --num_processes=1 eval_llada_baseline.py --tasks gsm8k --num_fewshot 5 \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=${GEN_LENGTH},block_length=${BLOCK_LENGTH},show_speed=True

# HumanEval
accelerate launch --num_processes=1 eval_llada_baseline.py --tasks humaneval --num_fewshot 0 \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=${GEN_LENGTH},block_length=${BLOCK_LENGTH},show_speed=True \
--output_path eval_results/humaneval/baseline --log_samples
```

#### 2. Threshold-based Parallel Decoding (Fast-dLLM)
- `steps = gen_length / block_length`
- `threshold=0.9` for parallel decoding

```bash
# GSM8K
accelerate launch --num_processes=1 eval_llada_baseline.py --tasks gsm8k --num_fewshot 5 \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=$((GEN_LENGTH/BLOCK_LENGTH)),block_length=${BLOCK_LENGTH},threshold=${THRESHOLD},show_speed=True

# HumanEval
accelerate launch --num_processes=1 eval_llada_baseline.py --tasks humaneval --num_fewshot 0 \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=$((GEN_LENGTH/BLOCK_LENGTH)),block_length=${BLOCK_LENGTH},threshold=${THRESHOLD},show_speed=True \
--output_path eval_results/humaneval/thres_no_cache --log_samples
```

#### 3. Threshold-based Parallel Decoding + Prefix Cache (Fast-dLLM)
- `steps = gen_length / block_length`
- `threshold=0.9`, `use_cache=True`

```bash
# GSM8K
accelerate launch --num_processes=1 eval_llada_baseline.py --tasks gsm8k --num_fewshot 5 \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=$((GEN_LENGTH/BLOCK_LENGTH)),block_length=${BLOCK_LENGTH},threshold=${THRESHOLD},use_cache=True,show_speed=True

# HumanEval
accelerate launch --num_processes=1 eval_llada_baseline.py --tasks humaneval --num_fewshot 0 \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=$((GEN_LENGTH/BLOCK_LENGTH)),block_length=${BLOCK_LENGTH},threshold=${THRESHOLD},use_cache=True,show_speed=True \
--output_path eval_results/humaneval/thres_prefix_cache --log_samples
```

#### 4. Threshold-based Parallel Decoding + Dual Cache (Fast-dLLM)
- `steps = gen_length / block_length`
- `threshold=0.9`, `use_cache=True`, `dual_cache=True`

```bash
# GSM8K
accelerate launch --num_processes=1 eval_llada_baseline.py --tasks gsm8k --num_fewshot 5 \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=$((GEN_LENGTH/BLOCK_LENGTH)),block_length=${BLOCK_LENGTH},threshold=${THRESHOLD},use_cache=True,dual_cache=True,show_speed=True

# HumanEval
accelerate launch --num_processes=1 eval_llada_baseline.py --tasks humaneval --num_fewshot 0 \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=$((GEN_LENGTH/BLOCK_LENGTH)),block_length=${BLOCK_LENGTH},threshold=${THRESHOLD},use_cache=True,dual_cache=True,show_speed=True \
--output_path eval_results/humaneval/thres_dual_cache --log_samples
```

## AdaBlock Evaluation (`eval_llada_adablock.py`)

AdaBlock uses adaptive block length based on delimiter confidence.

### Common Parameters
```bash
MODEL_PATH="GSAI-ML/LLaDA-8B-Instruct"  # or "GSAI-ML/LLaDA-1.5-8B-Instruct"
GEN_LENGTH=512
INIT_BLOCK_LENGTH=32
THRESHOLD=0.9
```

### AdaBlock-specific Parameters
```bash
DELIMITER_THRESHOLD=0.3
DELIMITER_IDS="198"  # 198=newline; for multiple delimiters: "198,11,13"
```

### Methods

#### 1. AdaBlock + Fast-dLLM (No Cache)
- `delimiter_threshold=0.3` for adaptive block length

```bash
# GSM8K
accelerate launch --num_processes=1 eval_llada_adablock.py --tasks gsm8k --num_fewshot 5 \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=$((GEN_LENGTH/INIT_BLOCK_LENGTH)),block_length=${INIT_BLOCK_LENGTH},threshold=${THRESHOLD},delimiter_ids=${DELIMITER_IDS},delimiter_threshold=${DELIMITER_THRESHOLD},show_speed=True

# HumanEval
accelerate launch --num_processes=1 eval_llada_adablock.py --tasks humaneval --num_fewshot 0 \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=$((GEN_LENGTH/INIT_BLOCK_LENGTH)),block_length=${INIT_BLOCK_LENGTH},threshold=${THRESHOLD},delimiter_ids=${DELIMITER_IDS},delimiter_threshold=${DELIMITER_THRESHOLD},show_speed=True \
--output_path eval_results_adablock/humaneval/no_cache --log_samples
```

#### 2. AdaBlock + Fast-dLLM (Prefix Cache)
- `delimiter_threshold=0.3`
- `use_cache=True`

```bash
# GSM8K
accelerate launch --num_processes=1 eval_llada_adablock.py --tasks gsm8k --num_fewshot 5 \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=$((GEN_LENGTH/INIT_BLOCK_LENGTH)),block_length=${INIT_BLOCK_LENGTH},threshold=${THRESHOLD},delimiter_ids=${DELIMITER_IDS},delimiter_threshold=${DELIMITER_THRESHOLD},use_cache=True,show_speed=True

# HumanEval
accelerate launch --num_processes=1 eval_llada_adablock.py --tasks humaneval --num_fewshot 0 \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=$((GEN_LENGTH/INIT_BLOCK_LENGTH)),block_length=${INIT_BLOCK_LENGTH},threshold=${THRESHOLD},delimiter_ids=${DELIMITER_IDS},delimiter_threshold=${DELIMITER_THRESHOLD},use_cache=True,show_speed=True \
--output_path eval_results_adablock/humaneval/prefix_cache --log_samples
```

#### 3. AdaBlock + Fast-dLLM (Dual Cache)
- `delimiter_threshold=0.3`
- `use_cache=True`, `dual_cache=True`

```bash
# GSM8K
accelerate launch --num_processes=1 eval_llada_adablock.py --tasks gsm8k --num_fewshot 5 \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=$((GEN_LENGTH/INIT_BLOCK_LENGTH)),block_length=${INIT_BLOCK_LENGTH},threshold=${THRESHOLD},delimiter_ids=${DELIMITER_IDS},delimiter_threshold=${DELIMITER_THRESHOLD},use_cache=True,dual_cache=True,show_speed=True

# HumanEval
accelerate launch --num_processes=1 eval_llada_adablock.py --tasks humaneval --num_fewshot 0 \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=$((GEN_LENGTH/INIT_BLOCK_LENGTH)),block_length=${INIT_BLOCK_LENGTH},threshold=${THRESHOLD},delimiter_ids=${DELIMITER_IDS},delimiter_threshold=${DELIMITER_THRESHOLD},use_cache=True,dual_cache=True,show_speed=True \
--output_path eval_results_adablock/humaneval/dual_cache --log_samples
```

## PAG Predictor Evaluation (`eval_llada_pag.py`)

PAG uses a trained `phase_predict` checkpoint to predict one
`(block_size, refinement_steps)` tuple per block. The first block uses the
explicit seed tuple, and later blocks use realized `(block_size, nfe)` history
to predict the next schedule.

### PAG-specific Parameters
```bash
PREDICTOR_CKPT="/path/to/phase_predict.ckpt"
SEED_BLOCK_LENGTH=32
SEED_REFINEMENT_STEPS=4
```

Optional PAG-specific model args:
- `predictor_device=cpu`
- `max_block_length=${BLOCK_LENGTH}` (defaults to `block_length`)
- `max_refinement_steps=${GEN_LENGTH}` (defaults to `steps`)

### Example
```bash
accelerate launch --num_processes=1 eval_llada_pag.py --tasks gsm8k --num_fewshot 5 \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=$((GEN_LENGTH/BLOCK_LENGTH)),block_length=${BLOCK_LENGTH},threshold=${THRESHOLD},predictor_ckpt=${PREDICTOR_CKPT},seed_block_length=${SEED_BLOCK_LENGTH},seed_refinement_steps=${SEED_REFINEMENT_STEPS},use_cache=True,dual_cache=True,show_speed=True
```

### Prompt Log Runner
For one-off qualitative analysis, `run_pag_dummy_api.py` writes structured JSONL
records with decoded block text, predicted tuple, applied block size, and actual
NFE per block.

By default, this runner now probes AdaBlock for block 0 and uses the realized
first `(block_size, nfe)` as PAG's seed tuple. Pass
`--no-seed-from-adablock-first-block` to force the explicit `--seed-*` values.

```bash
CUDA_VISIBLE_DEVICES=0 uv run --python 3.11 python AdaBlock-dLLM/llada/run_pag_dummy_api.py \
  --model-path GSAI-ML/LLaDA-8B-Instruct \
  --prompt-file AdaBlock-dLLM/llada/sample_prompts.jsonl \
  --max-prompts 3 \
  --gen-length 256 \
  --steps 64 \
  --threshold 0.9 \
  --seed-block-length 32 \
  --seed-refinement-steps 4 \
  --predictor-ckpt output/phase_predict_model_checkpoint.pt \
  --predictor-device cuda \
  --device cuda \
  --dtype bfloat16 \
  --use-cache \
  --dual-cache \
  --disable-torch-compile \
  --log-file logs/llada_pag_inference.jsonl
```

View the log UI:
```bash
uv run --python 3.11 --group phase_cpd streamlit run scripts/view_llada_pag_logs.py -- \
  --log-file logs/llada_pag_inference.jsonl
```

### Quick PAG vs AdaBlock Comparison
Use this for preliminary side-by-side runs on a small prompt suite. It writes one
JSONL record per prompt with both generations, total NFE, block counts, elapsed
time, answer accuracy, and simple substring checks. Re-run this command after
pulling prompt or checker changes so the dashboard has fresh `answer_check`
fields.

By default, this comparison seeds PAG block 0 from AdaBlock's realized first
block for the same prompt: `seed_block_length = adablock.block_history[0]` and
`seed_refinement_steps = adablock.nfe_history[0]`. This keeps the initial
block-size/refinement setup aligned with AdaBlock. Pass
`--no-seed-from-adablock-first-block` to use the explicit `--seed-*` values
instead.

```bash
CUDA_VISIBLE_DEVICES=0 uv run --python 3.11 python AdaBlock-dLLM/llada/run_pag_vs_adablock_eval.py \
  --model-path GSAI-ML/LLaDA-8B-Instruct \
  --prompt-file AdaBlock-dLLM/llada/quick_eval_prompts.jsonl \
  --max-prompts 3 \
  --gen-length 256 \
  --steps 64 \
  --threshold 0.9 \
  --seed-block-length 32 \
  --seed-refinement-steps 4 \
  --predictor-ckpt output/phase_predict_model_checkpoint.pt \
  --predictor-device cuda \
  --adablock-init-block-length 32 \
  --delimiter-threshold 0.3 \
  --device cuda \
  --dtype bfloat16 \
  --use-cache \
  --dual-cache \
  --disable-torch-compile \
  --log-file logs/llada_pag_vs_adablock_eval.jsonl
```

View the comparison dashboard:
```bash
uv run --python 3.11 --group phase_cpd streamlit run scripts/view_llada_pag_vs_adablock.py -- \
  --log-file logs/llada_pag_vs_adablock_eval.jsonl
```

## Parameter Reference

### Common Parameters
| Parameter | Description |
|-----------|-------------|
| `model_path` | Path to the LLaDA model |
| `gen_length` | Total generation length |
| `steps` | Number of denoising steps, used if denoising step is fixed |
| `block_length` | Block size for semi-AR Decoding |
| `threshold` | Confidence threshold for token transfer |
| `show_speed` | Display speed metrics |

### Cache Parameters
| Parameter | Description |
|-----------|-------------|
| `use_cache` | Enable block-level KV cache (defaults to prefix cache) |
| `dual_cache` | Use dual cache instead of prefix cache |

### AdaBlock-specific Parameters
| Parameter | Description |
|-----------|-------------|
| `delimiter_ids` | Token IDs for delimiters (e.g., 198=newline, 11=comma, 13=period; refer to [LLaDA tokenizer](https://huggingface.co/GSAI-ML/LLaDA-8B-Base/blob/main/tokenizer.json)) |
| `delimiter_threshold` | Confidence threshold for adaptive block length |

## Post-processing (HumanEval/MBPP)

For code generation tasks, post-processing is required:
```bash
# For HumanEval
python postprocess_humaneval.py {samples_xxx.jsonl}

# For MBPP
python postprocess_mbpp.py {samples_xxx.jsonl}
```

## Notes

1. Run scripts from the `llada/` directory
2. Batch evaluation is currently not supported (`batch_size` must be 1)
3. For HumanEval/MBPP, samples are logged with `--log_samples` for post-processing
4. If GSM8K dataset fails to load (refer to this [issue](https://github.com/EleutherAI/lm-evaluation-harness/issues/3528)), update `dataset_path` in `lm_eval/tasks/gsm8k/*.yaml`:
   ```bash
   # Find your lm_eval installation path
   python -c "import lm_eval; print(lm_eval.__path__[0])"
   
   # Update all gsm8k yaml files (replace <path> with the output above)
   sed -i 's/dataset_path: gsm8k/dataset_path: openai\/gsm8k/g' <path>/tasks/gsm8k/*.yaml
   ```
