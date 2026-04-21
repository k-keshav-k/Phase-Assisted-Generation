# Phase CPD

`phase_cpd/` is an offline research tool for inspecting pre-generated diffusion trace files and running change-point detection over token-level features.

## What it does

- loads curated trace JSON files from `phase_cpd/data/traces_real/`
- extracts scalar token-level feature series from the stabilization step
- runs change-point detection with tunable settings
- visualizes token boundary overlays, feature trajectories, and segment summary statistics

The app does not call a model, host an API, or generate traces live. It is intended for fast qualitative analysis over traces you already produced elsewhere.

## Setup

```bash
uv sync --group phase_cpd
```

For Dream trace collection on GPU, use the Dream runtime group:

```bash
uv sync --group phase_cpd_dream
```

## Run

```bash
uv run --group phase_cpd streamlit run phase_cpd/app.py
```

Or use the Make target:

```bash
make run-phase-cpd
```

## Data source

The UI reads curated traces from:

```text
phase_cpd/data/traces_real/
```

If that directory is missing or empty, the app raises an error instead of falling back to synthetic data.

Current Dream traces are profile-qualified. A typical trace id looks like:

```text
prompt-001__entropy_stochastic__seed-0
```

The app catalog exposes filters for backend, model, tags, run id, Dream trace profile, and seed. It also invalidates its trace cache when JSON files in `phase_cpd/data/traces_real/` change, so replacing or adding trace files should be visible after a Streamlit rerun.

## Current feature and detector support

- Features:
  - `stabilizing_entropy`
  - `stabilizing_margin`
  - `stabilizing_prob`
  - `stabilizing_refinement_step`
- Detectors:
  - `pelt`
  - `kernel_cpd`
- PELT costs: `l2`, `normal`
- Kernel CPD kernels: `rbf`, `linear`, `cosine`
- Controls: penalty, min segment length, smoothing window

`stabilizing_prob` uses the selected-token probability at the first refinement step after which a token's identity no longer changes.
`stabilizing_margin` uses `top1_prob - top2_prob` at that same stabilization step.
`stabilizing_entropy` uses the full-vocabulary entropy at the stabilization step and is usually the better scalar when top-1 probabilities saturate.
`stabilizing_refinement_step` uses the raw `step_index` of that first stabilization step, which is useful if you want to model when tokens settle.

The UI also includes a per-token stabilization table so you can inspect refinement step, entropy, margin, and probability together before exporting features for downstream modeling.

The smoothing window is a centered moving average applied to the chosen feature before CPD.
`1` means no smoothing. Larger values suppress local noise, but they also blur sharp short-lived
transitions and can move weak boundaries.

## Real backend status

- `Dream`: local trace-dump entrypoint added in `phase_cpd/trace_jobs/`

Dream raw step dumps are converted into the unified `TraceRecord` schema through `phase_cpd/collect_traces.py` and `phase_cpd/importers/common.py`.

For Dream, the repo now includes a real Hugging Face runtime path built around the official Dream inference API:

- `AutoModel.from_pretrained(..., trust_remote_code=True)`
- `model.diffusion_generate(...)`
- `generation_tokens_hook_func(step, x, logits)` to record per-step token probabilities/logits

The runtime assumption matches the official Dream README and demo scripts.

The built-in Dream trace profiles are:

| Profile | Dream `alg` | Dream `alg_temp` | Notes |
| --- | --- | --- | --- |
| `entropy_stochastic` | `entropy` | `0.1` | Preferred phase-predictor training profile when generation `temperature=0.0`. The name refers to refinement-order stochasticity, not token sampling. |
| `entropy_det` | `entropy` | `0.0` | Comparison profile. It preserves answer quality well but can make stabilization steps too monotone with token index. |
| `origin_random` | `origin` | omitted | Ablation profile. Random refinement order adds policy noise and can hurt task correctness, especially on math. |

The recommended/default collection settings for phase-predictor training are:

```text
--trace-profile entropy_stochastic
--temperature 0.0
```

That profile resolves to `alg=entropy` and `alg_temp=0.1`.

Use `--trace-profile all` only when you want comparison/ablation traces. `--alg` and `--alg-temp` are still accepted for backward compatibility, but they must match one of the supported profiles.

`temperature` and `alg_temp` control different sources of randomness:

- generation `temperature` controls token sampling randomness in the emitted answer. Keep it at `0.0` for phase-predictor training traces so math/task correctness is not degraded by token sampling.
- `alg_temp` controls randomness in Dream's entropy/confidence-based refinement order. A small nonzero value such as `0.1` reduces overly monotone stabilization labels without changing token sampling.

## Trace format

Each trace JSON stores:

- trace metadata: `trace_id`, `backend`, `model_name`, `prompt`, `tags`
- final output text: `final_text`
- token list with `token_text`, `char_start`, `char_end`
- per-step observations such as `top1_prob`, `selected_logit`, `top2_prob`, and scalar extras like `entropy`, `is_mask`, `changed_from_prev_step`, and delimiter probabilities
- per-step summaries such as mask count, changed-token count, active mask span, and the highest-confidence delimiter position
- decoding metadata such as run id, Dream profile, algorithm settings, seed, prompt seed, generation length, denoising steps, dtype, device, and delimiter feature definitions

## Raw step-dump format for Dream

For Dream, the easiest path is:

1. run or instrument generation to write one raw JSON dump per prompt/profile/seed
2. run `phase_cpd/collect_traces.py` to convert those dumps into unified trace JSON files
3. point the Streamlit app at the converted traces

Each raw dump can use this stepwise format:

```json
{
  "trace_id": "prompt-001__entropy_stochastic__seed-0",
  "prompt": "Explain why adaptive diffusion decoding can help.",
  "model_name": "Dream-org/Dream-v0-Instruct-7B",
  "decoding_metadata": {
    "run_id": "dream-run-001",
    "trace_profile": "entropy_stochastic",
    "alg": "entropy",
    "temperature": 0.0,
    "alg_temp": 0.1,
    "seed": 0,
    "prompt_seed": 123456,
    "max_new_tokens": 256,
    "steps": 256,
    "delimiter_features": [
      {
        "text": ".",
        "feature_key": "delimiter_prob_period",
        "token_id": 13
      }
    ]
  },
  "steps": [
    {
      "step_index": 0,
      "summary": {
        "mask_count": 12,
        "changed_count": 3,
        "active_start": 4,
        "active_end": 16,
        "active_count": 12,
        "best_delimiter_index": 10,
        "max_delimiter_confidence": 0.81
      },
      "tokens": [
        {
          "token_index": 0,
          "token_text": "Adaptive",
          "top1_prob": 0.61,
          "selected_logit": 2.44,
          "top2_prob": 0.42,
          "extras": {
            "entropy": 1.23,
            "is_mask": 0.0,
            "changed_from_prev_step": 1.0,
            "delimiter_prob_max": 0.04,
            "delimiter_prob_period": 0.04
          }
        }
      ]
    }
  ]
}
```

The converter will group token rows by `token_index` and build the per-token observation history used by the UI and PELT analysis.

The local Dream runner writes this raw format directly:

```bash
uv run --group phase_cpd_dream --python 3.11 \
  python -m phase_cpd.trace_jobs.run_dream_trace_dump \
  --prompts phase_cpd/data/prompts/research_prompts.jsonl \
  --output-dir outputs/phase_cpd_raw/dream \
  --model-name Dream-org/Dream-v0-Instruct-7B \
  --temperature 0.0 \
  --trace-profile entropy_stochastic \
  --seed 0
```

## Converting raw Dream dumps

Dream:

```bash
python phase_cpd/collect_traces.py \
  --backend dream \
  --source /path/to/raw/dream_dumps \
  --output-dir phase_cpd/data/traces_real
```

## NYU Burst / Slurm

A starter Slurm job is included at:

```text
phase_cpd/slurm/collect_phase_traces_nyu.sbatch
```

It assumes:

- you have `uv` available inside your Singularity/container environment
- the node can load the Dream weights from Hugging Face or a local cache
- those raw dumps are converted in-place by `phase_cpd/collect_traces.py`

The main env vars you need to set before `sbatch` are:

- `PROJECT_DIR`
- `PROMPTS_FILE`
- `RAW_TRACE_ROOT`
- `TRACE_OUTPUT_DIR`
- `UV_PROJECT_ENVIRONMENT` if you want the Dream venv somewhere other than the default project-local path
- `UV_CACHE_DIR` if you want the uv cache somewhere other than the default project-local path
- `DREAM_TRACE_PROFILE` if you want a profile other than the default Slurm value of `entropy_stochastic`
- `DREAM_TEMPERATURE` if you want a generation temperature other than the default `0.0`
- `DREAM_SEED` if you want a base seed other than `0`
- optionally `DREAM_TRACE_CMD` if you do not want to use the default local runner

Example:

```bash
sbatch phase_cpd/slurm/collect_phase_traces_nyu.sbatch
```

By default the job now does two uv-managed steps:

1. Create or update a dedicated project-local environment:

```bash
uv sync --group phase_cpd_dream --python 3.11
```

2. Run Dream trace collection inside that uv environment:

```bash
uv run --group phase_cpd_dream --python 3.11 \
  python -m phase_cpd.trace_jobs.run_dream_trace_dump \
  --prompts "$PROMPTS_FILE" \
  --output-dir "$RAW_TRACE_ROOT/dream" \
  --model-name "$DREAM_MODEL_NAME" \
  --max-new-tokens "$DREAM_MAX_NEW_TOKENS" \
  --steps "$DREAM_STEPS" \
  --temperature "$DREAM_TEMPERATURE" \
  --top-p "$DREAM_TOP_P" \
  --trace-profile "$DREAM_TRACE_PROFILE" \
  --seed "$DREAM_SEED" \
  --torch-dtype "$DREAM_TORCH_DTYPE"
```

Recommended Dream environment versions from the official repo:

```text
Python 3.11+
torch==2.5.1
transformers==4.46.2
```

On Linux, `torch`, `torchvision`, and `torchaudio` are pinned to the PyTorch CUDA 12.1 wheel index through `pyproject.toml`, so the first `uv sync --group phase_cpd_dream` should provision the GPU runtime into the dedicated uv environment instead of reusing `llmr`.

Once the raw files are written, the Slurm job converts them into unified `TraceRecord` JSON under `TRACE_OUTPUT_DIR` via:

```bash
uv run --group phase_cpd_dream --python 3.11 \
  python -m phase_cpd.collect_traces \
  --backend dream \
  --source "$RAW_TRACE_ROOT/dream" \
  --output-dir "$TRACE_OUTPUT_DIR"
```

## Scheduler report utilities

Two CLI helpers build scheduler-supervision data from converted Dream traces:

```bash
python -m phase_cpd.export_scheduler_dataset \
  --source phase_cpd/data/traces_real \
  --output outputs/phase_cpd_scheduler_rows.jsonl
```

```bash
python -m phase_cpd.report_trace_profiles \
  --source phase_cpd/data/traces_real \
  --output outputs/phase_cpd_trace_profile_report.json
```

`export_scheduler_dataset.py` writes one JSONL row per synthetic frontier decision. `report_trace_profiles.py` summarizes trace count, token count, scheduler row count, direct mask-to-final fraction, rewrite count, stabilization monotonicity, and oracle variance metrics by `trace_profile`.

The profile report also includes small comparison metrics intended for profile selection:

- task correctness and exact-match rates when the trace metadata includes labels such as `task_correct`, `exact_match`, or an expected answer
- stabilization monotonicity versus token index
- `token_index_stabilization_r2`, the variance in stabilization step explained by token index under a simple linear fit
- stabilization-step min, mean, standard deviation, and max

Use these metrics to confirm the expected tradeoff: `entropy_det` often produces very monotone stabilization steps, `origin_random` can inject too much policy noise, and `entropy` with generation `temperature=0.0` plus small nonzero `alg_temp` is the preferred middle ground for scheduler training.

## Adding new traces

1. Produce trace artifacts offline on your own machine or GPU workflow.
2. Convert them into the local `TraceRecord` schema.
3. Save the resulting JSON into `phase_cpd/data/traces_real/`.

The app will pick up the new file automatically on reload.
