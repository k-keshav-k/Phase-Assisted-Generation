# Phase CPD

`phase_cpd/` is an offline research tool for inspecting pre-generated diffusion trace files and running change-point detection over token-level features.

## What it does

- loads curated trace JSON files from `phase_cpd/data/traces_real/`
- loads curated trace JSON files from `phase_cpd/data/traces_real/`
- extracts scalar token-level feature series from the stabilization step
- runs change-point detection with tunable settings
- visualizes segmented text, feature trajectories, and segment summary statistics

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

## Current feature and detector support

- Features:
  - `stabilizing_entropy`
  - `stabilizing_margin`
  - `stabilizing_prob`
- Detectors:
  - `pelt`
  - `kernel_cpd`
- PELT costs: `l2`, `normal`
- Kernel CPD kernels: `rbf`, `linear`, `cosine`
- Controls: penalty, min segment length, smoothing window

`stabilizing_prob` uses the selected-token probability at the first refinement step after which a token's identity no longer changes.
`stabilizing_margin` uses `top1_prob - top2_prob` at that same stabilization step.
`stabilizing_entropy` uses the full-vocabulary entropy at the stabilization step and is usually the better scalar when top-1 probabilities saturate.

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

## Trace format

Each trace JSON stores:

- trace metadata: `trace_id`, `backend`, `model_name`, `prompt`, `tags`
- final output text: `final_text`
- token list with `token_text`, `char_start`, `char_end`
- per-step observations such as `top1_prob`, `selected_logit`, `top2_prob`, and scalar extras like `entropy`
- decoding metadata such as chunk size, refinement steps, and run id

## Raw step-dump format for Dream

For Dream, the easiest path is:

1. instrument your generation loop to write one raw JSON dump per prompt
2. run `phase_cpd/collect_traces.py` to convert those dumps into unified trace JSON files
3. point the Streamlit app at the converted traces

Each raw dump can use this stepwise format:

```json
{
  "trace_id": "dream-sample-001",
  "prompt": "Explain why adaptive diffusion decoding can help.",
  "model_name": "dream-7b",
  "decoding_metadata": {
    "run_id": "dream-run-001",
    "chunk_size": 4,
    "refinement_steps": 8
  },
  "steps": [
    {
      "step_index": 0,
      "tokens": [
        {
          "token_index": 0,
          "token_text": "Adaptive",
          "top1_prob": 0.61,
          "selected_logit": 2.44,
          "top2_prob": 0.42
        }
      ]
    }
  ]
}
```

The converter will group token rows by `token_index` and build the per-token observation history used by the UI and PELT analysis.

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
  --alg "$DREAM_ALG" \
  --alg-temp "$DREAM_ALG_TEMP" \
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

## Adding new traces

1. Produce trace artifacts offline on your own machine or GPU workflow.
2. Convert them into the local `TraceRecord` schema.
3. Save the resulting JSON into `phase_cpd/data/traces_real/`.

The app will pick up the new file automatically on reload.
