# Phase CPD

`phase_cpd/` is an offline research tool for inspecting pre-generated diffusion trace files and running change-point detection over token-level features.

## What it does

- loads curated trace JSON files from `phase_cpd/data/traces/`
- extracts a token-level feature series
- runs PELT change-point detection with tunable settings
- visualizes segmented text, feature trajectories, and segment summary statistics

The app does not call a model, host an API, or generate traces live. It is intended for fast qualitative analysis over traces you already produced elsewhere.

## Setup

```bash
uv sync --group phase_cpd
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
phase_cpd/data/traces/
```

Two mock traces are included so the app runs immediately.

## Current feature and detector support

- Feature: `top1_prob`
- Detector: PELT via `ruptures`
- Costs: `l2`, `normal`
- Controls: penalty, min segment length, smoothing window

`top1_prob` uses the selected token probability from the final available refinement step for each token.

## Real backend status

- `mock`: ready now through checked-in example traces
- `Dream`: importer contract added in `phase_cpd/importers/dream.py`
- `LLaDA`: importer contract added in `phase_cpd/importers/llada.py`

The Dream and LLaDA importers are explicit extension points for converting locally produced trace artifacts into the unified `TraceRecord` schema.

## Trace format

Each trace JSON stores:

- trace metadata: `trace_id`, `backend`, `model_name`, `prompt`, `tags`
- final output text: `final_text`
- token list with `token_text`, `char_start`, `char_end`
- per-step observations such as `top1_prob`, `selected_logit`, `top2_prob`
- decoding metadata such as chunk size, refinement steps, and run id

## Raw step-dump format for real backends

For Dream and LLaDA, the easiest path is:

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

## Converting raw Dream/LLaDA dumps

Dream:

```bash
python phase_cpd/collect_traces.py \
  --backend dream \
  --source /path/to/raw/dream_dumps \
  --output-dir phase_cpd/data/traces_real
```

LLaDA:

```bash
python phase_cpd/collect_traces.py \
  --backend llada \
  --source /path/to/raw/llada_dumps \
  --output-dir phase_cpd/data/traces_real
```

## NYU Burst / Slurm

A starter Slurm job is included at:

```text
phase_cpd/slurm/collect_phase_traces_nyu.sbatch
```

It assumes:

- you already have a working Dream and/or LLaDA environment inside your Singularity container
- your instrumented backend command writes raw step-dump JSON files
- those raw dumps are converted in-place by `phase_cpd/collect_traces.py`

The main env vars you need to set before `sbatch` are:

- `PROJECT_DIR`
- `PROMPTS_FILE`
- `RAW_TRACE_ROOT`
- `TRACE_OUTPUT_DIR`
- `DREAM_TRACE_CMD`
- `LLADA_TRACE_CMD`

Example:

```bash
export DREAM_TRACE_CMD='python student/trace_jobs/run_dream_trace_dump.py --prompts "$PROMPTS_FILE" --output-dir "$RAW_TRACE_ROOT/dream"'
export LLADA_TRACE_CMD='python student/trace_jobs/run_llada_trace_dump.py --prompts "$PROMPTS_FILE" --output-dir "$RAW_TRACE_ROOT/llada"'
sbatch phase_cpd/slurm/collect_phase_traces_nyu.sbatch
```

## Adding new traces

1. Produce trace artifacts offline on your own machine or GPU workflow.
2. Convert them into the local `TraceRecord` schema.
3. Save the resulting JSON into `phase_cpd/data/traces/`.

The app will pick up the new file automatically on reload.
