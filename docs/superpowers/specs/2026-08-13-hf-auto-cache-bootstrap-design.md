# Hugging Face automatic cache bootstrap

## Goal

Make the one-command v8 Slurm run self-healing when its pinned Hugging Face models or datasets
are not yet in the job's cache. The job should use cached assets without network traffic, fetch
only missing assets when network access is available, and run the experiment offline after the
bootstrap completes.

## Current failure

`rc_pag_a100.sbatch` exports `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, and
`TRANSFORMERS_OFFLINE=1` before preflight. The recent runtime changes correctly accept a pinned
model that is already cached, but deliberately reject an absent snapshot. No component switches
online to populate the empty cache, so a first run fails in preflight.

## Selected design

Add a small, independently testable asset-bootstrap script and run it inside the existing
Singularity environment before `scripts/run_rc_pag.py`.

The bootstrap will:

1. Load the selected RC-PAG YAML and enumerate every pinned model and dataset revision.
2. Start a child Python process in strict offline mode and verify/materialize all required assets.
3. If the offline probe reports missing assets, start one online child process to fetch only those
   assets into the same `HF_HOME` and `HF_DATASETS_CACHE` used by the experiment.
4. Run the strict offline probe again. The main experiment starts only when this final probe
   succeeds.
5. Keep the parent worker and the full experiment in offline mode after bootstrap.

Model assets are resolved with `snapshot_download(repository, revision=<pinned SHA>)`. Dataset
assets are materialized using the exact path, configuration, split, and revision declared in the
experiment config. Revisions are never changed or inferred from a branch name.

The launcher default becomes `RC_PAG_HF_MODE=auto`. Supported modes are:

- `auto`: offline probe, online fetch only if needed, final offline verification.
- `offline`: never access the network; fail early if anything is missing.
- `online`: allow ordinary Hugging Face online resolution for troubleshooting.

The existing `RC_PAG_HF_OFFLINE` variable remains accepted as a backwards-compatible override:
`1` maps to `offline` and `0` maps to `online`. An explicitly supplied `RC_PAG_HF_MODE` takes
precedence.

## Error handling

Missing cache entries are not experiment failures in `auto` mode; they trigger bootstrap download.
If the online child cannot reach Hugging Face, the job exits before model allocation or generation
with the missing repository/dataset identities and instructions to pre-populate the shared cache or
use a network-enabled node. Authentication errors are similarly reported without exposing tokens.

Partial Hugging Face downloads remain resumable in their standard cache layout. Re-submitting the
same one-command job reuses completed files.

## Testing

Unit tests will cover:

- all assets cached: no online fetch;
- model missing: only that pinned snapshot is fetched;
- dataset missing: only that exact dataset configuration/split/revision is materialized;
- failed fetch: clear nonzero exit and missing-asset report;
- final offline verification is mandatory;
- mode and legacy-variable precedence;
- Slurm worker invokes bootstrap before the experiment and uses the same cache paths.

Shell syntax, focused Python tests, Ruff, and a mock v8 orchestration run will verify integration.

## Non-goals

The change does not weaken pinned revisions, silently select newer artifacts, embed Hugging Face
credentials, or claim that an uncached asset can be fetched from a node with no network route.
