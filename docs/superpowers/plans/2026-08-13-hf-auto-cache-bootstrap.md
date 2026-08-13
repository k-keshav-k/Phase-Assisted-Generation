# Hugging Face Auto Cache Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the v8 one-command Slurm job automatically download missing pinned Hugging Face models and datasets, then execute the experiment from cache in offline mode.

**Architecture:** A standalone bootstrap script discovers assets from the frozen experiment config and runs cache probes/downloads in isolated child Python processes so Hugging Face offline flags are applied before its libraries import. The Slurm worker resolves an `auto|offline|online` mode, invokes the bootstrap against its real cache paths, and starts the experiment only after successful verification.

**Tech Stack:** Python 3.11, `huggingface_hub`, `datasets`, Bash/Slurm, pytest, Ruff.

---

## File map

- Create `scripts/bootstrap_rc_pag_hf.py`: asset discovery, isolated probe/fetch workers, mode resolution, diagnostics.
- Create `tests/experiments/test_bootstrap_rc_pag_hf.py`: unit and subprocess-orchestration behavior.
- Modify `scripts/slurm/rc_pag_a100.sbatch`: resolve mode, invoke bootstrap, set final offline flags.
- Modify `tests/experiments/test_submit_rc_pag_all.py`: assert Slurm bootstrap wiring and compatibility.
- Modify `docs/rc_pag_one_command.md`: document automatic behavior and explicit overrides.
- Modify `README.md`: keep the primary v8 command description accurate.

### Task 1: Asset manifest and mode resolution

**Files:**
- Create: `scripts/bootstrap_rc_pag_hf.py`
- Create: `tests/experiments/test_bootstrap_rc_pag_hf.py`

- [x] **Step 1: Write failing manifest and precedence tests**

Add tests that import the script with `importlib.util`, load
`configs/experiments/rc_pag_neurips_workshop_v8.yaml`, and assert that `discover_assets()` returns
both pinned model repositories and unique dataset `(path, config, split, revision)` entries. Add
parameterized assertions that `resolve_mode({}) == "auto"`, legacy `RC_PAG_HF_OFFLINE=1/0` maps to
`offline/online`, and explicit `RC_PAG_HF_MODE` wins.

- [x] **Step 2: Run tests and verify failure**

Run:

```bash
uv run pytest tests/experiments/test_bootstrap_rc_pag_hf.py -q
```

Expected: failure because `scripts/bootstrap_rc_pag_hf.py` does not exist.

- [x] **Step 3: Implement typed asset discovery and mode validation**

Create a frozen `HFAsset` dataclass with `kind`, `name`, `repository`, `revision`, `config`, and
`split`; implement JSON serialization, deterministic deduplication, `discover_assets(config)`, and:

```python
def resolve_mode(env: Mapping[str, str]) -> str:
    explicit = env.get("RC_PAG_HF_MODE", "").strip().lower()
    if explicit:
        if explicit not in {"auto", "offline", "online"}:
            raise ValueError("RC_PAG_HF_MODE must be auto, offline, or online")
        return explicit
    legacy = env.get("RC_PAG_HF_OFFLINE", "").strip().lower()
    if legacy in {"1", "true", "yes"}:
        return "offline"
    if legacy in {"0", "false", "no"}:
        return "online"
    if legacy:
        raise ValueError("RC_PAG_HF_OFFLINE must be a boolean value")
    return "auto"
```

- [x] **Step 4: Run manifest tests**

Run the Task 1 pytest command. Expected: all new tests pass.

### Task 2: Strict offline probe and selective fetch

**Files:**
- Modify: `scripts/bootstrap_rc_pag_hf.py`
- Modify: `tests/experiments/test_bootstrap_rc_pag_hf.py`

- [x] **Step 1: Write failing orchestration tests**

Inject a fake worker runner into `bootstrap_assets()`. Assert:

```python
assert calls == [
    ("probe", all_assets, True),
    ("fetch", missing_assets, False),
    ("probe", all_assets, True),
]
```

Also assert cached auto mode performs only the first probe, offline mode never fetches, online mode
fetches all assets once, and a failed final probe raises `BootstrapError` listing each missing asset.

- [x] **Step 2: Run tests and verify the new assertions fail**

Run the Task 1 pytest command. Expected: failures for missing `bootstrap_assets` behavior.

- [x] **Step 3: Implement workers and diagnostics**

Implement worker materialization as follows:

```python
if asset.kind == "model":
    snapshot_download(
        repo_id=asset.repository,
        revision=asset.revision,
        local_files_only=offline,
    )
else:
    load_dataset(
        asset.repository,
        asset.config,
        split=asset.split,
        revision=asset.revision,
    )
```

Catch errors per asset and write a JSON result file rather than parsing library stdout. For isolated
children, copy the environment and set all three flags (`HF_HUB_OFFLINE`, `HF_DATASETS_OFFLINE`,
`TRANSFORMERS_OFFLINE`) to `1` for probes and `0` for fetches. `auto` must probe all, fetch only
missing, and probe all again; `offline` performs one probe; `online` fetches all once. Print a short
cache-hit/download summary and exit nonzero with actionable identities when verification fails.

- [x] **Step 4: Run bootstrap tests**

Run the Task 1 pytest command. Expected: all bootstrap tests pass without network access.

### Task 3: Wire bootstrap into the A100 worker

**Files:**
- Modify: `scripts/slurm/rc_pag_a100.sbatch`
- Modify: `tests/experiments/test_submit_rc_pag_all.py`

- [x] **Step 1: Write failing launcher assertions**

Assert the worker defaults to `RC_PAG_HF_MODE=auto`, preserves the legacy override, invokes:

```bash
uv run --group phase_cpd_dream python scripts/bootstrap_rc_pag_hf.py \
    --config "$RC_PAG_CONFIG" \
    --mode "$RC_PAG_HF_MODE"
```

before `scripts/run_rc_pag.py`, and sets the three final offline flags to `1` for `auto/offline` and
`0` for `online`.

- [x] **Step 2: Run launcher tests and verify failure**

Run:

```bash
uv run pytest tests/experiments/test_submit_rc_pag_all.py -q
```

Expected: failure because the worker does not invoke the bootstrap.

- [x] **Step 3: Implement Bash mode resolution and bootstrap call**

Resolve mode before entering Singularity, validate its value, export it, and replace the unconditional
offline exports with mode-dependent final flags. Invoke the bootstrap after `uv sync` and before the
experiment. Keep `HF_HOME` and `HF_DATASETS_CACHE` unchanged so bootstrap and inference share files.

- [x] **Step 4: Run launcher tests and syntax checks**

Run:

```bash
uv run pytest tests/experiments/test_submit_rc_pag_all.py -q
bash -n scripts/slurm/rc_pag_a100.sbatch scripts/slurm/submit_rc_pag_all.sh
```

Expected: tests pass and Bash prints no syntax errors.

### Task 4: Update the one-command documentation

**Files:**
- Modify: `docs/rc_pag_one_command.md`
- Modify: `README.md`

- [x] **Step 1: Replace obsolete first-run instructions**

Document that the default `auto` mode probes the shared cache, downloads only missing pinned assets,
then runs offline. Document strict overrides:

```bash
RC_PAG_HF_MODE=offline scripts/slurm/submit_rc_pag_all.sh
RC_PAG_HF_MODE=online scripts/slurm/submit_rc_pag_all.sh
```

Retain `RC_PAG_HF_OFFLINE=0/1` only as a backwards-compatible option.

- [x] **Step 2: Check documentation consistency**

Run:

```bash
rg -n "RC_PAG_HF_(MODE|OFFLINE)|offline mode" README.md docs/rc_pag_one_command.md scripts/slurm
```

Expected: the one-command guide no longer instructs ordinary first-time users to manually disable
offline mode.

### Task 5: Full focused verification

**Files:**
- Verify all files above.

- [x] **Step 1: Run focused tests**

```bash
uv run pytest tests/experiments/test_bootstrap_rc_pag_hf.py \
  tests/experiments/test_submit_rc_pag_all.py \
  tests/experiments/test_rc_pag_runtime.py -q
```

Expected: all tests pass.

- [x] **Step 2: Run static checks**

```bash
uv run ruff check scripts/bootstrap_rc_pag_hf.py \
  tests/experiments/test_bootstrap_rc_pag_hf.py \
  tests/experiments/test_submit_rc_pag_all.py \
  src/pag/experiments/rc_pag_runtime.py
uv run ruff format --check scripts/bootstrap_rc_pag_hf.py \
  tests/experiments/test_bootstrap_rc_pag_hf.py \
  tests/experiments/test_submit_rc_pag_all.py \
  src/pag/experiments/rc_pag_runtime.py
git diff --check
```

Expected: all commands exit zero.

- [x] **Step 3: Run the mock v8 funnel**

```bash
mock_root="$(mktemp -d)"
uv run python scripts/run_rc_pag.py all \
  --config configs/experiments/rc_pag_neurips_workshop_v8.yaml \
  --mock --device cpu --output-root "$mock_root" --limit 2
```

Expected: `All stages complete`; no network or GPU is used.

- [x] **Step 4: Review final diff**

Run `git status --short` and `git diff --stat`. Expected: only planned implementation, tests, and
documentation are changed beyond the already committed design and plan.
