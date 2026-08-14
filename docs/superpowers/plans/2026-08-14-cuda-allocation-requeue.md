# CUDA Allocation Smoke Check and Requeue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect a broken or busy Slurm-assigned GPU before model loading and requeue the resumable job at most twice with actionable diagnostics.

**Architecture:** A small Python command performs a real CUDA allocation, arithmetic operation, and synchronization inside the same Singularity/uv environment as the experiment. It returns exit code 75 for infrastructure unavailability. The outer Slurm worker recognizes only that code, uses Slurm's restart counter to cap automatic retries, and otherwise preserves existing exit and preemption behavior.

**Tech Stack:** Python 3.11, PyTorch, Bash, Slurm, pytest, Ruff.

---

### Task 1: CUDA allocation smoke command

**Files:**
- Create: `scripts/check_rc_pag_cuda.py`
- Create: `tests/experiments/test_check_rc_pag_cuda.py`

- [ ] **Step 1: Write failing checker tests**

Test a fake CUDA backend that succeeds and one whose tensor allocation raises `RuntimeError("CUDA-capable device(s) is/are busy or unavailable")`. Require structured JSON and exit codes 0 and 75 respectively.

- [ ] **Step 2: Verify the tests fail**

Run: `uv run pytest tests/experiments/test_check_rc_pag_cuda.py -q`

Expected: import failure because `scripts.check_rc_pag_cuda` does not exist.

- [ ] **Step 3: Implement the checker**

Implement `check_cuda(torch_module)` and `main()` so the check verifies `is_available()`, exactly one visible device or more, allocates a small tensor on `cuda:0`, performs an in-place arithmetic operation, synchronizes, and prints JSON diagnostics. Catch exceptions only at the command boundary and return 75.

- [ ] **Step 4: Verify checker tests**

Run: `uv run pytest tests/experiments/test_check_rc_pag_cuda.py -q`

Expected: all tests pass.

### Task 2: Capped Slurm recovery

**Files:**
- Modify: `scripts/slurm/rc_pag_a100.sbatch`
- Modify: `tests/experiments/test_submit_rc_pag_all.py`

- [ ] **Step 1: Write failing worker assertions**

Require the CUDA checker to run after `uv sync` and before Hugging Face bootstrap/model execution. Require exit code 75 handling, `SLURM_RESTART_COUNT`, a default maximum of two retries, and `scontrol requeue`.

- [ ] **Step 2: Verify the worker test fails**

Run: `uv run pytest tests/experiments/test_submit_rc_pag_all.py -q`

Expected: failure because the worker has no CUDA allocation check or exit-75 recovery.

- [ ] **Step 3: Add smoke invocation and capped recovery**

Run `uv run --group phase_cpd_dream python scripts/check_rc_pag_cuda.py` inside the container before asset bootstrap. In the outer worker, requeue exit 75 while `SLURM_RESTART_COUNT < RC_PAG_MAX_CUDA_REQUEUES`, defaulting to two. After the cap, print the job ID, node, visible devices, and an instruction to contact cluster support; return 75 without looping.

- [ ] **Step 4: Verify worker and shell syntax**

Run: `uv run pytest tests/experiments/test_submit_rc_pag_all.py -q`

Run: `bash -n scripts/slurm/rc_pag_a100.sbatch scripts/slurm/submit_rc_pag.sh scripts/slurm/submit_rc_pag_all.sh`

Expected: tests pass and Bash emits no syntax errors.

### Task 3: Regression verification and commit

**Files:**
- Modify only files required by failures.

- [ ] **Step 1: Run relevant suites**

Run: `uv run pytest tests/experiments/test_check_rc_pag_cuda.py tests/experiments/test_submit_rc_pag_all.py tests/experiments/test_rc_pag_orchestrator.py -q`

Expected: all tests pass.

- [ ] **Step 2: Run lint, formatting, and diff checks**

Run: `uv run ruff check scripts/check_rc_pag_cuda.py tests/experiments/test_check_rc_pag_cuda.py tests/experiments/test_submit_rc_pag_all.py`

Run: `uv run ruff format --check scripts/check_rc_pag_cuda.py tests/experiments/test_check_rc_pag_cuda.py tests/experiments/test_submit_rc_pag_all.py`

Run: `git diff --check`

Expected: all commands exit zero.

- [ ] **Step 3: Commit**

```bash
git add scripts/check_rc_pag_cuda.py scripts/slurm/rc_pag_a100.sbatch \
  tests/experiments/test_check_rc_pag_cuda.py tests/experiments/test_submit_rc_pag_all.py
git commit -m "fix: requeue unavailable cuda allocations"
```
