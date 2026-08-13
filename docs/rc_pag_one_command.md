# Run the complete v8 experiment

From the repository on the NYU cluster, run:

```bash
scripts/slurm/submit_rc_pag_all.sh
```

The default Hugging Face mode is `auto`: the job first checks the shared cache, temporarily goes
online only to download missing pinned models or dataset splits, verifies that every asset now
works offline, and then runs the full experiment offline. Partial downloads use the normal
Hugging Face cache and resume when the same command is submitted again.

Use a strict override only when needed:

```bash
# Never use the network; fail early if a pinned asset is absent.
RC_PAG_HF_MODE=offline scripts/slurm/submit_rc_pag_all.sh

# Keep Hugging Face online throughout the experiment for troubleshooting.
RC_PAG_HF_MODE=online scripts/slurm/submit_rc_pag_all.sh
```

The older `RC_PAG_HF_OFFLINE=1` and `RC_PAG_HF_OFFLINE=0` forms remain supported as aliases for
`offline` and `online`, respectively. If `auto` cannot reach Hugging Face from the GPU node, it
reports the exact missing assets; populate the same `RC_PAG_SCRATCH_ROOT` cache from a
network-enabled node and resubmit.

That single command runs every CPU and GPU stage in one resumable 48-hour A100 job:

```text
preflight → pilot → collect/reuse → fit → screen → calibrate → confirm → report → paper
```

The launcher automatically prefers the completed v7 traces in
`artifacts/rc_pag/rc-pag-c1eda289fb08`, then falls back to the completed v5 traces. To choose a
source explicitly:

```bash
export RC_PAG_REUSE_FROM="/gpfs/scratch/sm12779/Phase-Assisted-Generation/artifacts/rc_pag/rc-pag-c1eda289fb08"
scripts/slurm/submit_rc_pag_all.sh
```

Only the 600 native full-budget traces/model are reused. V8 discards every old policy decision,
refits its router, and uses fresh tuning, calibration, and confirmation prompts.

V8 speculates across AdaBlock iterations but verifies every accepted transition. It stops if any
generated sequence differs from paired AdaBlock, if either model misses 5% tuning savings, or if
the confirmatory NFE-reduction lower bound is not above 5%.

Before the long stages, a 32-prompt/model pilot also runs fixed-depth batched speculation. It aborts
on any AdaBlock token mismatch and writes an A100-hour estimate to `compute_projection.json`.

Monitor or resume with:

```bash
squeue -u "$USER"
ls -lt logs/rc_pag
scripts/slurm/submit_rc_pag_all.sh
```
