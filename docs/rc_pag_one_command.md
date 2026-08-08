# Run the complete v8 experiment

From the repository on the NYU cluster, run:

```bash
scripts/slurm/submit_rc_pag_all.sh
```

The job defaults to Hugging Face offline mode so a transient network outage cannot break a
resumed run. If this machine has not cached the pinned models and datasets yet, run once with
`RC_PAG_HF_OFFLINE=0 scripts/slurm/submit_rc_pag_all.sh`.

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
