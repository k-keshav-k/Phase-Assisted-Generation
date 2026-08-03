# Run the complete RC-PAG v6 experiment

From the repository on the NYU cluster, run:

```bash
scripts/slurm/submit_rc_pag_all.sh
```

That single command runs every CPU and GPU stage in one resumable 48-hour A100 job:

```text
preflight → pilot → collect/reuse → fit → screen → calibrate → confirm → report → paper
```

The launcher automatically reuses `artifacts/rc_pag/rc-pag-7688c5235bd4` when that completed
v5 directory exists. Otherwise, point it to the run explicitly:

```bash
export RC_PAG_REUSE_FROM="/gpfs/scratch/sm12779/Phase-Assisted-Generation/artifacts/rc_pag/rc-pag-7688c5235bd4"
scripts/slurm/submit_rc_pag_all.sh
```

Only the 600 native full-budget traces/model are reused. V6 discards the failed v5 advantage
heads and all old selection/certificate results, refits its new estimators, and uses fresh tuning,
calibration, and confirmation prompts.

With reuse, the run performs 8,984 prompt-method generations. It deliberately stops if either
model misses the 8% tuning gate or the held-out 2% harm certificate. A positive paper headline
also requires a paired-bootstrap NFE-reduction lower bound above 5% for each model.

Monitor or resume with:

```bash
squeue -u "$USER"
ls -lt logs/rc_pag
scripts/slurm/submit_rc_pag_all.sh
```
