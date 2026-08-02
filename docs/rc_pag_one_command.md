# Run the complete RC-PAG v5 experiment

From the repository on the NYU cluster:

```bash
scripts/slurm/submit_rc_pag_all.sh
```

This one command runs every CPU and GPU stage in one resumable A100 job. Run it again after a
timeout or preemption; completed prompt records are skipped.

To reuse the compatible traces and paired q500/AdaBlock screen from the failed v4 run:

```bash
export RC_PAG_REUSE_FROM="/gpfs/scratch/$USER/Phase-Assisted-Generation/artifacts/rc_pag/<V4_RUN_ID>"
scripts/slurm/submit_rc_pag_all.sh
```

V5 reuses only raw exact-loop traces and paired rollout rows. It refits both new advantage
heads and uses a disjoint tuning split; no v4 policy selection, certificate, or confirmation
result is reused.

```text
preflight → pilot → collect → fit → rollout → refit → screen
          → calibrate → confirm → report → paper
```

The run stops deliberately if either model fails the 8% tuning-headroom gate or the held-out joint
certificate (harm at most 2%, mean NFE saving above 5%). The pilot writes the machine-specific GPU
estimate to `compute_projection.json`.

Fresh workload: 10,784 prompt-method generations. With compatible v4 reuse: 8,984 generations.

Monitor with:

```bash
squeue -u "$USER"
ls -lt logs/rc_pag
```
