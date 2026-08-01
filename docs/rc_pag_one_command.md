# Run the complete RC-PAG v4 experiment

From the repository on the NYU cluster, submit everything with:

```bash
scripts/slurm/submit_rc_pag_all.sh
```

The command runs every CPU and GPU stage in one resumable A100 job: preflight, parity pilot,
trace collection, estimator fitting, tuning, joint calibration, fresh confirmation, reporting, and
paper generation. Run the same command after interruption; completed records are reused.

Set cluster paths only when they differ from the defaults:

```bash
export RC_PAG_OUTPUT_ROOT="/scratch/${USER}/rc_pag/artifacts"
export OVERLAY_PATH="/scratch/${USER}/overlay-25GB-500K.ext3"
export SIF_PATH="/scratch/${USER}/ubuntu-20.04.3.sif"
scripts/slurm/submit_rc_pag_all.sh
```

V4 uses one risk estimator and three thresholds. It proceeds to the fresh 5,184-generation
confirmation only if both LLaDA and Dream pass tuning and held-out calibration jointly certifies
at most 2% harmful regression and at least 5% mean paired NFE reduction. Otherwise it stops with
the scientifically valid AdaBlock fallback.

The fixed post-pilot workload is 9,384 prompt-method generations: 2,628 plain AdaBlock runs and
6,756 instrumented runs. This is about 16% fewer generations than the prior v3 reuse profile;
the real A100-hour estimate still comes from the measured pilot rather than a guessed throughput.

Do not point `RC_PAG_REUSE_FROM` at the completed v1 run: v1 lacks the temporal-JS feature
evidence. Compatible v3/v4 exact-loop raw traces may be reused explicitly; v4 always refits the
single estimator. The pilot writes `compute_projection.json`, which is the hardware-specific GPU
hour estimate.

Monitor with:

```bash
squeue -u "$USER"
ls -lt logs/rc_pag
```
