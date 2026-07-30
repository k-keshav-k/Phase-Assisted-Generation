# Run RC-PAG with one command

From the repository on the NYU cluster, set the shared paths once:

```bash
export PROJECT_DIR="$PWD"
export RC_PAG_OUTPUT_ROOT="/scratch/${USER}/rc_pag/artifacts"
export OVERLAY_PATH="/scratch/${USER}/overlay-25GB-500K.ext3"
export SIF_PATH="/scratch/${USER}/ubuntu-20.04.3.sif"
```

Submit the complete pipeline:

```bash
scripts/slurm/submit_rc_pag_all.sh
```

This single resumable A100 job runs `preflight`, `pilot`, `collect`, `fit`, `screen`,
`calibrate`, `confirm`, `report`, and `paper`. It stops if a required safety or statistical
gate fails. Monitor it with `squeue -u "$USER"` and `ls -lt logs/rc_pag`.

Results are written to:

```text
/scratch/$USER/rc_pag/artifacts/rc-pag-c855954c1a30
```

If interrupted, run the same command again; completed prompt records are reused. Override the
per-attempt Slurm limit with `RC_PAG_ALL_TIME=24:00:00` if required by the partition.
