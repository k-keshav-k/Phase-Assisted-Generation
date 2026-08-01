# Run RC-PAG with one command

From the repository on the NYU cluster, set the shared paths once:

```bash
export PROJECT_DIR="$PWD"
export RC_PAG_OUTPUT_ROOT="/scratch/${USER}/rc_pag/artifacts"
export OVERLAY_PATH="/scratch/${USER}/overlay-25GB-500K.ext3"
export SIF_PATH="/scratch/${USER}/ubuntu-20.04.3.sif"
export RC_PAG_REUSE_FROM="/scratch/${USER}/rc_pag/artifacts/rc-pag-5197d981c0bf"
```

Submit the complete pipeline:

```bash
scripts/slurm/submit_rc_pag_all.sh
```

This single resumable A100 job runs `preflight`, `pilot`, `collect`, `fit`, `screen`,
`calibrate`, `confirm`, `report`, and `paper`. It stops if a required safety or statistical
gate fails. Monitor it with `squeue -u "$USER"` and `ls -lt logs/rc_pag`.

The v2 profile first checks exact no-stop AdaBlock parity. It reuses only the hash-validated
LLaDA local risk estimator; Dream traces are recollected because its decoder changed. It freezes
one tail-only policy per model, calibrates the two policies on end-to-end task harm, and evaluates
AdaBlock, the per-model best nonlearned method, and RC-PAG on fresh complements of the v1 sample:
500 GSM8K, 200 MATH-500, 100 MBPP, and 64 HumanEval prompts (5,184 generations).

The log prints the config hash and exact results directory under:

```text
/scratch/$USER/rc_pag/artifacts/rc-pag-e2d98174c4f9
```

If interrupted, run the same command again; completed prompt records are reused. Override the
per-attempt Slurm limit with `RC_PAG_ALL_TIME=24:00:00` if required by the partition.
The pilot's `compute_projection.json` is the authoritative hardware-specific runtime estimate;
48 hours is the default Slurm attempt length, and the same job requeues if more time is needed.

If the old run is elsewhere, change `RC_PAG_REUSE_FROM`. Omit it to retrain both estimators.
