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

The default workshop profile keeps the complete 12-hypothesis calibration protocol, then
confirms three methods on both models: AdaBlock, the screened best nonlearned method, and the
certified history RC-PAG policy. It uses deterministic index-stratified samples of 500 GSM8K,
300 MATH-500, 100 MBPP, and 100 HumanEval prompts: 6,000 confirmatory generations instead of
17,920.

The log prints the config hash and exact results directory under:

```text
/scratch/$USER/rc_pag/artifacts/rc-pag-5197d981c0bf
```

If interrupted, run the same command again; completed prompt records are reused. Override the
per-attempt Slurm limit with `RC_PAG_ALL_TIME=24:00:00` if required by the partition.
The pilot's `compute_projection.json` is the authoritative hardware-specific runtime estimate;
48 hours is the default Slurm attempt length, and the same job requeues if more time is needed.

To restore the larger four-method protocol, set
`RC_PAG_CONFIG="$PROJECT_DIR/configs/experiments/rc_pag_neurips.yaml"` before submission.
