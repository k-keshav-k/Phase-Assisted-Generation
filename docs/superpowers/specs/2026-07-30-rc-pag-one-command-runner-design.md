# RC-PAG One-Command Runner Design

## Goal

Provide one command that allocates one NYU A100 and runs the complete registered RC-PAG
pipeline without manual stage submission:

```text
preflight -> pilot -> collect -> fit -> screen -> calibrate -> confirm -> report -> paper
```

The `fit`, `report`, and `paper` stages run on the CPU of the same allocated GPU node. This
intentionally favors operational simplicity over minimizing the time for which the A100 is
reserved.

## Interface

The user runs:

```bash
scripts/slurm/submit_rc_pag_all.sh
```

The launcher uses the existing environment variables for the project directory, output root,
Slurm account and partition, Singularity overlay, image, and scratch cache. It accepts one
additional wall-time variable, `RC_PAG_ALL_TIME`, whose conservative default is seven days.

The launcher validates every required local path before calling `sbatch`. It sets
`RC_PAG_ALLOW_CONFIRMATORY=1`; invoking the all-stage launcher is the user's explicit opt-in to
the confirmatory experiment.

## Worker behavior

A new `rc_pag_all_a100.sbatch` worker requests one A100, eight CPU cores, and 80 GB of host
memory. It enters the same Singularity and uv environment as the existing per-stage worker,
then invokes `scripts/run_rc_pag.py` once per stage.

GPU stages use `--device cuda`. CPU stages use `--device cpu`, but remain inside the same Slurm
allocation. Every invocation passes `--resume`, so rerunning the one-command launcher reuses
atomic prompt records and completed artifacts.

The worker checks each command's exit status before advancing. A nonzero status, including the
registered controlled-stop status used when confirmation is unsafe or futile, terminates the
pipeline and prevents later stages from running. This preserves the frozen calibration and
claim gates.

## Preemption and time limits

The worker retains the existing `--requeue` and `USR1` checkpoint behavior. On a preemption or
time-limit signal, it terminates the active stage cleanly, waits for atomic writes to settle,
and requeues the same Slurm job. The restarted worker begins at preflight, but `--resume` makes
completed stages and prompt records cheap to revisit.

No stage changes the protocol hash. Users must not edit the registered YAML after calibration
has started; changed experiments require a new output directory and config hash.

## Outputs and documentation

All artifacts remain under the existing deterministic run directory. Successful completion
writes the risk certificate, confirmatory records, report, claim audit, and paper manifest.
Building the final PDF with the official NeurIPS style remains a separate local command because
the style file is external to the repository.

A concise document, `docs/rc_pag_one_command.md`, will contain only prerequisites, environment
variables, the one launch command, monitoring commands, expected output location, and the
resume behavior.

## Verification

Tests will verify that the stage list and CPU/GPU device assignments are correct, confirmation
receives explicit authorization, and execution stops on the first failed stage. Shell syntax
checks will cover both new scripts. The existing RC-PAG test suite and a mock end-to-end run
will be executed before completion is reported.
