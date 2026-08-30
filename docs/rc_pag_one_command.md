# Run EC-PAG v9

From the repository on the HPC cluster, run exactly:

```bash
bash scripts/slurm/submit_rc_pag_all.sh
```

This submits one resumable 48-hour, one-A100 job and runs every CPU and GPU stage. The run is
`artifacts/rc_pag/rc-pag-588204cfb482` for the frozen v9 config.

The job automatically reuses only the 32 paired AdaBlock references/model from
`artifacts/rc_pag/rc-pag-d36b982c2388`. It regenerates the numerical audit, held-out pilot,
tuning, calibration, and confirmation evidence.

V9 continues past its 32-prompt audit and fresh 64-prompt pilot only if both models have exact
token/state trajectories, complete numerical-guard evidence, no evaluated-row increase, and a
paired latency-reduction lower bound above 5%. The same checks are repeated on 150 tuning and 500
calibration prompts/model before the 5,184 confirmation generations.

The earlier real pilot averaged about 3.1 seconds per AdaBlock prompt. Budget roughly 8--15 A100
hours for v9; the 48-hour request leaves room for cache setup and workload variation. After the
pilot, use the run-specific estimate in:

```bash
cat artifacts/rc_pag/rc-pag-588204cfb482/compute_projection.json
```

Monitor and resume with the same command:

```bash
squeue -u "$USER"
tail -f logs/rc_pag/*_rc_pag.out
bash scripts/slurm/submit_rc_pag_all.sh
```

Hugging Face mode defaults to `auto`: missing pinned assets are downloaded once, verified, then
the experiment runs offline. Use `RC_PAG_HF_MODE=offline` only when the shared cache is already
complete.

Final evidence is in `equivalence_audit.json`, `equivalence_pilot.json`,
`readiness_audit.json`, `risk_certificate.json`, and `report/claim_audit.json`. A controlled stop
is intentional: it prevents expensive later stages or paper rendering when a preregistered gate
fails.
