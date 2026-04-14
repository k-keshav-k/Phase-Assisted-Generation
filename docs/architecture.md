# Architecture

## Pipeline workflow

```mermaid
flowchart LR
    A[RunConfig + Dataset Samples] --> B[Baseline Stage]
    B --> C[Generation Traces]
    B --> D[Token Signals]
    B --> E[Baseline Completions]
    C --> F[Phase Analysis]
    D --> F
    F --> G[Phase Annotations]
    F --> H[Predictor Dataset]
    F --> I[Phase Predictions]
    E --> J[Scheduler]
    I --> J
    J --> K[Schedule Plans + Decisions]
    K --> L[Adaptive Decode]
    L --> M[Adaptive Completions]
    E --> N[Evaluation]
    M --> N
    N --> O[Comparison Records + Run Summaries]
```

## Integration sequence

```mermaid
sequenceDiagram
    participant O as Orchestration
    participant B as Baseline
    participant PA as Phase Analysis
    participant PR as Predictor
    participant S as Scheduler
    participant AD as Adaptive Decode

    O->>B: run_baseline(run_config, samples)
    B-->>O: traces + token_signals + completions + summary
    O->>PA: run_phase_analysis(run_config, baseline_artifacts)
    PA->>PR: build predictor dataset and phase predictions
    PR-->>PA: labels + predictions + metadata
    PA-->>O: phase_artifacts
    O->>S: run_adaptive_decoding(run_config, baseline_artifacts, phase_artifacts)
    S->>AD: schedule plans + decisions
    AD-->>S: adaptive completions
    S-->>O: adaptive_artifacts
    O->>O: evaluate_runs(run_config, baseline_artifacts, adaptive_artifacts)
```

## Notes

- The contracts package defines the only shared data shapes that all teams must honor.
- Module internals remain intentionally unconstrained.
- Each stage entrypoint accepts an optional implementation callable, so teams can swap internals without changing orchestration.
