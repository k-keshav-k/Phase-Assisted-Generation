# Architecture

## Two-Track System

```mermaid
flowchart LR
    subgraph Training["Offline Predictor Training"]
        AD[AdaBlock LLaDA on GSM8K] --> TE[Block tuple extraction]
        TE --> SW[Sliding windows w=8 + z-score]
        SW --> PE[Transformer encoder 2×4×64]
        PE --> DH[Dual heads: classifier + ordinal]
        DH --> CK[Checkpoint]
    end

    subgraph Inference["Online PAG Scheduling"]
        direction TB
        PS[PAG Tuple Scheduler] --> LB{Per-block loop}
        LB --> EV[Evaluation vs AdaBlock]
        EV --> MT[Metrics]
    end

    CK -.->|"loaded at init"| PS
```

## Integration Sequence

### Predictor Training

```mermaid
sequenceDiagram
    participant AD as AdaBlock LLaDA
    participant TE as Tuple Extractor
    participant DS as PhaseSequenceDataset
    participant TR as PhaseTransformer
    participant CK as Checkpoint

    AD->>TE: raw generation traces
    TE->>DS: (block_size, nfe, conf, digit, delim) tuples
    DS->>TR: sliding-window sequences [w=8]
    TR->>TR: train classifier + ordinal heads
    TR->>CK: save weights, norm stats, fields
```

### PAG Scheduling Loop

```mermaid
sequenceDiagram
    participant CK as Checkpoint
    participant PS as PAGTupleScheduler
    participant PR as Predictor
    participant LD as LLaDA Decoder
    participant EV as Evaluator

    CK->>PS: loaded at init
    loop Per block
        PS->>PR: padded history [w=8]
        PR-->>PS: (predicted_b, predicted_r)
        PS->>LD: block_size, refinement_budget
        LD->>LD: threshold-based unmasking
        Note over LD: soft-cap: conf ≥ 0.8 or stable ≥ 2 → early exit
        LD-->>PS: realized tuple (b, nfe, conf, ...)
        PS->>PS: append to rolling context
    end
    PS->>EV: PAG completions + AdaBlock completions
    EV-->>EV: per-prompt comparison
```

## Notes

- The training and inference tracks share the same `phase_predict` module.
- The checkpoint is the only artifact that crosses the train/inference boundary.
- The `src/pag/` pipeline provides a structured experimentation skeleton with typed contracts and swappable implementations, running parallel to the offline tools.
