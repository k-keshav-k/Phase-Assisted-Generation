# phase_predict

A Transformer-based sequence predictor for **phase tuple prediction** in Phase-Assisted Generation (PAG).

## Overview

Given a sliding window of previous `(block_size, stabilizing_steps, refinement_steps)` integer tuples, `phase_predict` trains and runs a compact Transformer model that predicts the *next* tuple in the sequence.

This replaces the variable-order Markov predictor previously used in the pipeline.

---

## Data Format

Each **phase tuple** is a triplet of non-negative integers:

| Field | Description |
|-------|-------------|
| `block_size` | Number of tokens decoded together in one generation block |
| `stabilizing_steps` | Diffusion step at which the block's tokens first stabilised |
| `refinement_steps` | Total diffusion steps applied to the block |

Data can be extracted from `phase_cpd` `TraceRecord` objects using `phase_predict.data_utils`:

```python
from phase_cpd.catalog import load_trace_by_id
from phase_predict.data_utils import tuples_from_trace

trace = load_trace_by_id("prompt-001")
phase_tuples = tuples_from_trace(trace)           # one tuple per token
# or, with CPD breakpoints:
phase_tuples = tuples_from_trace(trace, breakpoints=[30, 60, 90])  # one per segment
```

---

## Model Choice

We evaluated four approaches:

| Architecture | Verdict | Reason |
|---|---|---|
| Variable-order Markov | ❌ Replaced | Only captures co-occurrence statistics; no temporal reasoning |
| LSTM / GRU | ⚠️ Viable | Sequential computation limits parallelism; weaker long-range dependencies |
| TCN | ⚠️ Viable | Good for long fixed-period signals; less flexible context handling |
| **Transformer encoder** | ✅ **Chosen** | Self-attention over entire context window; parallel training; SOTA performance |

The chosen model is a **compact 2-layer Transformer encoder** (`d_model=64`, 4 heads) with sinusoidal positional encoding and per-field MSE regression heads. It is:
- Fast to train on CPU; GPU-scalable via `torch.device("cuda")`
- Extensible: changing `ModelConfig.tuple_size` accommodates different input structures
- Stable: pre-norm (`norm_first=True`) and gradient clipping prevent training instability

---

## Quick Start

### Install dependencies

```bash
pip install torch numpy
```

Or add the `phase_predict` group to your environment:

```bash
uv sync --group phase_predict
```

### Train on synthetic data

```python
from phase_predict.schema import ModelConfig, TrainConfig, PhaseTuple
from phase_predict.dataset import PhaseSequenceDataset
from phase_predict.model import PhaseTransformer
from phase_predict.train import Trainer
from phase_predict.predict import Predictor

# create some example sequences
tuples = [PhaseTuple(4, 2, 3), PhaseTuple(8, 3, 4), PhaseTuple(4, 2, 3)] * 30

model_cfg = ModelConfig(window_size=4)
dataset = PhaseSequenceDataset(tuples, model_cfg)
model = PhaseTransformer(model_cfg)

trainer = Trainer(model, TrainConfig(max_epochs=50, log_interval=10))
history = trainer.fit(dataset)

predictor = Predictor(model, mean=dataset.mean, std=dataset.std)
result = predictor.predict(tuples[-4:])
print(result.predicted_tuple)   # PhaseTuple(block_size=..., ...)
```

### Save and load a checkpoint

```python
predictor.save_checkpoint("phase_predict.pt")
loaded = Predictor.from_checkpoint("phase_predict.pt")
```

### Train with real phase_cpd data

See `scripts/train_phase_predict.py` for a runnable training script using the traces in `phase_cpd/data/traces_real/`.

---

## Package Structure

```
phase_predict/
├── __init__.py      # public API: Predictor, Trainer, ModelConfig, …
├── schema.py        # PhaseTuple, ModelConfig, TrainConfig, PredictionResult
├── model.py         # PhaseTransformer (Transformer encoder + regression head)
├── dataset.py       # PhaseSequenceDataset, build_windows, split_dataset
├── train.py         # Trainer (training loop, early stopping)
├── predict.py       # Predictor (inference, checkpoint save/load)
├── data_utils.py    # extract PhaseTuples from phase_cpd TraceRecord objects
└── README.md        # this file
```

---

## Running Tests

```bash
pytest tests/phase_predict/
```
