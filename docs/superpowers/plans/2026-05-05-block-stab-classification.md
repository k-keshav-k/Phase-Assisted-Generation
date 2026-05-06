# Classification Transformer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the MSE regression PhaseTransformer with a classification head (block_size) + ordinal head (max_stab_step) conditioned on predicted block logits.

**Architecture:** Same 2-layer encoder (d_model=64). Two heads: `Linear(d_model → 128)` for block classification, `Linear(d_model+128 → 83)` for stab ordinal thresholds. Combined loss: CE(block) + mean-BCE(stab).

**Spec:** `docs/superpowers/specs/2026-05-05-block-stab-classification-design.md`

---

## File Structure

| File | What changes |
|------|-------------|
| `phase_predict/schema.py` | Add `num_block_classes=128`, `num_stab_thresholds=83` to `ModelConfig` |
| `phase_predict/model.py` | Two heads, stab conditioned on block logits, returns `(block_logits, stab_logits)` |
| `phase_predict/dataset.py` | `PhaseSequenceDataset` — targets become `(class_id, ordinal_vector)` tuple |
| `phase_predict/dataset.py` | `PhaseFullSequenceDataset` — same multi-target change |
| `phase_predict/train.py` | Combined CE+BCE loss, updated `train_epoch` and `evaluate` |
| `phase_predict/predict.py` | `Predictor.predict` — argmax for block, threshold-count for stab |
| `scripts/train_phase_predict.py` | Wire new config, point to `traces/rich/` data |
| `tests/phase_predict/test_model.py` | New: test forward shapes, gradient flow, conditioning |
| `tests/phase_predict/test_predict.py` | Existing — update `_make_predictor` and verify save/load |


### Task 1: Schema — Add head dimensions to ModelConfig

**Files:**
- Modify: `phase_predict/schema.py:65-120`

- [ ] **Add fields and validation**

```python
@dataclass(slots=True)
class ModelConfig:
    # ... existing fields (window_size, d_model, n_heads, n_layers, dropout, input_tuple_size, output_tuple_size) ...
    num_block_classes: int = 128
    num_stab_thresholds: int = 83

    def __post_init__(self) -> None:
        # ... existing validation ...
        if self.num_block_classes < 1:
            msg = "num_block_classes must be >= 1"
            raise ValueError(msg)
        if self.num_stab_thresholds < 1:
            msg = "num_stab_thresholds must be >= 1"
            raise ValueError(msg)
```

- [ ] **Verify existing tests still pass**

Run: `python -m pytest tests/phase_predict/ -v`

- [ ] **Commit**

```bash
git add phase_predict/schema.py
git commit -m "feat(schema): add num_block_classes and num_stab_thresholds to ModelConfig"
```


### Task 2: Model — Two heads with conditioning

**Files:**
- Modify: `phase_predict/model.py:87-170`
- Create: `tests/phase_predict/test_model.py`

- [ ] **Write tests for new model architecture**

```python
# tests/phase_predict/test_model.py
import torch
from phase_predict.model import PhaseTransformer
from phase_predict.schema import ModelConfig


def test_model_returns_two_outputs():
    cfg = ModelConfig(d_model=16, n_heads=2, n_layers=1)
    model = PhaseTransformer(cfg)
    x = torch.randn(2, cfg.window_size, cfg.input_tuple_size)
    out = model(x)
    assert isinstance(out, tuple) and len(out) == 2


def test_block_logits_shape():
    cfg = ModelConfig(d_model=16, n_heads=2, n_layers=1)
    model = PhaseTransformer(cfg)
    x = torch.randn(4, cfg.window_size, cfg.input_tuple_size)
    block_logits, _ = model(x)
    assert block_logits.shape == (4, 128)


def test_stab_logits_shape():
    cfg = ModelConfig(d_model=16, n_heads=2, n_layers=1)
    model = PhaseTransformer(cfg)
    x = torch.randn(4, cfg.window_size, cfg.input_tuple_size)
    _, stab_logits = model(x)
    assert stab_logits.shape == (4, 83)


def test_stab_logits_depend_on_block_logits():
    cfg = ModelConfig(d_model=16, n_heads=2, n_layers=1)
    model = PhaseTransformer(cfg)
    x1 = torch.randn(1, cfg.window_size, cfg.input_tuple_size)
    x2 = x1 * 10
    _, s1 = model(x1)
    _, s2 = model(x2)
    assert not torch.allclose(s1, s2)


def test_causal_conditioning_gradient_flow():
    cfg = ModelConfig(d_model=16, n_heads=2, n_layers=1)
    model = PhaseTransformer(cfg)
    x = torch.randn(2, cfg.window_size, cfg.input_tuple_size)
    block_logits, stab_logits = model(x)
    loss = stab_logits.sum()
    loss.backward()
    grads = model.block_head.weight.grad
    assert grads is not None
    assert torch.any(grads != 0)
```

- [ ] **Run tests — expect AttributeError (model has no block_head yet)**

Run: `python -m pytest tests/phase_predict/test_model.py -v`

- [ ] **Implement the new model forward**

```python
class PhaseTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.input_projection = nn.Linear(config.input_tuple_size, config.d_model)
        self.pos_encoding = _SinusoidalPositionalEncoding(
            d_model=config.d_model, max_len=config.window_size + 1,
            dropout=config.dropout,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model, nhead=config.n_heads,
            dim_feedforward=config.d_model * 4, dropout=config.dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

        self.block_head = nn.Linear(config.d_model, config.num_block_classes)
        stab_input_dim = config.d_model + config.num_block_classes
        self.stab_head = nn.Linear(stab_input_dim, config.num_stab_thresholds)
        self._init_weights()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.input_projection(x)
        emb = self.pos_encoding(emb)
        encoded = self.encoder(emb)
        last = encoded[:, -1, :]
        block_logits = self.block_head(last)
        stab_features = torch.cat([last, block_logits], dim=-1)
        stab_logits = self.stab_head(stab_features)
        return block_logits, stab_logits

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
```

- [ ] **Run tests — expect 5 PASS**

Run: `python -m pytest tests/phase_predict/test_model.py -v`

- [ ] **Commit**

```bash
git add phase_predict/model.py tests/phase_predict/test_model.py
git commit -m "feat(model): classification head for block_size, ordinal head for stab_steps"
```


### Task 3: Dataset — Build classification + ordinal targets

**Files:**
- Modify: `phase_predict/dataset.py`
- Note: both `PhaseSequenceDataset` and `PhaseFullSequenceDataset` need updates

- [ ] **Update PhaseSequenceDataset**

Store `model_config` on self. In `__init__`, after building normalized input tensors, build multi-target windows:

```python
class PhaseSequenceDataset(Dataset):
    def __init__(self, sequence, model_config, *, normalize=True, stats=None,
                 feature_fields=None, output_fields=None):
        self.window_size = model_config.window_size
        self.input_tuple_size = model_config.input_tuple_size
        self.output_tuple_size = model_config.output_tuple_size
        self.model_config = model_config
        self.feature_fields = feature_fields
        self.output_fields = output_fields

        # Build input_raw, output_raw, normalization stats (same as current code) ...

        # Build windows with multi-target
        self._windows: list[tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]] = []
        for i in range(len(sequence) - self.window_size):
            context_input = input_norm[i : i + self.window_size]
            raw_next = sequence[i + self.window_size]

            if hasattr(raw_next, 'values'):  # ExtendedPhaseTuple
                block_val = raw_next.values.get('block_size', 0)
                stab_val = raw_next.values.get('max_stab_step',
                            raw_next.values.get('nfe', 0))
            else:  # PhaseTuple
                block_val = raw_next.block_size
                stab_val = raw_next.refinement_steps

            block_target = torch.tensor(max(0, int(block_val) - 1), dtype=torch.long)

            n_thresh = model_config.num_stab_thresholds
            stab_target = torch.zeros(n_thresh, dtype=torch.float32)
            clamped = min(max(0, int(stab_val)), n_thresh)
            if clamped > 0:
                stab_target[:clamped] = 1.0

            self._windows.append((context_input, (block_target, stab_target)))
```

- [ ] **Update PhaseFullSequenceDataset**

Same pattern: store `model_config`, build multi-target in constructor, make `__getitem__` return `(input, (block_target, stab_target))`. Build targets from `raw_next = sequence[-1]` (the target tuple).

- [ ] **Update `_make_predictor` in tests**

The predictor test creates a `PhaseSequenceDataset` — verify it works with the new target format. The predictor test primarily tests `Predictor.predict`, which doesn't depend on dataset output format directly.

Run: `python -m pytest tests/phase_predict/test_predict.py -v`

- [ ] **Commit**

```bash
git add phase_predict/dataset.py
git commit -m "feat(dataset): multi-target (class_id, ordinal_vector) for new heads"
```


### Task 4: Training — Combined CE + mean-BCE loss

**Files:**
- Modify: `phase_predict/train.py`

- [ ] **Update train_epoch and evaluate**

Add `import torch.nn.functional as F` at the top. Update the forward/loss section in both `train_epoch` and `evaluate`:

```python
block_logits, stab_logits = model(inputs)
block_targets, stab_targets = targets
block_targets = block_targets.to(device)
stab_targets = stab_targets.to(device)

loss_block = F.cross_entropy(block_logits, block_targets)
loss_stab = F.binary_cross_entropy_with_logits(stab_logits, stab_targets, reduction='mean')
loss = loss_block + loss_stab
```

The rest of the training loop (optimizer, gradient clipping, scaler, early stopping) stays unchanged.

- [ ] **Run existing predictor tests**

Run: `python -m pytest tests/phase_predict/test_predict.py -v`

- [ ] **Commit**

```bash
git add phase_predict/train.py
git commit -m "feat(train): combined cross-entropy + ordinal BCE loss"
```


### Task 5: Predictor — Argmax block, threshold-count stab

**Files:**
- Modify: `phase_predict/predict.py:127-201`

- [ ] **Update Predictor.predict**

Replace the single-head forward with multi-head forward and new rounding logic:

```python
@torch.no_grad()
def predict(self, context):
    window_size = self.config.window_size
    in_tuple_size = self.config.input_tuple_size

    raw_in = torch.zeros(window_size, in_tuple_size, dtype=torch.float32)
    effective = context[-window_size:]
    for i, t in enumerate(effective):
        offset = window_size - len(effective)
        if isinstance(t, ExtendedPhaseTuple) and self.input_fields is not None:
            vals = t.as_list(self.input_fields)
        else:
            try:
                vals = list(t)
            except Exception:
                b, r = self._coerce_tuple(t)
                vals = [b, r]
        for j in range(min(len(vals), in_tuple_size)):
            raw_in[offset + i, j] = float(vals[j])

    normed = (raw_in - self.input_mean.cpu()) / self.input_std.cpu()
    normed = normed.unsqueeze(0).to(self.device)

    block_logits, stab_logits = self.model(normed)
    block_logits = block_logits.squeeze(0)
    stab_logits = stab_logits.squeeze(0)

    block_pred = max(1, int(block_logits.argmax(dim=-1).item()) + 1)
    stab_pred = int((torch.sigmoid(stab_logits) > 0.5).sum().item())

    return PredictionResult(
        predicted_tuple=PhaseTuple(block_size=block_pred, refinement_steps=stab_pred),
        raw_output=[float(v) for v in block_logits],
        metadata={"num_stab_thresholds_active": stab_pred},
    )
```

No changes to `from_checkpoint` or `save_checkpoint` needed — `dataclasses.asdict(self.config)` already serializes the new fields.

- [ ] **Check that `_coerce_tuple` works with the updated model output format**

Look at `PhaseFullSequenceDataset` to see if `_make_predictor` or any other test helper needs updating for the output to be `(block_logits, stab_logits)` tuple.

Run: `python -m pytest tests/phase_predict/test_predict.py -v`

- [ ] **Commit**

```bash
git add phase_predict/predict.py
git commit -m "feat(predict): argmax for block_size, threshold-count for stab_steps"
```


### Task 6: Training script — Point to rich data, update config

**Files:**
- Modify: `scripts/train_phase_predict.py:227-530`

- [ ] **Update default paths and args**

```python
parser.add_argument("--train-jsonl", type=Path,
    default=Path("traces/rich/stab_tuples_conf_train_rich.jsonl"))
parser.add_argument("--test-jsonl", type=Path,
    default=Path("traces/rich/stab_tuples_conf_test_rich.jsonl"))
parser.add_argument("--input-features", type=str, nargs="+",
    default=["block_size", "nfe", "mean_stab_step", "max_stab_step",
             "mean_ref_step", "max_ref_step", "mean_gap", "max_gap",
             "mean_top1_confidence", "min_top1_confidence",
             "digit_fraction", "delimiter_fraction"])
parser.add_argument("--tuple-second-field", type=str, default="max_stab_step")
parser.add_argument("--num-block-classes", type=int, default=128)
parser.add_argument("--num-stab-thresholds", type=int, default=83)
```

Pass new args into `ModelConfig`:
```python
model_cfg = ModelConfig(
    window_size=effective_window_size,
    d_model=args.d_model,
    n_heads=args.n_heads,
    n_layers=args.n_layers,
    dropout=args.dropout,
    input_tuple_size=input_tuple_size,
    output_tuple_size=2,
    num_block_classes=args.num_block_classes,
    num_stab_thresholds=args.num_stab_thresholds,
)
```

- [ ] **Quick smoke test**

Run: `python scripts/train_phase_predict.py --epochs 1 --device cpu`

Expected: trains 1 epoch without errors, saves a checkpoint with new config.

- [ ] **Commit**

```bash
git add scripts/train_phase_predict.py
git commit -m "feat(scripts): point to rich trace data, 12 input features, new head config"
```


### Task 7: Full training run

- [ ] **Train on GPU**

```bash
python scripts/train_phase_predict.py \
  --epochs 100 \
  --device cuda \
  --learning-rate 1e-3 \
  --output output/classifier_model.pt
```

Expected: validation loss decreases over epochs. Best checkpoint saved.

- [ ] **Sanity-check predictions**

Run a quick comparison on a few test sequences to verify the classifier produces sensible outputs (block_size should be 1 or 16 most often, not 8-9).

- [ ] **Commit**

```bash
git add output/classifier_model_bestval=*.pt
git commit -m "feat: trained classification transformer on rich traces"
```
