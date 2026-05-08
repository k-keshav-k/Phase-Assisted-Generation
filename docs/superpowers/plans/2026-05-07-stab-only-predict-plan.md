# Stabilising-Step-Only Prediction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove block-size prediction from `PhaseTransformer` so the model only predicts stabilising steps (`refinement_steps`) via ordinal regression, while keeping `block_size` as an input feature.

**Architecture:** Transformer encoder → pool last token → `Linear(d_model, num_stab_thresholds)` → stab logits. Remove `block_head` and conditional concat. Return single tensor from `forward()`. Dataset yields single target. Loss is only BCE on stab logits.

**Tech Stack:** Python 3.11, PyTorch, pytest

---

## File Structure

**Modified files (9):**
- `phase_predict/schema.py` — Remove `num_block_classes`, `output_tuple_size` from `ModelConfig`
- `phase_predict/model.py` — Remove `block_head`, simplify `stab_head`, single output
- `phase_predict/dataset.py` — Single-target yield, remove block target encoding
- `phase_predict/train.py` — Single-target loss, single model output
- `phase_predict/predict.py` — Single-output decode, simplified coercion
- `scripts/train_phase_predict.py` — Remove `--num-block-classes`, update `ModelConfig`
- `tests/phase_predict/test_model.py` — Single-output tests
- `tests/phase_predict/test_dataset.py` — Single-target assertions
- `tests/phase_predict/test_predict.py` — Update `raw_output` length, value assertions

**Data utilities (`data_utils.py`, `vomm.py`):** Unchanged. `PhaseTuple` keeps both fields for data loading compatibility.

---

### Task 1: Schema — Remove `num_block_classes` and `output_tuple_size` from `ModelConfig`

**Files:**
- Modify: `phase_predict/schema.py:62-127`

- [ ] **Step 1: Remove `num_block_classes` and `output_tuple_size` fields, remove their validation**

In `ModelConfig`:
- Remove `output_tuple_size: int = 2` (currently lines 93-94)
- Remove `num_block_classes: int = 128` (currently lines 97-98)
- In `__post_init__`: remove the `output_tuple_size` validation block (lines 119-121) and the `num_block_classes` validation block (lines 122-124)

Final `ModelConfig`:

```python
@dataclass(slots=True)
class ModelConfig:
    window_size: int = 8
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    input_tuple_size: int = 2
    tuple_size: int | None = None
    num_stab_thresholds: int = 83

    def __post_init__(self) -> None:
        if self.tuple_size is not None and self.input_tuple_size == 2:
            self.input_tuple_size = self.tuple_size

        if self.d_model % self.n_heads != 0:
            msg = (
                f"ModelConfig.d_model ({self.d_model}) must be divisible by "
                f"n_heads ({self.n_heads})"
            )
            raise ValueError(msg)
        if self.window_size < 1:
            msg = "ModelConfig.window_size must be >= 1"
            raise ValueError(msg)
        if self.input_tuple_size < 1:
            msg = "ModelConfig.input_tuple_size must be >= 1"
            raise ValueError(msg)
        if self.num_stab_thresholds < 1:
            msg = "ModelConfig.num_stab_thresholds must be >= 1"
            raise ValueError(msg)
```

- [ ] **Step 2: Run tests to verify schema change doesn't break existing model imports**

Run: `python -c "from phase_predict.schema import ModelConfig; m = ModelConfig(); print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add phase_predict/schema.py
git commit -m "refactor: remove num_block_classes and output_tuple_size from ModelConfig"
```

---

### Task 2: Model — Remove `block_head`, simplify `stab_head`, single output

**Files:**
- Modify: `phase_predict/model.py`

- [ ] **Step 1: Update `__init__` — remove block_head, simplify stab_head**

Remove `self.block_head` line and change `stab_input_dim`:

```python
# Remove this line (current line 124):
self.block_head = nn.Linear(config.d_model, config.num_block_classes)

# Change lines 125-126 from:
stab_input_dim = config.d_model + config.num_block_classes
self.stab_head = nn.Linear(stab_input_dim, config.num_stab_thresholds)

# To:
self.stab_head = nn.Linear(config.d_model, config.num_stab_thresholds)
```

- [ ] **Step 2: Update `forward()` — return single tensor**

Change from:
```python
def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ...
    block_logits = self.block_head(last)
    stab_features = torch.cat([last, block_logits], dim=-1)
    stab_logits = self.stab_head(stab_features)
    return block_logits, stab_logits
```

To:
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    ...
    stab_logits = self.stab_head(last)
    return stab_logits
```

- [ ] **Step 3: Update class docstring**

Replace:
```python
"""Transformer encoder with classification + ordinal heads.

Given a window of ``window_size`` past phase tuples the model produces:
  - block_logits:  logits over ``num_block_classes`` for block size
  - stab_logits:   ordinal logits over ``num_stab_thresholds`` for
                   max stabilizing step

The stab head is conditioned on block logits via concatenation.
"""
```

With:
```python
"""Transformer encoder with ordinal regression head.

Given a window of ``window_size`` past phase tuples the model produces
``stab_logits``: ordinal logits over ``num_stab_thresholds`` for the
max stabilizing step of the next tuple.
"""
```

- [ ] **Step 4: Quick smoke test — model instantiates and runs**

Run: `python -c "
from phase_predict.schema import ModelConfig
from phase_predict.model import PhaseTransformer
import torch
m = PhaseTransformer(ModelConfig())
x = torch.randn(2, 8, 2)
out = m(x)
print('Output shape:', out.shape)
print('Output dtype:', out.dtype)
"`

Expected:
```
Output shape: torch.Size([2, 83])
Output dtype: torch.float32
```

- [ ] **Step 5: Commit**

```bash
git add phase_predict/model.py
git commit -m "refactor: remove block_head, simplify stab_head to single Linear layer"
```

---

### Task 3: Dataset — Remove block target encoding, yield single target

**Files:**
- Modify: `phase_predict/dataset.py`

- [ ] **Step 1: Update `PhaseSequenceDataset.__init__`**

Change the target-creation block (lines 136-152):

From:
```python
if hasattr(raw_next, "values"):
    block_val = raw_next.values.get("block_size", 0)
    stab_val = raw_next.values.get("max_stab_step", raw_next.values.get("nfe", 0))
else:
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

To:
```python
if hasattr(raw_next, "values"):
    stab_val = raw_next.values.get("max_stab_step", raw_next.values.get("nfe", 0))
else:
    stab_val = raw_next.refinement_steps

n_thresh = model_config.num_stab_thresholds
stab_target = torch.zeros(n_thresh, dtype=torch.float32)
clamped = min(max(0, int(stab_val)), n_thresh)
if clamped > 0:
    stab_target[:clamped] = 1.0

self._windows.append((context_input, stab_target))
```

- [ ] **Step 2: Update `PhaseSequenceDataset.__getitem__`**

Change return type signature (line 157-159):

From:
```python
def __getitem__(self, idx: int) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    context, target = self._windows[idx]
    return context, target
```

To:
```python
def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    context, target = self._windows[idx]
    return context, target
```

- [ ] **Step 3: Update `PhaseFullSequenceDataset.__init__`**

Same target-creation change (lines 232-246):

From:
```python
if hasattr(raw_next, "values"):
    block_val = raw_next.values.get("block_size", 0)
    stab_val = raw_next.values.get("max_stab_step", raw_next.values.get("nfe", 0))
else:
    block_val = raw_next.block_size
    stab_val = raw_next.refinement_steps

block_target = torch.tensor(max(0, int(block_val) - 1), dtype=torch.long)
n_thresh = model_config.num_stab_thresholds
stab_target = torch.zeros(n_thresh, dtype=torch.float32)
clamped = min(max(0, int(stab_val)), n_thresh)
if clamped > 0:
    stab_target[:clamped] = 1.0

self._samples.append((context, (block_target, stab_target)))
```

To:
```python
if hasattr(raw_next, "values"):
    stab_val = raw_next.values.get("max_stab_step", raw_next.values.get("nfe", 0))
else:
    stab_val = raw_next.refinement_steps

n_thresh = model_config.num_stab_thresholds
stab_target = torch.zeros(n_thresh, dtype=torch.float32)
clamped = min(max(0, int(stab_val)), n_thresh)
if clamped > 0:
    stab_target[:clamped] = 1.0

self._samples.append((context, stab_target))
```

- [ ] **Step 4: Update `PhaseFullSequenceDataset.__getitem__`**

Same as Step 2 — change return type from `tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]` to `tuple[torch.Tensor, torch.Tensor]`.

- [ ] **Step 5: Quick smoke test**

Run: `python -c "
from phase_predict.schema import ModelConfig, PhaseTuple
from phase_predict.dataset import PhaseSequenceDataset
seq = [PhaseTuple(i % 8 + 1, i % 6) for i in range(20)]
ds = PhaseSequenceDataset(seq, ModelConfig(window_size=4))
x, t = ds[0]
print('Input shape:', x.shape)
print('Target shape:', t.shape)
print('Target dtype:', t.dtype)
"`

Expected:
```
Input shape: torch.Size([4, 2])
Target shape: torch.Size([83])
Target dtype: torch.float32
```

- [ ] **Step 6: Commit**

```bash
git add phase_predict/dataset.py
git commit -m "refactor: remove block target encoding, yield single stab target"
```

---

### Task 4: Training — Single-target loss, remove model output unpacking

**Files:**
- Modify: `phase_predict/train.py`

- [ ] **Step 1: Update `train_epoch`**

Change the data-unpacking and loss computation (lines 73-88):

From:
```python
for inputs, targets in loader:
    inputs = inputs.to(device)
    block_targets, stab_targets = targets
    block_targets = block_targets.to(device)
    stab_targets = stab_targets.to(device)
    optimizer.zero_grad()
    autocast_context = torch.amp.autocast("cuda", enabled=use_amp) if use_amp else nullcontext()
    with autocast_context:
        block_logits, stab_logits = model(inputs)
        loss_block = F.cross_entropy(block_logits, block_targets)
        loss_stab = F.binary_cross_entropy_with_logits(
            stab_logits, stab_targets,
            pos_weight=stab_pos_weight,
            reduction="mean",
        )
        loss = loss_block + loss_stab
```

To:
```python
for inputs, targets in loader:
    inputs = inputs.to(device)
    stab_targets = targets.to(device)
    optimizer.zero_grad()
    autocast_context = torch.amp.autocast("cuda", enabled=use_amp) if use_amp else nullcontext()
    with autocast_context:
        stab_logits = model(inputs)
        loss = F.binary_cross_entropy_with_logits(
            stab_logits, stab_targets,
            pos_weight=stab_pos_weight,
            reduction="mean",
        )
```

- [ ] **Step 2: Update `evaluate`**

Same change — from:
```python
for inputs, targets in loader:
    inputs = inputs.to(device)
    block_targets, stab_targets = targets
    block_targets = block_targets.to(device)
    stab_targets = stab_targets.to(device)
    autocast_context = torch.amp.autocast("cuda", enabled=use_amp) if use_amp else nullcontext()
    with autocast_context:
        block_logits, stab_logits = model(inputs)
        loss_block = F.cross_entropy(block_logits, block_targets)
        loss_stab = F.binary_cross_entropy_with_logits(
            stab_logits, stab_targets,
            pos_weight=stab_pos_weight,
            reduction="mean",
        )
        loss = loss_block + loss_stab
```

To:
```python
for inputs, targets in loader:
    inputs = inputs.to(device)
    stab_targets = targets.to(device)
    autocast_context = torch.amp.autocast("cuda", enabled=use_amp) if use_amp else nullcontext()
    with autocast_context:
        stab_logits = model(inputs)
        loss = F.binary_cross_entropy_with_logits(
            stab_logits, stab_targets,
            pos_weight=stab_pos_weight,
            reduction="mean",
        )
```

- [ ] **Step 3: Quick smoke test — trainer runs end-to-end**

Run: `python -c "
from phase_predict.schema import ModelConfig, PhaseTuple, TrainConfig
from phase_predict.dataset import PhaseSequenceDataset
from phase_predict.model import PhaseTransformer
from phase_predict.train import Trainer
seq = [PhaseTuple(i % 8 + 1, i % 6) for i in range(40)]
ds = PhaseSequenceDataset(seq, ModelConfig(window_size=4, d_model=16, n_heads=2, n_layers=1))
model = PhaseTransformer(ModelConfig(window_size=4, d_model=16, n_heads=2, n_layers=1))
trainer = Trainer(model, TrainConfig(max_epochs=3, batch_size=8, log_interval=0))
history = trainer.fit(ds)
print('Train losses:', len(history.train_losses))
print('Val losses:', len(history.val_losses))
"`

Expected: `Train losses: 3`, `Val losses: 3` (no errors)

- [ ] **Step 4: Commit**

```bash
git add phase_predict/train.py
git commit -m "refactor: remove block loss, single-target training loop"
```

---

### Task 5: Prediction — Single-output decode

**Files:**
- Modify: `phase_predict/predict.py`

Note: `_coerce_tuple` is used for input tensor construction (parsing context tuples),
not output decoding. It stays unchanged — the input format still includes `block_size`.

- [ ] **Step 1: Update `predict()` — single-output decode**

Change the model call and decoding (lines 158-175):

From:
```python
block_logits, stab_logits = self.model(normed)
block_logits = block_logits.squeeze(0)
stab_logits = stab_logits.squeeze(0)

block_pred = max(1, int(block_logits.argmax(dim=-1).item()) + 1)
stab_pred = int((torch.sigmoid(stab_logits) > 0.5).sum().item())

return PredictionResult(
    predicted_tuple=PhaseTuple(
        block_size=block_pred,
        refinement_steps=stab_pred,
    ),
    raw_output=[float(v) for v in block_logits],
    metadata={
        "window_size_used": len(effective),
        "num_stab_thresholds_active": stab_pred,
    },
)
```

To:
```python
stab_logits = self.model(normed)
stab_logits = stab_logits.squeeze(0)

stab_pred = int((torch.sigmoid(stab_logits) > 0.5).sum().item())

return PredictionResult(
    predicted_tuple=PhaseTuple(
        block_size=0,
        refinement_steps=stab_pred,
    ),
    raw_output=[float(v) for v in stab_logits],
    metadata={
        "window_size_used": len(effective),
        "num_stab_thresholds_active": stab_pred,
    },
)
```

- [ ] **Step 3: Quick smoke test — predictor runs**

Run: `python -c "
from phase_predict.schema import ModelConfig, PhaseTuple
from phase_predict.model import PhaseTransformer
from phase_predict.predict import Predictor
model = PhaseTransformer(ModelConfig())
p = Predictor(model)
ctx = [PhaseTuple(i % 8 + 1, i % 6) for i in range(4)]
result = p.predict(ctx)
print('Prediction:', result.predicted_tuple)
print('Raw output length:', len(result.raw_output))
"`

Expected:
```
Prediction: PhaseTuple(block_size=0, refinement_steps=...)
Raw output length: 83
```

- [ ] **Step 4: Commit**

```bash
git add phase_predict/predict.py
git commit -m "refactor: single-output prediction for stab-only model"
```

---

### Task 6: Training script — Remove `--num-block-classes`, update `ModelConfig`

**Files:**
- Modify: `scripts/train_phase_predict.py`

- [ ] **Step 1: Remove `--num-block-classes` argument**

Remove lines 303-304:
```python
parser.add_argument("--num-block-classes", type=int, default=128,
                    help="Number of block size classes for classification head.")
```

- [ ] **Step 2: Update `ModelConfig` construction**

At line 454-464, change from:
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

To:
```python
model_cfg = ModelConfig(
    window_size=effective_window_size,
    d_model=args.d_model,
    n_heads=args.n_heads,
    n_layers=args.n_layers,
    dropout=args.dropout,
    input_tuple_size=input_tuple_size,
    num_stab_thresholds=args.num_stab_thresholds,
)
```

- [ ] **Step 3: Commit**

```bash
git add scripts/train_phase_predict.py
git commit -m "refactor: remove --num-block-classes, update ModelConfig construction"
```

---

### Task 7: Tests — Update `test_model.py` for single output

**Files:**
- Modify: `tests/phase_predict/test_model.py`

- [ ] **Step 1: Update `_make_cfg` — remove `num_block_classes`, `output_tuple_size`**

Change:
```python
defaults = dict(window_size=4, d_model=16, n_heads=2, n_layers=1,
                input_tuple_size=2, output_tuple_size=2,
                num_block_classes=128, num_stab_thresholds=10, dropout=0.0)
```

To:
```python
defaults = dict(window_size=4, d_model=16, n_heads=2, n_layers=1,
                input_tuple_size=2, num_stab_thresholds=10, dropout=0.0)
```

- [ ] **Step 2: Rework `test_model_returns_two_outputs` to `test_model_returns_single_output`**

Replace:
```python
def test_model_returns_two_outputs(self) -> None:
    cfg = _make_cfg()
    model = PhaseTransformer(cfg)
    x = torch.randn(2, cfg.window_size, cfg.input_tuple_size)
    out = model(x)
    assert isinstance(out, tuple) and len(out) == 2
```

With:
```python
def test_model_returns_single_output(self) -> None:
    cfg = _make_cfg()
    model = PhaseTransformer(cfg)
    x = torch.randn(2, cfg.window_size, cfg.input_tuple_size)
    out = model(x)
    assert isinstance(out, torch.Tensor)
```

- [ ] **Step 3: Remove `test_block_logits_shape`**

Delete the entire `test_block_logits_shape` method.

- [ ] **Step 4: Update `test_single_sample_batch`**

From:
```python
def test_single_sample_batch(self) -> None:
    cfg = _make_cfg()
    model = PhaseTransformer(cfg)
    x = torch.randn(1, cfg.window_size, cfg.input_tuple_size)
    block_logits, stab_logits = model(x)
    assert block_logits.shape == (1, cfg.num_block_classes)
    assert stab_logits.shape == (1, cfg.num_stab_thresholds)
```

To:
```python
def test_single_sample_batch(self) -> None:
    cfg = _make_cfg()
    model = PhaseTransformer(cfg)
    x = torch.randn(1, cfg.window_size, cfg.input_tuple_size)
    out = model(x)
    assert out.shape == (1, cfg.num_stab_thresholds)
```

- [ ] **Step 5: Update `test_no_nan_in_output`**

From:
```python
def test_no_nan_in_output(self) -> None:
    cfg = _make_cfg()
    model = PhaseTransformer(cfg)
    x = torch.randn(4, cfg.window_size, cfg.input_tuple_size)
    block_logits, stab_logits = model(x)
    assert not torch.isnan(block_logits).any()
    assert not torch.isnan(stab_logits).any()
```

To:
```python
def test_no_nan_in_output(self) -> None:
    cfg = _make_cfg()
    model = PhaseTransformer(cfg)
    x = torch.randn(4, cfg.window_size, cfg.input_tuple_size)
    out = model(x)
    assert not torch.isnan(out).any()
```

- [ ] **Step 6: Update `test_eval_mode_deterministic`**

From:
```python
def test_eval_mode_deterministic(self) -> None:
    cfg = _make_cfg(dropout=0.1)
    model = PhaseTransformer(cfg)
    model.eval()
    x = torch.randn(3, cfg.window_size, cfg.input_tuple_size)
    with torch.no_grad():
        b1, s1 = model(x)
        b2, s2 = model(x)
    assert torch.allclose(b1, b2)
    assert torch.allclose(s1, s2)
```

To:
```python
def test_eval_mode_deterministic(self) -> None:
    cfg = _make_cfg(dropout=0.1)
    model = PhaseTransformer(cfg)
    model.eval()
    x = torch.randn(3, cfg.window_size, cfg.input_tuple_size)
    with torch.no_grad():
        o1 = model(x)
        o2 = model(x)
    assert torch.allclose(o1, o2)
```

- [ ] **Step 7: Update `test_gradients_flow`**

From:
```python
def test_gradients_flow(self) -> None:
    cfg = _make_cfg()
    model = PhaseTransformer(cfg)
    x = torch.randn(4, cfg.window_size, cfg.input_tuple_size)
    block_logits, stab_logits = model(x)
    loss = block_logits.sum() + stab_logits.sum()
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
```

To:
```python
def test_gradients_flow(self) -> None:
    cfg = _make_cfg()
    model = PhaseTransformer(cfg)
    x = torch.randn(4, cfg.window_size, cfg.input_tuple_size)
    out = model(x)
    loss = out.sum()
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
```

- [ ] **Step 8: Remove `test_causal_conditioning_stab_receives_block_gradients`**

Delete the entire method — no longer applicable.

- [ ] **Step 9: Update `test_custom_input_tuple_size`**

From:
```python
def test_custom_input_tuple_size(self) -> None:
    cfg = _make_cfg(input_tuple_size=12, d_model=32, n_heads=2)
    model = PhaseTransformer(cfg)
    x = torch.randn(2, cfg.window_size, 12)
    block_logits, stab_logits = model(x)
    assert block_logits.shape == (2, cfg.num_block_classes)
    assert stab_logits.shape == (2, cfg.num_stab_thresholds)
```

To:
```python
def test_custom_input_tuple_size(self) -> None:
    cfg = _make_cfg(input_tuple_size=12, d_model=32, n_heads=2)
    model = PhaseTransformer(cfg)
    x = torch.randn(2, cfg.window_size, 12)
    out = model(x)
    assert out.shape == (2, cfg.num_stab_thresholds)
```

- [ ] **Step 10: Final test file content check**

Run: `make lint` or `ruff check tests/phase_predict/test_model.py`

Expected: No lint errors.

- [ ] **Step 11: Commit**

```bash
git add tests/phase_predict/test_model.py
git commit -m "tests: update model tests for single-output PhaseTransformer"
```

---

### Task 8: Tests — Update `test_dataset.py` for single target

**Files:**
- Modify: `tests/phase_predict/test_dataset.py`

- [ ] **Step 1: Update `test_item_shapes`**

From:
```python
def test_item_shapes(self) -> None:
    seq = _make_sequence(20)
    cfg = ModelConfig(window_size=4, input_tuple_size=2)
    ds = PhaseSequenceDataset(seq, cfg)
    inp, target = ds[0]
    assert inp.shape == (4, 2)
    block_target, stab_target = target
    assert block_target.shape == ()
    assert block_target.dtype == torch.long
    assert stab_target.shape == (cfg.num_stab_thresholds,)
    assert stab_target.dtype == torch.float32
```

To:
```python
def test_item_shapes(self) -> None:
    seq = _make_sequence(20)
    cfg = ModelConfig(window_size=4, input_tuple_size=2)
    ds = PhaseSequenceDataset(seq, cfg)
    inp, target = ds[0]
    assert inp.shape == (4, 2)
    assert target.shape == (cfg.num_stab_thresholds,)
    assert target.dtype == torch.float32
```

- [ ] **Step 2: Remove `test_block_target_is_correct_class`**

Delete the entire method.

- [ ] **Step 3: Update `test_stab_target_first_elements_are_one`**

From:
```python
def test_stab_target_first_elements_are_one(self) -> None:
    seq = _make_sequence(20)
    cfg = ModelConfig(window_size=4)
    ds = PhaseSequenceDataset(seq, cfg, normalize=False)
    _, target = ds[0]
    _, stab_target = target
    assert stab_target[:4].sum().item() == 4.0
    assert stab_target[4:].sum().item() == 0.0
```

To:
```python
def test_stab_target_first_elements_are_one(self) -> None:
    seq = _make_sequence(20)
    cfg = ModelConfig(window_size=4)
    ds = PhaseSequenceDataset(seq, cfg, normalize=False)
    _, stab_target = ds[0]
    assert stab_target[:4].sum().item() == 4.0
    assert stab_target[4:].sum().item() == 0.0
```

- [ ] **Step 4: Run dataset tests**

Run: `python -m pytest tests/phase_predict/test_dataset.py -v`

Expected: All tests pass (except the removed one).

- [ ] **Step 5: Lint check**

Run: `ruff check tests/phase_predict/test_dataset.py`

Expected: No errors.

- [ ] **Step 6: Commit**

```bash
git add tests/phase_predict/test_dataset.py
git commit -m "tests: update dataset tests for single-target yield"
```

---

### Task 9: Tests — Update `test_predict.py` for stab-only prediction

**Files:**
- Modify: `tests/phase_predict/test_predict.py`

- [ ] **Step 1: Update `test_predicted_values_non_negative`**

Remove the `block_size` assertion:
```python
def test_predicted_values_non_negative(self) -> None:
    predictor, ds = _make_predictor()
    seq = _make_sequence(20)
    result = predictor.predict(seq[-4:])
    t = result.predicted_tuple
    assert t.refinement_steps >= 0
```

- [ ] **Step 2: Update `test_raw_output_length`**

Change the expected length from 128 (num_block_classes) to `num_stab_thresholds`:

```python
def test_raw_output_length(self) -> None:
    predictor, ds = _make_predictor()
    seq = _make_sequence(20)
    result = predictor.predict(seq[-4:])
    assert len(result.raw_output) == 83
```

(Note: 83 comes from ModelConfig default `num_stab_thresholds=83`)

- [ ] **Step 3: Run predict tests**

Run: `python -m pytest tests/phase_predict/test_predict.py -v`

Expected: All pass.

- [ ] **Step 4: Lint check**

Run: `ruff check tests/phase_predict/test_predict.py`

Expected: No errors.

- [ ] **Step 5: Commit**

```bash
git add tests/phase_predict/test_predict.py
git commit -m "tests: update predict tests for stab-only output"
```

---

### Task 10: Run full test suite and lint

- [ ] **Step 1: Run the full phase_predict test suite**

Run: `python -m pytest tests/phase_predict/ -v`

Expected: All tests pass.

- [ ] **Step 2: Run lint**

Run: `make lint`

Or: `ruff check phase_predict tests/phase_predict scripts/train_phase_predict.py`

Expected: No errors.

- [ ] **Step 3: Run format check**

Run: `make format` or `ruff format --check phase_predict tests/phase_predict scripts/train_phase_predict.py`

Expected: No changes needed (or auto-format if desired).

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "chore: lint and format after stab-only refactor"
```
