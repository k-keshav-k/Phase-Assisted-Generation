# PhasePredict: Stabilising-Step-Only Prediction

**Date:** 2026-05-07

## Motivation

Remove block-size prediction from `PhaseTransformer`, keeping only ordinal-regression
prediction of stabilising steps (`refinement_steps`). Block size remains available as
an input feature but is no longer an output target. This simplifies the model,
reduces parameters, and focuses training on the single output that matters for the
downstream scheduler.

## Schema Changes

### `ModelConfig`

| Field | Change |
|---|---|
| `num_block_classes` | **Removed** — no longer used |
| `output_tuple_size` | **Removed** — single output, no need for explicit field |
| `num_stab_thresholds` | **Kept** — ordinal regression head size |
| `input_tuple_size` | **Kept** — still controls input feature count |
| All other fields | **Kept** — unchanged |

### `PhaseTuple`

**Unchanged.** Both `block_size` and `refinement_steps` fields are preserved.
`block_size` persists because it is a useful input signal. The model will consume
it as an input feature but will not predict it.

### `PredictionResult`

- `predicted_tuple`: `PhaseTuple` with `block_size=0` (placeholder) and
  `refinement_steps=<predicted_value>`
- `raw_output`: list of stab logits (length `num_stab_thresholds`) — the only raw
  output from the model
- `metadata`: unchanged

## Model Architecture

### Before (current)

```
Input → Linear(d_model) + PosEnc → TransformerEncoder
  → pool last token
  → Linear(d_model, 128) → block_logits          # classification head (REMOVED)
  → concat(pooled, block_logits) → Linear(d_model+128, 83) → stab_logits
Loss = CE(block_logits) + BCE(stab_logits)
```

### After (proposed)

```
Input → Linear(d_model) + PosEnc → TransformerEncoder
  → pool last token
  → Linear(d_model, num_stab_thresholds) → stab_logits
Loss = BCE(stab_logits)
```

Specific changes:

- `self.block_head` removed entirely
- `self.stab_head` changed from `Linear(d_model + num_block_classes, num_stab_thresholds)`
  to `Linear(d_model, num_stab_thresholds)` — no longer conditioned on block logits
- `forward()` returns a single tensor `(batch, num_stab_thresholds)` instead of
  a tuple of two tensors

## Dataset Changes

### `PhaseSequenceDataset` and `PhaseFullSequenceDataset`

- **Yield signature changes:**
  - Before: `(input, (block_target, stab_target))`
  - After:  `(input, stab_target)`
- Block-target creation (`block_target` tensor, `block_val` extraction) removed
- Stab-target creation unchanged
- Input tensor construction unchanged (still reads all `feature_fields` / both
  `PhaseTuple` fields)

### `build_windows()`

Unchanged — returns raw `(list[PhaseTuple], PhaseTuple)` pairs.

## Training Changes

### `train_epoch()` and `evaluate()`

- Remove `block_targets` unpacking from the data loader loop
- Remove `loss_block = F.cross_entropy(block_logits, block_targets)`
- Remove `loss = loss_block + loss_stab` — now just `loss = loss_stab`
- Model forward call: `stab_logits = model(inputs)` (single output)

### `Trainer`

- `_compute_stab_pos_weight()` unchanged (only touches stab targets)
- `fit()` unchanged (the DataLoader yields one target instead of two, but the
  trainer just passes it through)

## Inference Changes

### `Predictor._coerce_tuple()`

Simplified from extracting `(block_size, refinement_steps)` to extracting a single
integer. Still handles `PhaseTuple`, `ExtendedPhaseTuple`, and `Sequence` inputs
for flexibility.

### `Predictor.predict()`

- Model returns single tensor → decode only stab logits
- `PhaseTuple` constructed with `block_size=0, refinement_steps=stab_pred`
- `raw_output` contains stab logits (length `num_stab_thresholds`)

### Checkpoint save/load

Unchanged — `save_checkpoint` and `from_checkpoint` still work. The checkpoint
format just has fewer keys in `model_state_dict` (no `block_head` weights).

## Training Script Changes

- Remove `--num-block-classes` CLI argument
- Remove `--tuple-block-field` CLI argument
- Update `ModelConfig` construction: remove `num_block_classes` and
  `output_tuple_size` kwargs
- Update sanity-check prediction display (single output)

## Test Changes

### `test_model.py`

- Remove `test_block_logits_shape` (head no longer exists)
- Remove `test_causal_conditioning_stab_receives_block_gradients` (no longer
  applicable)
- Rework `test_model_returns_two_outputs` → `test_model_returns_single_output`
- Rework `test_single_sample_batch` — single output assertion
- Rework `test_gradients_flow` — single output
- Rework `test_custom_input_tuple_size` — single output assertion
- Keep `test_stab_logits_shape`, `test_no_nan_in_output`, `test_eval_mode_deterministic`

### `test_dataset.py`

- `test_item_shapes`: target is a single tensor (not a tuple of two). Assert
  `stab_target.shape == (num_stab_thresholds,)` directly
- `test_block_target_is_correct_class`: remove entirely
- `test_stab_target_first_elements_are_one`: unpack target directly (no block target)
- Keep all other tests unchanged

### `test_predict.py`

- `test_raw_output_length`: update expected length from `num_block_classes` (128) to
  `num_stab_thresholds`
- `test_predicted_values_non_negative`: remove `block_size` assertion, keep
  `refinement_steps` assertion
- All other tests unchanged (checkpoint save/load, deterministic, context
  padding/truncation)

### `test_train.py`

Unchanged — loss is still a scalar, trainer interface unchanged.

## Files in Scope

| File | Changes |
|---|---|
| `phase_predict/schema.py` | Remove `num_block_classes`, `output_tuple_size` from `ModelConfig` |
| `phase_predict/model.py` | Remove `block_head`, simplify `stab_head`, single output |
| `phase_predict/dataset.py` | Single-target yield, remove block target encoding |
| `phase_predict/train.py` | Single-target loss, single model output |
| `phase_predict/predict.py` | Single-output decode, simplified coercion |
| `phase_predict/__init__.py` | Possibly update `__all__` if `PhaseTuple` changes (it doesn't) |
| `scripts/train_phase_predict.py` | Remove block-specific CLI args, update ModelConfig |
| `tests/phase_predict/test_model.py` | Single-output tests |
| `tests/phase_predict/test_dataset.py` | Single-target assertions |
| `tests/phase_predict/test_predict.py` | `raw_output` length, value assertions |

## Out of Scope

- `data_utils.py` — unchanged (still constructs `PhaseTuple` with both fields)
- `vomm.py` — unchanged (baseline model, not affected)
- `tests/phase_predict/test_train.py` — unchanged
- `tests/phase_predict/test_data_utils.py` — unchanged
