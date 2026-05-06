"""Overfit test: train on 2 sequences until near-zero loss.

Run from repo root:
    uv run python scripts/overfit_test.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase_predict.data_utils import extended_tuple_sequences_from_phase_tuples_jsonl
from phase_predict.dataset import PhaseFullSequenceDataset
from phase_predict.model import PhaseTransformer
from phase_predict.schema import ModelConfig, TrainConfig
from phase_predict.train import Trainer


INPUT_FEATURES = [
    "block_size", "nfe", "mean_stab_step", "max_stab_step",
    "mean_ref_step", "max_ref_step", "mean_gap", "max_gap",
    "mean_top1_confidence", "min_top1_confidence",
    "digit_fraction", "delimiter_fraction",
]

rich_path = Path("traces/rich/stab_tuples_conf_train_rich.jsonl")
all_sequences = extended_tuple_sequences_from_phase_tuples_jsonl(
    rich_path,
    output_fields=("block_size", "max_stab_step"),
    input_feature_fields=INPUT_FEATURES,
)
print(f"Loaded {len(all_sequences)} sequences")

# Take 2 sequences for overfit test
train_seqs = all_sequences[:2]
print(f"Sequence 0 length: {len(train_seqs[0])}")
print(f"Sequence 1 length: {len(train_seqs[1])}")

# Print first few targets so we know what to expect
for i, seq in enumerate(train_seqs):
    target = seq[-1]
    block_val = target.values.get("block_size", 0)
    stab_val = target.values.get("max_stab_step", 0)
    print(f"  Seq {i} target: block_size={block_val}, max_stab_step={stab_val}")

cfg = ModelConfig(
    window_size=max(len(s) for s in train_seqs) - 1,
    d_model=64,
    n_heads=4,
    n_layers=2,
    dropout=0.0,
    input_tuple_size=len(INPUT_FEATURES),
    output_tuple_size=2,
    num_block_classes=128,
    num_stab_thresholds=83,
)

dataset = PhaseFullSequenceDataset(
    train_seqs,
    cfg,
    feature_fields=INPUT_FEATURES,
    output_fields=["block_size", "max_stab_step"],
)

model = PhaseTransformer(cfg)

train_cfg = TrainConfig(
    max_epochs=500,
    learning_rate=1e-3,
    batch_size=32,
    log_interval=20,
    patience=50,
    val_fraction=0.01,
)

trainer = Trainer(model, train_cfg, device="cpu")
print("\nOverfitting on 2 sequences...")
history = trainer.fit(dataset)

print(f"\nFinal train loss: {history.train_losses[-1]:.6f}")
print(f"Best train loss:  {min(history.train_losses):.6f}")
print(f"Epochs trained:   {len(history.train_losses)}")

# Verify: the model should have near-zero loss
best_loss = min(history.train_losses)
loss_ok = best_loss < 0.01

# Run predictions to check sanity
model.eval()
with torch.no_grad():
    for i in range(len(dataset)):
        inp, targets = dataset[i]
        block_target, stab_target = targets
        inp = inp.unsqueeze(0)
        block_logits, stab_logits = model(inp)
        block_pred = max(1, int(block_logits.argmax(dim=-1).item()) + 1)
        stab_pred = int((torch.sigmoid(stab_logits) > 0.5).sum().item())
        block_actual = int(block_target.item()) + 1
        stab_actual = int(stab_target.sum().item())  # sum of 1s = the actual value
        match = block_pred == block_actual and stab_pred == stab_actual
        print(f"  Sample {i}: pred=({block_pred:>3d}, {stab_pred:>2d}) actual=({block_actual:>3d}, {stab_actual:>2d}) {'✓' if match else '✗'}")

print(f"\nOverfit test: {'PASSED' if loss_ok else 'FAILED'} (best loss = {best_loss:.6f})")
