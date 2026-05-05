from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase_predict.predict import Predictor
from phase_predict.schema import PhaseTuple, ExtendedPhaseTuple


CHECKPOINT_PATH = "output/ablations/medium_ws67_d64_h2_l1_dp0_lr1.0_bestval=0.488357.pt"
# CHECKPOINT_PATH = "output/ablations/medium_ws67_d64_h2_l1_dp0_lr1.0_bestval=0.109380.pt"

print("1. Initializing synthetic test data...")
# Create a repeating sequence of PhaseTuples: (block_size, refinement_steps)
tuple1 = [ExtendedPhaseTuple(values={
    "block_size": 15,
    "nfe": 4,
    "max_stab_step": 2
}), ExtendedPhaseTuple(values={
    "block_size": 16,
    "nfe": 7,
    "max_stab_step": 5
}), ExtendedPhaseTuple(values={
    "block_size": 16,
    "nfe": 9,
    "max_stab_step": 6
}), ExtendedPhaseTuple(values={
    "block_size": 16,
    "nfe": 11,
    "max_stab_step": 7
}), ExtendedPhaseTuple(values={
    "block_size": 14,
    "nfe": 8,
    "max_stab_step": 4
}), ExtendedPhaseTuple(values={
    "block_size": 15,
    "nfe": 3,
    "max_stab_step": 2
}), ExtendedPhaseTuple(values={
    "block_size": 8,
    "nfe": 3,
    "max_stab_step": 2
})]
tuple2 = [ExtendedPhaseTuple(values={
    "block_size": 26,
    "nfe": 4,
    "max_stab_step": 2
}), ExtendedPhaseTuple(values={
    "block_size": 14,
    "nfe": 9,
    "max_stab_step": 5
}), ExtendedPhaseTuple(values={
    "block_size": 7,
    "nfe": 4,
    "max_stab_step": 3
}), ExtendedPhaseTuple(values={
    "block_size": 8,
    "nfe": 4,
    "max_stab_step": 3
}), ExtendedPhaseTuple(values={
    "block_size": 15,
    "nfe": 3,
    "max_stab_step": 2
}), ExtendedPhaseTuple(values={
    "block_size": 8,
    "nfe": 3,
    "max_stab_step": 2
})]
tuple3 = [ExtendedPhaseTuple(values={'block_size': 18, 'nfe': 5, 'max_stab_step': 3}), ExtendedPhaseTuple(values={'block_size': 1, 'nfe': 1, 'max_stab_step': 0}), ExtendedPhaseTuple(values={'block_size': 15, 'nfe': 7, 'max_stab_step': 2}), ExtendedPhaseTuple(values={'block_size': 11, 'nfe': 8, 'max_stab_step': 4}), ExtendedPhaseTuple(values={'block_size': 11, 'nfe': 5, 'max_stab_step': 2}), ExtendedPhaseTuple(values={'block_size': 16, 'nfe': 7, 'max_stab_step': 3}), ExtendedPhaseTuple(values={'block_size': 1, 'nfe': 1, 'max_stab_step': 0}), ExtendedPhaseTuple(values={'block_size': 17, 'nfe': 3, 'max_stab_step': 0}), ExtendedPhaseTuple(values={'block_size': 1, 'nfe': 1, 'max_stab_step': 0}), ExtendedPhaseTuple(values={'block_size': 16, 'nfe': 6, 'max_stab_step': 2}), ExtendedPhaseTuple(values={'block_size': 16, 'nfe': 9, 'max_stab_step': 8}), ExtendedPhaseTuple(values={'block_size': 6, 'nfe': 3, 'max_stab_step': 0}), ExtendedPhaseTuple(values={'block_size': 16, 'nfe': 8, 'max_stab_step': 5}), ExtendedPhaseTuple(values={'block_size': 29, 'nfe': 5, 'max_stab_step': 1}), ExtendedPhaseTuple(values={'block_size': 1, 'nfe': 1, 'max_stab_step': 0}), ExtendedPhaseTuple(values={'block_size': 16, 'nfe': 9, 'max_stab_step': 7}), ExtendedPhaseTuple(values={'block_size': 9, 'nfe': 4, 'max_stab_step': 2}), ExtendedPhaseTuple(values={'block_size': 2, 'nfe': 1, 'max_stab_step': 0}), ExtendedPhaseTuple(values={'block_size': 29, 'nfe': 10, 'max_stab_step': 9}), ExtendedPhaseTuple(values={'block_size': 1, 'nfe': 1, 'max_stab_step': 0}), ExtendedPhaseTuple(values={'block_size': 1, 'nfe': 1, 'max_stab_step': 0}), ExtendedPhaseTuple(values={'block_size': 14, 'nfe': 1, 'max_stab_step': 0}), ExtendedPhaseTuple(values={'block_size': 1, 'nfe': 1, 'max_stab_step': 0}), ExtendedPhaseTuple(values={'block_size': 16, 'nfe': 10, 'max_stab_step': 7}), ExtendedPhaseTuple(values={'block_size': 16, 'nfe': 7, 'max_stab_step': 5}), ExtendedPhaseTuple(values={'block_size': 12, 'nfe': 3, 'max_stab_step': 1}), ExtendedPhaseTuple(values={'block_size': 28, 'nfe': 2, 'max_stab_step': 0}), ExtendedPhaseTuple(values={'block_size': 2, 'nfe': 1, 'max_stab_step': 0}), ExtendedPhaseTuple(values={'block_size': 1, 'nfe': 1, 'max_stab_step': 0}), ExtendedPhaseTuple(values={'block_size': 7, 'nfe': 1, 'max_stab_step': 0}), ExtendedPhaseTuple(values={'block_size': 2, 'nfe': 1, 'max_stab_step': 0}), ExtendedPhaseTuple(values={'block_size': 31, 'nfe': 1, 'max_stab_step': 0}), ExtendedPhaseTuple(values={'block_size': 2, 'nfe': 1, 'max_stab_step': 0}), ExtendedPhaseTuple(values={'block_size': 2, 'nfe': 1, 'max_stab_step': 0}), ExtendedPhaseTuple(values={'block_size': 29, 'nfe': 1, 'max_stab_step': 0}), ExtendedPhaseTuple(values={'block_size': 2, 'nfe': 1, 'max_stab_step': 0}), ExtendedPhaseTuple(values={'block_size': 1, 'nfe': 1, 'max_stab_step': 0}), ExtendedPhaseTuple(values={'block_size': 19, 'nfe': 1, 'max_stab_step': 0}), ExtendedPhaseTuple(values={'block_size': 1, 'nfe': 1, 'max_stab_step': 0}), ExtendedPhaseTuple(values={'block_size': 2, 'nfe': 2, 'max_stab_step': 0}), ExtendedPhaseTuple(values={'block_size': 29, 'nfe': 2, 'max_stab_step': 0}), ExtendedPhaseTuple(values={'block_size': 41, 'nfe': 2, 'max_stab_step': 1}), ExtendedPhaseTuple(values={'block_size': 1, 'nfe': 1, 'max_stab_step': 0}), ExtendedPhaseTuple(values={'block_size': 1, 'nfe': 1, 'max_stab_step': 0}), ExtendedPhaseTuple(values={'block_size': 16, 'nfe': 3, 'max_stab_step': 1}), ExtendedPhaseTuple(values={'block_size': 3, 'nfe': 1, 'max_stab_step': 0})]

print("\n2. Loading the checkpoint model from output/...")
predictor = Predictor.from_checkpoint(CHECKPOINT_PATH)

print("\n3. Running prediction inference with the full sequence...")
# Provide the entire tuple sequence as context; the predictor will use the
# most recent tuples needed by the checkpoint's configured context size.
context_sequence = tuple1[:]
print(*tuple1, sep="\n")
# Predict the next tuple
result = predictor.predict(context_sequence)
print(f"Predicted next tuple: {result.predicted_tuple}\n")

context_sequence = tuple2[:]
print(*tuple2, sep="\n")
# Predict the next tuple
result = predictor.predict(context_sequence)
print(f"Predicted next tuple: {result.predicted_tuple}\n")

context_sequence = tuple3[:]
print(*tuple3, sep="\n")
# Predict the next tuple
result = predictor.predict(context_sequence)
print(f"Predicted next tuple: {result.predicted_tuple}\n")