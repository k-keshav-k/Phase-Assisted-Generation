
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase_predict.predict import Predictor
from phase_predict.schema import PhaseTuple


CHECKPOINT_PATH = "output/phase_predict_model_checkpoint.pt"

print("1. Initializing synthetic test data...")
# Create a repeating sequence of PhaseTuples: (block_size, refinement_steps)
tuple1 = [PhaseTuple(15, 4), PhaseTuple(16, 7), PhaseTuple(16, 9), PhaseTuple(16, 11), PhaseTuple(14, 8), PhaseTuple(15, 3), PhaseTuple(8, 3), PhaseTuple(1, 1)]
tuple2 = [PhaseTuple(26, 4), PhaseTuple(1, 1), PhaseTuple(14, 9), PhaseTuple(7, 4), PhaseTuple(8, 4), PhaseTuple(15, 3), PhaseTuple(8, 3)]

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