
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
tuples = [PhaseTuple(1, 3), PhaseTuple(10, 4), PhaseTuple(100, 3)] * 300

print("\n2. Loading the checkpoint model from output/...")
predictor = Predictor.from_checkpoint(CHECKPOINT_PATH)

print("\n3. Running prediction inference with the full sequence...")
# Provide the entire tuple sequence as context; the predictor will use the
# most recent tuples needed by the checkpoint's configured context size.
context_sequence = tuples[:]
print(f"Context sequence length: {len(context_sequence)}")

# Predict the next tuple
result = predictor.predict(context_sequence)
print(f"\n✅ Predicted next tuple: {result.predicted_tuple}")