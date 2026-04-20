"""Transformer-based sequence predictor for phase tuple prediction.

Given a window of previous (block_size, refinement_steps) tuples,
predict the next such tuple. Designed for use with phase_cpd trace data but fully
standalone as a sequence prediction library.
"""

from __future__ import annotations

from phase_predict.predict import Predictor
from phase_predict.schema import ModelConfig, PhaseTuple, PredictionResult, TrainConfig
from phase_predict.train import Trainer

__all__ = [
    "ModelConfig",
    "PhaseTuple",
    "PredictionResult",
    "Predictor",
    "Trainer",
    "TrainConfig",
]
