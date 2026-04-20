"""High-level inference API for phase-tuple prediction.

Example::

    from phase_predict.schema import ModelConfig, PhaseTuple
    from phase_predict.model import PhaseTransformer
    from phase_predict.predict import Predictor

    model = PhaseTransformer(ModelConfig())
    # ... train model or load weights ...
    predictor = Predictor(model, mean=dataset.mean, std=dataset.std)
    result = predictor.predict(context_window)
    print(result.predicted_tuple)
"""

from __future__ import annotations

import torch

from phase_predict.model import PhaseTransformer
from phase_predict.schema import ModelConfig, PhaseTuple, PredictionResult


class Predictor:
    """Wraps a trained :class:`~phase_predict.model.PhaseTransformer` for inference.

    The predictor handles:
    - normalising the input context using training-set statistics,
    - running the forward pass,
    - denormalising and rounding the output to non-negative integers.

    Args:
        model:  a trained :class:`~phase_predict.model.PhaseTransformer`.
        mean:   float tensor of shape ``(tuple_size,)`` – per-field mean used
                during training (obtained from
                :attr:`~phase_predict.dataset.PhaseSequenceDataset.mean`).
        std:    float tensor of shape ``(tuple_size,)`` – per-field std used
                during training.
        device: compute device; defaults to the device of *model*'s first
                parameter, or CPU if the model has no parameters.
    """

    def __init__(
        self,
        model: PhaseTransformer,
        *,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.config: ModelConfig = model.config

        if device is not None:
            self.device = device
        else:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")

        ts = self.config.tuple_size
        self.mean = (mean if mean is not None else torch.zeros(ts)).to(self.device)
        self.std = (std if std is not None else torch.ones(ts)).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, context: list[PhaseTuple]) -> PredictionResult:
        """Predict the next phase tuple from a context window.

        Args:
            context: list of :class:`~phase_predict.schema.PhaseTuple` values
                     of length ``model_config.window_size``.  Shorter windows
                     are left-padded with zeros; longer windows are truncated
                     to the most recent ``window_size`` tuples.

        Returns:
            :class:`~phase_predict.schema.PredictionResult` with the
            predicted tuple and raw regression output.
        """
        window_size = self.config.window_size
        tuple_size = self.config.tuple_size

        # build (1, window_size, tuple_size) tensor
        raw = torch.zeros(window_size, tuple_size, dtype=torch.float32)
        # use the last window_size entries from context
        effective = context[-window_size:]
        for i, t in enumerate(effective):
            offset = window_size - len(effective)
            raw[offset + i] = torch.tensor(list(t)[:tuple_size], dtype=torch.float32)

        # normalise
        normed = (raw - self.mean.cpu()) / self.std.cpu()
        normed = normed.unsqueeze(0).to(self.device)  # (1, W, T)

        raw_pred = self.model(normed).squeeze(0)  # (T,)

        # denormalise
        denormed = raw_pred * self.std + self.mean  # (T,)

        # round to nearest non-negative integer for each field
        ints = [max(0, round(float(v))) for v in denormed.cpu().tolist()]

        # pad or truncate to exactly 2 fields for PhaseTuple
        while len(ints) < 2:
            ints.append(0)
        block_size, ref_steps = ints[0], ints[1]

        return PredictionResult(
            predicted_tuple=PhaseTuple(
                block_size=block_size,
                refinement_steps=ref_steps,
            ),
            raw_output=denormed.cpu().tolist(),
            metadata={"window_size_used": len(effective)},
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        *,
        device: torch.device | None = None,
    ) -> Predictor:
        """Load a :class:`Predictor` from a checkpoint saved by
        :meth:`save_checkpoint`.

        Args:
            path:   path to the ``.pt`` checkpoint file.
            device: target device; defaults to CPU.

        Returns:
            A ready-to-use :class:`Predictor`.
        """
        target = device or torch.device("cpu")
        checkpoint = torch.load(path, map_location=target, weights_only=True)

        config = ModelConfig(**checkpoint["model_config"])
        model = PhaseTransformer(config)
        model.load_state_dict(checkpoint["model_state"])
        model.to(target)

        mean = torch.tensor(checkpoint["mean"], dtype=torch.float32)
        std = torch.tensor(checkpoint["std"], dtype=torch.float32)

        return cls(model, mean=mean, std=std, device=target)

    def save_checkpoint(self, path: str) -> None:
        """Persist model weights and normalisation statistics to *path*.

        Args:
            path: destination file path (conventionally ``*.pt``).
        """
        import dataclasses

        checkpoint = {
            "model_config": dataclasses.asdict(self.config),
            "model_state": {k: v.cpu() for k, v in self.model.state_dict().items()},
            "mean": self.mean.cpu().tolist(),
            "std": self.std.cpu().tolist(),
        }
        torch.save(checkpoint, path)
