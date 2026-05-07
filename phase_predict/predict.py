"""High-level inference API for phase-tuple prediction.

Example::

    from phase_predict.schema import ModelConfig, PhaseTuple
    from phase_predict.model import PhaseTransformer
    from phase_predict.predict import Predictor

    model = PhaseTransformer(ModelConfig())
    # ... train model or load weights ...
    predictor = Predictor(model, input_mean=dataset.input_mean, input_std=dataset.input_std)
    result = predictor.predict(context_window)
    print(result.predicted_tuple)
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from phase_predict.model import PhaseTransformer
from phase_predict.schema import ExtendedPhaseTuple, ModelConfig, PhaseTuple, PredictionResult

TupleLike = PhaseTuple | Sequence[int]


class Predictor:
    """Wraps a trained :class:`~phase_predict.model.PhaseTransformer` for inference.

    The predictor handles:
    - normalising the input context using training-set statistics,
    - running the forward pass,
    - rounding block logits (argmax) and stab logits (threshold count).

    Args:
        model:      a trained :class:`~phase_predict.model.PhaseTransformer`.
        input_mean: float tensor per-field mean for input normalisation.
        input_std:  float tensor per-field std for input normalisation.
        input_fields: ordered feature field names for ExtendedPhaseTuple.
        device:     compute device.
    """

    def __init__(
        self,
        model: PhaseTransformer,
        *,
        input_mean: torch.Tensor | None = None,
        input_std: torch.Tensor | None = None,
        input_fields: list[str] | None = None,
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

        in_ts = self.config.input_tuple_size
        self.input_mean = (
            input_mean if input_mean is not None else torch.zeros(in_ts)
        ).to(self.device)
        self.input_std = (
            input_std if input_std is not None else torch.ones(in_ts)
        ).to(self.device)
        # optional ordered list of input field names (used when coercing
        # ExtendedPhaseTuple objects during inference)
        self.input_fields = input_fields
        self.model.eval()

    @staticmethod
    def _coerce_tuple(value: TupleLike) -> tuple[int, int]:
        """Convert a PhaseTuple or tuple-like value to a 2-int tuple.

        Args:
            value: input tuple-like object. The first two entries are used.

        Returns:
            ``(block_size, refinement_steps)`` as integers.

        Raises:
            ValueError: if fewer than 2 values are provided.
            TypeError: if values are not integer-like.
        """
        if isinstance(value, PhaseTuple):
            return value.block_size, value.refinement_steps

        # support ExtendedPhaseTuple-like objects with a mapping of values
        try:
            from phase_predict.schema import ExtendedPhaseTuple

            if isinstance(value, ExtendedPhaseTuple):
                return int(value.values.get("block_size", 0)), int(value.values.get("refinement_steps", 0))
        except Exception:
            # if import fails or value is not ExtendedPhaseTuple, continue
            pass

        if len(value) < 2:
            msg = "Each input tuple must contain at least 2 values"
            raise ValueError(msg)

        try:
            block_size = int(value[0])
            refinement_steps = int(value[1])
        except (TypeError, ValueError) as exc:
            msg = "Input tuples must contain integer-like values"
            raise TypeError(msg) from exc

        return block_size, refinement_steps

    @torch.no_grad()
    def predict(self, context: Sequence[TupleLike]) -> PredictionResult:
        """Predict the next phase tuple from a context window.

        Args:
            context: sequence of tuple-like values.

        Returns:
            :class:`~phase_predict.schema.PredictionResult` with the
            predicted tuple and raw logits.
        """
        window_size = self.config.window_size
        in_tuple_size = self.config.input_tuple_size

        raw_in = torch.zeros(window_size, in_tuple_size, dtype=torch.float32)
        effective = context[-window_size:]
        for i, t in enumerate(effective):
            offset = window_size - len(effective)
            if isinstance(t, ExtendedPhaseTuple):
                if self.input_fields is not None:
                    vals = t.as_list(self.input_fields)
                else:
                    vals = list(t.values.values())
                for j in range(min(len(vals), in_tuple_size)):
                    raw_in[offset + i, j] = float(vals[j])
            else:
                try:
                    seq = list(t)
                    for j in range(min(len(seq), in_tuple_size)):
                        raw_in[offset + i, j] = float(seq[j])
                except Exception:
                    try:
                        b, r = self._coerce_tuple(t)
                        raw_in[offset + i, 0] = float(b)
                        if in_tuple_size > 1:
                            raw_in[offset + i, 1] = float(r)
                    except Exception:
                        pass

        normed = (raw_in.to(self.device) - self.input_mean) / self.input_std
        normed = normed.unsqueeze(0)

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

        in_mean = torch.tensor(checkpoint.get("input_mean", []), dtype=torch.float32)
        in_std = torch.tensor(checkpoint.get("input_std", []), dtype=torch.float32)
        input_fields = checkpoint.get("input_fields", None)

        return cls(
            model,
            input_mean=in_mean,
            input_std=in_std,
            input_fields=input_fields,
            device=target,
        )

    def save_checkpoint(self, path: str) -> None:
        """Persist model weights and normalisation statistics to *path*.

        Args:
            path: destination file path (conventionally ``*.pt``).
        """
        import dataclasses

        checkpoint = {
            "model_config": dataclasses.asdict(self.config),
            "model_state": {k: v.cpu() for k, v in self.model.state_dict().items()},
            "input_mean": getattr(self, "input_mean", torch.zeros(self.config.input_tuple_size)).cpu().tolist(),
            "input_std": getattr(self, "input_std", torch.ones(self.config.input_tuple_size)).cpu().tolist(),
            "input_fields": getattr(self, "input_fields", None),
        }
        torch.save(checkpoint, path)
