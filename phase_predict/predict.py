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

        # output stats (mean/std) are for the model outputs (output_tuple_size)
        out_ts = self.config.output_tuple_size
        self.mean = (mean if mean is not None else torch.zeros(out_ts)).to(self.device)
        self.std = (std if std is not None else torch.ones(out_ts)).to(self.device)

        # input stats are used to normalise inputs before feeding the model
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
            context: sequence of tuple-like values. Each item can be either
                     :class:`~phase_predict.schema.PhaseTuple` or a plain
                     tuple/list with at least two integer-like values.
                     Shorter windows are left-padded with zeros; longer
                     windows are truncated to the most recent
                     ``window_size`` tuples.

        Returns:
            :class:`~phase_predict.schema.PredictionResult` with the
            predicted tuple and raw regression output.
        """
        window_size = self.config.window_size
        out_tuple_size = self.config.output_tuple_size
        in_tuple_size = self.config.input_tuple_size

        # build input tensor of shape (window_size, in_tuple_size)
        raw_in = torch.zeros(window_size, in_tuple_size, dtype=torch.float32)
        effective = context[-window_size:]
        for i, t in enumerate(effective):
            offset = window_size - len(effective)
            # ExtendedPhaseTuple preferred: try to extract full input feature vector
            if isinstance(t, ExtendedPhaseTuple):
                if self.input_fields is not None:
                    vals = t.as_list(self.input_fields)
                else:
                    # fallback to dict value order (insertion order)
                    vals = list(t.values.values())
                for j in range(min(len(vals), in_tuple_size)):
                    raw_in[offset + i, j] = float(vals[j])
            else:
                # fallback: sequence-like inputs or PhaseTuple — coerce first fields
                try:
                    seq = list(t)
                    for j in range(min(len(seq), in_tuple_size)):
                        raw_in[offset + i, j] = float(seq[j])
                except Exception:
                    # last-resort: use coerced (block, refinement) into first two positions
                    try:
                        b, r = self._coerce_tuple(t)
                        raw_in[offset + i, 0] = float(b)
                        if in_tuple_size > 1:
                            raw_in[offset + i, 1] = float(r)
                    except Exception:
                        pass

        # normalise inputs using input stats
        normed = (raw_in - self.input_mean.cpu()) / self.input_std.cpu()
        normed = normed.unsqueeze(0).to(self.device)  # (1, W, in_T)

        raw_pred = self.model(normed).squeeze(0)  # (out_T,)

        # denormalise outputs
        denormed = raw_pred * self.std + self.mean  # (out_T,)

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

        mean = torch.tensor(checkpoint.get("mean", []), dtype=torch.float32)
        std = torch.tensor(checkpoint.get("std", []), dtype=torch.float32)
        in_mean = torch.tensor(checkpoint.get("input_mean", []), dtype=torch.float32)
        in_std = torch.tensor(checkpoint.get("input_std", []), dtype=torch.float32)
        input_fields = checkpoint.get("input_fields", None)

        return cls(
            model,
            mean=mean,
            std=std,
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
            "mean": self.mean.cpu().tolist(),
            "std": self.std.cpu().tolist(),
            "input_mean": getattr(self, "input_mean", torch.zeros(self.config.input_tuple_size)).cpu().tolist(),
            "input_std": getattr(self, "input_std", torch.ones(self.config.input_tuple_size)).cpu().tolist(),
            "input_fields": getattr(self, "input_fields", None),
        }
        torch.save(checkpoint, path)
