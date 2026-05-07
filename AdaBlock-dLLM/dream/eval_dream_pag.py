from __future__ import annotations

import json
import logging
import os
import time
import types

import torch
import transformers
from eval_dream_adablock import Dream as AdaBlockDream
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from lm_eval.models.utils import get_dtype
from model.generation_utils_pag import DreamGenerationMixin
from model.modeling_dream import DreamModel
from pag_predictor import PAGTupleScheduler

eval_logger = logging.getLogger(__name__)


@register_model("dream")
class Dream(AdaBlockDream):
    def __init__(
        self,
        pretrained: str | transformers.PreTrainedModel,
        batch_size: int | str | None = 1,
        device: str | None = "cuda",
        dtype: str | torch.dtype | None = "auto",
        max_new_tokens: int | None = 128,
        max_length: int | None = 2048,
        add_bos_token: bool | None = False,
        nll_type: str | None = "mc",
        log_type: str | None = "ftb",
        mc_num: int | None = 128,
        classifier_free_guidance: float | None = 1.0,
        sampling_eps: float | None = 1e-3,
        diffusion_steps: int | None = 128,
        trust_remote_code: bool | None = True,
        parallelize: bool | None = False,
        autogptq: bool | str | None = False,
        temperature: float | None = 0.0,
        top_p: float | None = None,
        top_k: float | None = None,
        alg: str | None = "confidence_threshold",
        alg_temp: float | None = 0.0,
        escape_until: bool | None = False,
        threshold: float | None = 0.9,
        apply_chat_template: bool | None = False,
        use_cache: bool | None = False,
        dual_cache: bool | None = False,
        block_length: int | None = 32,
        predictor_ckpt: str | None = None,
        seed_block_length: int | None = None,
        seed_refinement_steps: int | None = None,
        predictor_device: str | None = "cpu",
        max_block_length: int | None = None,
        max_refinement_steps: int | None = None,
        min_refinement_steps: int | None = 3,
        context_seed_block_length: int | None = None,
        context_seed_stabilizing_steps: int | None = None,
        delimiter_threshold: float | None = None,
        save_dir: str | None = None,
        min_block_length: int = 4,
        refinement_step_offset: int = 1,
        **kwargs,
    ) -> None:
        del delimiter_threshold
        if predictor_ckpt is None:
            raise ValueError("predictor_ckpt is required for PAG generation")
        if seed_block_length is None or seed_refinement_steps is None:
            raise ValueError("seed_block_length and seed_refinement_steps are required")

        super().__init__(
            pretrained=pretrained,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
            max_length=max_length,
            add_bos_token=add_bos_token,
            nll_type=nll_type,
            log_type=log_type,
            mc_num=mc_num,
            classifier_free_guidance=classifier_free_guidance,
            sampling_eps=sampling_eps,
            diffusion_steps=diffusion_steps,
            trust_remote_code=trust_remote_code,
            parallelize=parallelize,
            autogptq=autogptq,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            alg=alg,
            alg_temp=alg_temp,
            escape_until=escape_until,
            threshold=threshold,
            apply_chat_template=apply_chat_template,
            use_cache=use_cache,
            dual_cache=dual_cache,
            block_length=block_length,
            delimiter_threshold=None,
            save_dir=save_dir,
            **kwargs,
        )
        if not hasattr(self, "_rank"):
            self._rank = 0
            self._world_size = 1

        self.predictor_ckpt = predictor_ckpt
        self.seed_block_length = int(seed_block_length)
        self.seed_refinement_steps = int(seed_refinement_steps)
        self.predictor_device = predictor_device or "cpu"
        self.max_block_length = (
            int(block_length if max_block_length is None else max_block_length)
        )
        self.max_refinement_steps = (
            int(diffusion_steps if max_refinement_steps is None else max_refinement_steps)
        )
        self.min_refinement_steps = int(min_refinement_steps or 1)
        self.pag_scheduler = PAGTupleScheduler(
            predictor_ckpt=self.predictor_ckpt,
            seed_block_length=self.seed_block_length,
            seed_refinement_steps=self.seed_refinement_steps,
            predictor_device=self.predictor_device,
            context_seed_block_length=context_seed_block_length,
            context_seed_stabilizing_steps=context_seed_stabilizing_steps,
            min_refinement_steps=self.min_refinement_steps,
            min_block_length=min_block_length,
            refinement_step_offset=refinement_step_offset,
        )
        self.model.pag_scheduler = self.pag_scheduler

    def _create_model_and_tokenizer(self, pretrained, dtype, trust_remote_code):
        self.model = (
            DreamModel.from_pretrained(
                pretrained,
                torch_dtype=get_dtype(dtype),
                trust_remote_code=trust_remote_code,
            )
            .eval()
        ).to(self.device)
        self.model.diffusion_generate = types.MethodType(
            DreamGenerationMixin.diffusion_generate,
            self.model,
        )
        self.model._sample = types.MethodType(DreamGenerationMixin._sample_pag, self.model)
        self.model.pag_scheduler = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained, trust_remote_code=trust_remote_code
        )

    def _generate_batch(
        self,
        prompts: list[str],
    ) -> tuple[list[str], list[int] | None, list[int] | None, list | None]:
        if len(prompts) != 1:
            raise ValueError("PAG Dream generation currently supports batch_size=1")

        if self.if_apply_chat_template:
            messages = [{"role": "user", "content": prompts[0]}]
            prompts = [self.apply_chat_template(messages)]
        elif self.add_bos_token:
            prompts = [self.tokenizer.bos_token + p for p in prompts]

        prompt_ids = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        ).input_ids
        if len(prompt_ids) > self.max_length - self.max_new_tokens:
            eval_logger.warning(
                "Prompt length %s is larger than %s, cutoff on the left side",
                len(prompt_ids),
                self.max_length - self.max_new_tokens,
            )
            prompt_ids = prompt_ids[-(self.max_length - self.max_new_tokens) :]

        attn_mask = prompt_ids.ne(self.tokenizer.pad_token_id)
        prompt_ids = prompt_ids.to(device=self.device)
        attn_mask = attn_mask.to(device=self.device)

        self.pag_scheduler.reset()
        self.model.pag_scheduler = self.pag_scheduler
        generation_ids = self.model.diffusion_generate(
            prompt_ids,
            attention_mask=attn_mask,
            max_new_tokens=self.max_new_tokens,
            output_history=False,
            return_dict_in_generate=True,
            steps=self.diffusion_steps,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            alg=self.alg,
            alg_temp=self.alg_temp,
            threshold=self.threshold,
            dual_cache=self.dual_cache,
            block_length=self.block_length,
            max_block_length=self.max_block_length,
            max_refinement_steps=self.max_refinement_steps,
        )

        generated_sequence = generation_ids.sequences[0][prompt_ids.shape[1] :]
        self.num_tokens += generated_sequence.numel()
        self.num_tokens_excluding_eos += (
            generated_sequence != self.tokenizer.eos_token_id
        ).sum().item()
        responses = [
            self.tokenizer.decode(g[len(p) :].tolist()).split(self.tokenizer.eos_token)[0]
            for p, g in zip(prompt_ids, generation_ids.sequences, strict=False)
        ]
        print("=" * 20)
        print(f"NFE for each block: {generation_ids.nfe_history}")
        print(f"Block length for each block: {generation_ids.block_history}")
        print(f"Schedule history: {generation_ids.schedule_history}")
        print("=" * 20, end="\n\n")
        return (
            responses,
            generation_ids.nfe_history,
            generation_ids.block_history,
            generation_ids.schedule_history,
        )

    def generate_until(self, requests: list[Instance], disable_tqdm: bool = False):
        del disable_tqdm

        res = []
        total_nfe = []
        num_blocks = []
        if self.use_cache:
            assert self.dual_cache, "Cached PAG decoding requires dual_cache=True"
            self.model._sample = types.MethodType(
                DreamGenerationMixin._sample_pag_cache,
                self.model,
            )
        elif self.alg == "confidence_threshold":
            self.model._sample = types.MethodType(DreamGenerationMixin._sample_pag, self.model)
        else:
            raise NotImplementedError(self.alg)

        processed_count = 0
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            rank = self.rank
            save_path = os.path.join(self.save_dir, f"rank_{rank}.jsonl")
            print(f"save_path: {save_path}")
            if os.path.exists(save_path):
                print(f"load from {save_path}")
                with open(save_path, encoding="utf-8") as f:
                    res = [json.loads(line) for line in f]
                    processed_count = len(res)
                print(f"processed_count: {processed_count}")

        start_time = time.time()
        for batch_idx in range(0, len(requests), self.batch_size):
            batch_requests = requests[batch_idx : batch_idx + self.batch_size]
            contexts, gen_args = zip(
                *[req.arguments for req in batch_requests],
                strict=False,
            )

            if batch_idx < processed_count:
                continue

            responses, nfe_history, block_history, _schedule_history = self._generate_batch(
                list(contexts)
            )
            total_nfe.append(sum(nfe_history))
            num_blocks.append(len(block_history))
            if not self.escape_until:
                for i, response in enumerate(responses):
                    for stop_seq in gen_args[0]["until"]:
                        response = response.split(stop_seq)[0]
                    responses[i] = response

            res.extend(responses)

            if self.save_dir is not None:
                for response in responses:
                    with open(save_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(response, ensure_ascii=False) + "\n")

        end_time = time.time()
        print()
        print(f"Total number of tokens: {self.num_tokens}")
        print(f"Total number of tokens excluding EOS: {self.num_tokens_excluding_eos}")
        print(f"Total time taken: {end_time - start_time} seconds")
        print(f"Tokens per second: {self.num_tokens / (end_time - start_time)}")
        print()
        print(f"NFE for each sample: {total_nfe}")
        print(f"Total NFE: {sum(total_nfe)}")
        print(f"Average NFE per sample: {sum(total_nfe) / len(total_nfe)}")
        print()
        print(f"Number of blocks for each sample: {num_blocks}")
        print(f"Average number of blocks per sample: {sum(num_blocks) / len(num_blocks)}")
        print(f"Average block length: {self.max_new_tokens / (sum(num_blocks) / len(num_blocks))}")
        print()
        return res


if __name__ == "__main__":
    cli_evaluate()
