from __future__ import annotations

import json
import os
import time

import torch
from eval_llada_adablock import LLaDAEvalHarness as AdaBlockLLaDAEvalHarness
from generate_pag import (
    generate_pag,
    generate_pag_dual_cache,
    generate_pag_prefix_cache,
)
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model
from pag_predictor import PAGTupleScheduler
from tqdm import tqdm


@register_model("llada_dist")
class LLaDAEvalHarness(AdaBlockLLaDAEvalHarness):
    def __init__(
        self,
        model_path: str = "",
        mask_id: int = 126336,
        max_length: int = 4096,
        batch_size: int = 32,
        mc_num: int = 128,
        is_check_greedy: bool = True,
        steps: int = 1024,
        gen_length: int = 1024,
        block_length: int = 1024,
        remasking: str = "low_confidence",
        device: str = "cuda",
        use_cache: bool = False,
        threshold: float | None = None,
        save_dir: str | None = None,
        show_speed: bool = False,
        dual_cache: bool = False,
        predictor_ckpt: str | None = None,
        seed_block_length: int | None = None,
        seed_refinement_steps: int | None = None,
        predictor_device: str | None = "cpu",
        max_block_length: int | None = None,
        max_refinement_steps: int | None = None,
        min_refinement_steps: int | None = 1,
        context_seed_block_length: int | None = None,
        context_seed_stabilizing_steps: int | None = None,
        **kwargs,
    ) -> None:
        if predictor_ckpt is None:
            msg = "predictor_ckpt is required for PAG generation"
            raise ValueError(msg)
        if seed_block_length is None or seed_refinement_steps is None:
            msg = "seed_block_length and seed_refinement_steps are required"
            raise ValueError(msg)

        super().__init__(
            model_path=model_path,
            mask_id=mask_id,
            max_length=max_length,
            batch_size=batch_size,
            mc_num=mc_num,
            is_check_greedy=is_check_greedy,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            remasking=remasking,
            device=device,
            use_cache=use_cache,
            threshold=threshold,
            save_dir=save_dir,
            show_speed=show_speed,
            dual_cache=dual_cache,
            **kwargs,
        )
        if not hasattr(self, "_rank"):
            self._rank = 0
            self._world_size = 1

        self.predictor_ckpt = predictor_ckpt
        self.seed_block_length = int(seed_block_length)
        self.seed_refinement_steps = int(seed_refinement_steps)
        self.predictor_device = predictor_device or "cpu"
        self.max_block_length = int(
            block_length if max_block_length is None else max_block_length
        )
        self.max_refinement_steps = int(
            steps if max_refinement_steps is None else max_refinement_steps
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
        )
        self.schedule_histories: list[list[dict[str, object]]] = []

    def _generate_one(self, input_ids: torch.Tensor):
        self.pag_scheduler.reset()
        if self.use_cache:
            if self.dual_cache:
                return generate_pag_dual_cache(
                    self.model,
                    input_ids,
                    self.pag_scheduler,
                    steps=self.steps,
                    gen_length=self.gen_length,
                    temperature=0.0,
                    remasking=self.remasking,
                    mask_id=self.mask_id,
                    threshold=self.threshold,
                    max_block_length=self.max_block_length,
                    max_refinement_steps=self.max_refinement_steps,
                )
            return generate_pag_prefix_cache(
                self.model,
                input_ids,
                self.pag_scheduler,
                steps=self.steps,
                gen_length=self.gen_length,
                temperature=0.0,
                remasking=self.remasking,
                mask_id=self.mask_id,
                threshold=self.threshold,
                max_block_length=self.max_block_length,
                max_refinement_steps=self.max_refinement_steps,
            )
        return generate_pag(
            self.model,
            input_ids,
            self.pag_scheduler,
            steps=self.steps,
            gen_length=self.gen_length,
            temperature=0.0,
            remasking=self.remasking,
            mask_id=self.mask_id,
            threshold=self.threshold,
            max_block_length=self.max_block_length,
            max_refinement_steps=self.max_refinement_steps,
        )

    def generate_until(self, requests):
        output = []
        num_tokens = 0
        num_tokens_excluding_eos = 0
        total_nfe = []
        num_blocks = []
        self.schedule_histories = []
        processed_count = 0
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            rank = self.rank
            save_path = os.path.join(self.save_dir, f"rank_{rank}.jsonl")
            print(f"save_path: {save_path}")
            if os.path.exists(save_path):
                print(f"load from {save_path}")
                with open(save_path, encoding="utf-8") as file_obj:
                    output = [json.loads(line) for line in file_obj]
                    processed_count = len(output)
                print(f"processed_count: {processed_count}")
        start_time = time.time()
        for i, req in enumerate(tqdm(requests, desc="Generating...")):
            if i < processed_count:
                continue

            question = req.args[0]
            if self.is_instruct:
                messages = [{"role": "user", "content": question}]
                user_input = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                input_ids = self.tokenizer(user_input)["input_ids"]
            else:
                user_input = question
                input_ids = self.tokenizer(user_input)["input_ids"]

            stop_tokens = req.args[1]["until"]
            input_ids = torch.tensor(input_ids).to(self.device).unsqueeze(0)
            generated_answer, nfe_history, block_history, schedule_history = self._generate_one(
                input_ids
            )
            self.schedule_histories.append(schedule_history)

            print("=" * 20)
            print(f"NFE for each block: {nfe_history}")
            print(f"Block length for each block: {block_history}")
            print(f"Schedule history: {schedule_history}")
            print("=" * 20, end="\n\n")

            is_humaneval = "task_id" in req.doc and str(req.doc["task_id"]).lower().startswith(
                "humaneval"
            )
            if self.is_instruct and is_humaneval:
                if self.show_speed:
                    num_tokens += generated_answer.numel()
                    num_tokens_excluding_eos += (generated_answer != 126081).sum()
                    total_nfe.append(sum(nfe_history))
                    num_blocks.append(len(block_history))
                generated_answer = self.tokenizer.decode(
                    generated_answer[0][input_ids.shape[1] :],
                    skip_special_tokens=True,
                )
            else:
                generated_answer = self.tokenizer.decode(
                    generated_answer[0][input_ids.shape[1] :],
                    skip_special_tokens=False,
                )
                for stop_seq in stop_tokens:
                    if stop_seq in generated_answer:
                        generated_answer = generated_answer.split(stop_seq)[0]

                generated_answer_ids = torch.tensor(self.tokenizer(generated_answer)["input_ids"])
                if self.show_speed:
                    num_tokens += generated_answer_ids.numel()
                    num_tokens_excluding_eos += (generated_answer_ids != 126081).sum()
                    total_nfe.append(sum(nfe_history))
                    num_blocks.append(len(block_history))
                generated_answer = self.tokenizer.decode(
                    generated_answer_ids,
                    skip_special_tokens=True,
                )
            output.append(generated_answer)

            if self.save_dir is not None:
                with open(save_path, "a", encoding="utf-8") as file_obj:
                    file_obj.write(json.dumps(generated_answer, ensure_ascii=False) + "\n")

        end_time = time.time()
        if self.show_speed:
            print()
            print(f"Total number of tokens: {num_tokens}")
            print(f"Total number of tokens excluding EOS: {num_tokens_excluding_eos}")
            print(f"Total time taken: {end_time - start_time} seconds")
            print(f"Tokens per second: {num_tokens / (end_time - start_time)}")
            print()
            print(f"NFE for each sample: {total_nfe}")
            print(f"Total NFE: {sum(total_nfe)}")
            print(f"Average NFE per sample: {sum(total_nfe) / len(total_nfe)}")
            print()
            print(f"Number of blocks for each sample: {num_blocks}")
            print(f"Average number of blocks per sample: {sum(num_blocks) / len(num_blocks)}")
            print(f"Average block length: {self.gen_length / (sum(num_blocks) / len(num_blocks))}")
            print()
        return output


if __name__ == "__main__":
    cli_evaluate()
