"""Generate a GSM8K eval prompts JSONL in the quick_eval_prompts.jsonl schema.

Pulls a disjoint slice of GSM8K (default: test[200:400]) and emits records of
the form:
    {
      "id": "gsm8k_test_0200",
      "category": "multi_step_arithmetic",
      "tags": ["math", "gsm8k"],
      "expected_answers": ["18"],
      "expected_contains": ["9", "18"],
      "prompt": "<question>\n\nSolve the problem step by step ... End with Final answer: <number>."
    }

`expected_answers` comes from the gold "#### <answer>" line.
`expected_contains` collects the right-hand sides of every "<<expr=result>>"
annotation in the chain-of-thought, plus the final answer.

Usage
-----
    python scripts/eval_gen.py
    python scripts/eval_gen.py --start 200 --limit 200 --split test \
        --output AdaBlock-dLLM/llada/gsm8k_eval_prompts.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset


_CALC_RE  = re.compile(r"<<[^=]+=([^>]+)>>")
_FINAL_RE = re.compile(r"####\s*(.+?)\s*$", re.MULTILINE)

PROMPT_SUFFIX = (
    "\n\nSolve the problem step by step, showing each intermediate calculation. "
    "End with a line formatted exactly as Final answer: <number>."
)


def _normalize_number(s: str) -> str:
    """Strip $, commas, trailing zeros for a clean comparison key."""
    return s.replace("$", "").replace(",", "").strip()


def _expected_answers(final: str) -> list[str]:
    """Variants the model might emit for the final number."""
    clean = _normalize_number(final)
    variants = {clean}
    # Drop trailing ".0" / ".00" so "18.0" matches "18"
    if "." in clean:
        try:
            f = float(clean)
            if f.is_integer():
                variants.add(str(int(f)))
            variants.add(f"{f:.2f}")
            variants.add(f"{f:g}")
        except ValueError:
            pass
    return sorted(variants, key=len, reverse=True)


def _expected_contains(answer_field: str, final: str) -> list[str]:
    """Intermediate calc results from <<...=R>> plus the final answer, deduped, in order."""
    seen: set[str] = set()
    out: list[str] = []
    for m in _CALC_RE.finditer(answer_field):
        v = _normalize_number(m.group(1))
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    final_clean = _normalize_number(final)
    if final_clean and final_clean not in seen:
        out.append(final_clean)
    return out


def _parse_gsm8k_answer(answer_field: str) -> tuple[str, list[str]]:
    m = _FINAL_RE.search(answer_field)
    final = m.group(1) if m else ""
    return final, _expected_contains(answer_field, final)


def main() -> int:
    p = argparse.ArgumentParser(description="Build a GSM8K eval prompts JSONL in quick_eval_prompts schema.")
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--start", type=int, default=200,
                   help="Starting index in the split (default 200, since test[:200] is already used).")
    p.add_argument("--limit", type=int, default=200, help="Number of examples to take.")
    p.add_argument("--output", type=Path,
                   default=Path("AdaBlock-dLLM/llada/gsm8k_eval_prompts.jsonl"))
    args = p.parse_args()

    print(f"Loading GSM8K ({args.split}) ...")
    ds = load_dataset("openai/gsm8k", "main", split=args.split)
    if args.start >= len(ds):
        raise SystemExit(f"start={args.start} >= split size {len(ds)}")
    end = min(args.start + args.limit, len(ds))
    if end - args.start < args.limit:
        print(f"Warning: only {end - args.start} examples available from index {args.start}.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("w", encoding="utf-8") as f:
        for i in range(args.start, end):
            ex = ds[i]
            final, contains = _parse_gsm8k_answer(ex["answer"])
            if not final:
                print(f"  skip idx={i}: no '#### <answer>' found")
                continue
            record = {
                "id":                f"gsm8k_{args.split}_{i:04d}",
                "category":          "multi_step_arithmetic",
                "tags":              ["math", "gsm8k"],
                "expected_answers":  _expected_answers(final),
                "expected_contains": contains,
                "prompt":            ex["question"].strip() + PROMPT_SUFFIX,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} prompts → {args.output}")
    print(f"IDs: gsm8k_{args.split}_{args.start:04d} … gsm8k_{args.split}_{end-1:04d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
