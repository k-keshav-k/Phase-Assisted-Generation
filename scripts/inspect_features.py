"""Inspect feature/target distributions in a phase_tuples JSONL file.

Sanity check before training: prints sequence lengths, per-field distributions,
and the empirical bucket vocabularies for the prediction targets so we can
size classification heads later.

Usage
-----
    python scripts/inspect_features.py traces/rich/stab_tuples_conf_train_rich.jsonl
    python scripts/inspect_features.py traces/rich/stab_tuples_conf_train_rich_nodlm.jsonl --top 30
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def _load(path: Path) -> list[list[dict]]:
    out: list[list[dict]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tuples = rec.get("tuples")
            if isinstance(tuples, list) and tuples:
                out.append(tuples)
    return out


def _summarise_numeric(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "min": float(min(values)),
        "p25": float(statistics.quantiles(values, n=4)[0]) if len(values) >= 4 else float(min(values)),
        "median": float(statistics.median(values)),
        "p75": float(statistics.quantiles(values, n=4)[2]) if len(values) >= 4 else float(max(values)),
        "max": float(max(values)),
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.pstdev(values)),
    }


def _print_table(title: str, rows: list[tuple[str, str]]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    width = max(len(k) for k, _ in rows) if rows else 0
    for k, v in rows:
        print(f"  {k.ljust(width)}  {v}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Inspect a phase_tuples JSONL file")
    ap.add_argument("jsonl", type=Path, help="Path to phase_tuples JSONL")
    ap.add_argument("--top", type=int, default=20, help="Show top-K bucketed values per categorical target")
    args = ap.parse_args(argv)

    seqs = _load(args.jsonl)
    if not seqs:
        print(f"No usable sequences in {args.jsonl}")
        return 1

    n_problems = len(seqs)
    seq_lens = [len(s) for s in seqs]
    n_blocks = sum(seq_lens)
    fields = list(seqs[0][0].keys())

    print(f"File: {args.jsonl}")
    print(f"Problems: {n_problems}    Blocks: {n_blocks}    Fields/block: {len(fields)}")

    _print_table(
        "Sequence length (blocks per problem)",
        [(k, f"{v:.2f}") for k, v in _summarise_numeric([float(x) for x in seq_lens]).items()],
    )

    per_field_values: dict[str, list[float]] = {f: [] for f in fields}
    for s in seqs:
        for t in s:
            for f in fields:
                v = t.get(f, 0)
                if v is None:
                    v = 0
                per_field_values[f].append(float(v))

    print("\nPer-field summary")
    print("-" * 80)
    header = f"{'field':28} {'min':>8} {'p25':>8} {'median':>8} {'p75':>8} {'max':>8} {'mean':>8} {'std':>8}"
    print(header)
    for f in fields:
        s = _summarise_numeric(per_field_values[f])
        if not s:
            continue
        print(
            f"{f:28} "
            f"{s['min']:>8.3g} {s['p25']:>8.3g} {s['median']:>8.3g} "
            f"{s['p75']:>8.3g} {s['max']:>8.3g} {s['mean']:>8.3g} {s['std']:>8.3g}"
        )

    for target in ("block_size", "nfe"):
        if target not in fields:
            continue
        counter = Counter(int(round(v)) for v in per_field_values[target])
        total = sum(counter.values())
        items = counter.most_common(args.top)
        print(f"\nTop {len(items)} `{target}` buckets (out of {len(counter)} unique, n={total})")
        print("-" * 50)
        for value, count in items:
            print(f"  {value:>4d}    n={count:>7d}    {100.0 * count / total:5.2f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
