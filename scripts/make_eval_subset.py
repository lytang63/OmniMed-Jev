#!/usr/bin/env python3
"""Build the fixed evaluation subset used for base-vs-fine-tuned comparisons.

Design requirements, each of which bit us once:

1. **Same subset across runs.** Sampling is seeded and derived only from the
   source data, so the base model and every checkpoint are scored on identical
   rows. A moving evaluation set makes metric deltas uninterpretable.

2. **Balanced across task families.** The raw test split is 79% classification,
   so a proportional draw would barely measure the other five tasks.

3. **Both answer classes for the "none" tasks.** multi_label_classification and
   detection are the two tasks whose ground truth includes a "no findings" case.
   A subset containing only non-empty answers cannot tell whether the model
   learned to say "none" when it should — which is precisely the regression this
   subset exists to catch. The earlier subset had 0 empty detection rows out of
   200, so it was blind to that failure mode.

Usage:
    python scripts/make_eval_subset.py --per-task 200 --out datas/eval
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

# Must match omnimedjev/engine/evaluate.py EMPTY_LABELS, plus the converter's sentinel.
EMPTY_MARKERS = {"", "[]", "{}", "null", "none", "no finding", "no findings", "n/a", "na"}


def is_empty(row: dict) -> bool:
    answer = row.get("answer")
    answer = answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)
    return answer.strip().lower() in EMPTY_MARKERS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="datas/preprocessed/Preprocessed-omnimed-suite/testing.jsonl")
    ap.add_argument("--out", default="datas/eval")
    ap.add_argument("--split", default="testing")
    ap.add_argument("--per-task", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.source).open() if line.strip()]
    by_task = collections.defaultdict(list)
    for row in rows:
        by_task[row.get("task_type", "?")].append(row)
    print(f"  source: {args.source}  ({len(rows)} rows, {len(by_task)} tasks)")

    rng = random.Random(args.seed)
    selected = []
    for task, items in by_task.items():
        n = min(args.per_task, len(items))
        empties = [r for r in items if is_empty(r)]
        filled = [r for r in items if not is_empty(r)]

        if not empties:
            picked = rng.sample(filled, n)
        else:
            # Keep the task's own empty/non-empty proportions, but guarantee at
            # least a handful of each so both behaviours are scored.
            share = len(empties) / len(items)
            n_empty = max(min(5, len(empties)), min(len(empties), round(n * share)))
            n_filled = min(len(filled), n - n_empty)
            n_empty = min(len(empties), n - n_filled)
            picked = rng.sample(empties, n_empty) + rng.sample(filled, n_filled)
        selected.extend(picked)
        print(f"    {task:<36} {len(picked):>4}  (empty {sum(1 for r in picked if is_empty(r)):>3} / filled {sum(1 for r in picked if not is_empty(r)):>4}, of {len(items)})")

    rng.shuffle(selected)
    out = Path(args.out) / "Preprocessed-omnimed-suite"
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"{args.split}.jsonl").open("w") as fh:
        for row in selected:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    # The engine's preprocessed-dataset check wants a train/validation file too.
    for name in ("train", "validation"):
        with (out / f"{name}.jsonl").open("w") as fh:
            for row in selected[:5]:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = collections.Counter(r["task_type"] for r in selected)
    print(f"\n  wrote {len(selected)} rows -> {out / (args.split + '.jsonl')}")
    for t, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        e = sum(1 for r in selected if r["task_type"] == t and is_empty(r))
        print(f"    {t:<36} {n:>4}  (empty {e})")
    print(f"  sources: {len(set(r.get('source') for r in selected))}")


if __name__ == "__main__":
    main()
