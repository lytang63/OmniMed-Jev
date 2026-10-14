#!/usr/bin/env python3
"""Build a task-balanced training subset for the unified multi-task SFT run.

WHY balanced rather than proportional:

  The mixed train split is dominated by classification — MedMNIST alone
  contributes ~500k rows, so a proportional 60k draw comes out 82.8%
  classification with cell_counting at 0.2% (99 rows). A model trained on that
  sees almost no gradient from counting, detection or report generation, which
  defeats the point of unified multi-task fine-tuning: the whole claim is that
  one model handles every task family.

  Sampling an equal number per task_type gives each family ~1/6 of the signal.
  Within a family, rows are drawn proportionally across its source datasets so
  e.g. classification still sees all nine MedMNIST subsets in their natural
  proportions relative to each other.

  This changes the training distribution relative to the raw data. That is
  deliberate and must be stated in any write-up.

Usage:
    python scripts/make_train_subset.py --per-task 10000 --out datas/balanced/train.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="datas/preprocessed/Preprocessed-omnimed-suite/train.jsonl")
    ap.add_argument("--out", default="datas/balanced/train.jsonl")
    ap.add_argument("--per-task", type=int, default=10000,
                    help="how many rows to draw for EACH task_type (0 = keep everything)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--empty-cap", type=float, default=0.25,
                    help="max fraction of empty-answer rows WITHIN a task that has any "
                         "(0 = no cap). See the note below for why this exists.")
    args = ap.parse_args()

    src = Path(args.source)
    rows = [json.loads(line) for line in src.open() if line.strip()]
    print(f"  source: {src}  ({len(rows)} rows)")

    # Must match the engine's own EMPTY_LABELS, otherwise the empty-answer cap
    # silently stops working the moment the converter switches from "" to the
    # explicit "none" sentinel.
    EMPTY_MARKERS = {"", "[]", "{}", "null", "none", "no finding", "no findings", "n/a", "na"}

    def is_empty(row: dict) -> bool:
        answer = row.get("answer")
        answer = answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)
        return answer.strip().lower() in EMPTY_MARKERS

    by_task = collections.defaultdict(list)
    for row in rows:
        by_task[row.get("task_type", "?")].append(row)

    print("  original per-task counts:")
    for task, items in sorted(by_task.items(), key=lambda kv: -len(kv[1])):
        n_empty = sum(1 for r in items if is_empty(r))
        print(f"    {task:<36} {len(items):>7}  ({len(items)/len(rows)*100:5.1f}%)"
              f"   empty {n_empty:>6} ({n_empty/len(items)*100:5.1f}%)")

    if args.per_task <= 0:
        selected = rows
    else:
        rng = random.Random(args.seed)
        selected = []
        for task, items in by_task.items():
            empties = [r for r in items if is_empty(r)]
            filled = [r for r in items if not is_empty(r)]
            n = min(args.per_task, len(items))

            # Cap the empty-answer share within the task.
            #
            # WHY: the raw distribution is 52.7% empty for multi_label_classification
            # (ChestMNIST ships 53.9% all-zero label vectors — those radiographs truly
            # have no findings) and 12.6% for detection. Training on that teaches the
            # model that emitting nothing is a low-risk default: a model that always
            # answers "" scores better on loss than one that guesses. Measured: after
            # 233 steps the adapter returned "" for EVERY regression, multi-label and
            # detection sample.
            #
            # The empties are factually correct and must stay — saying "no findings" is
            # a real capability — but they must not dominate. Cap them, and let the
            # task's non-empty rows fill the rest.
            n_empty = min(len(empties), int(round(n * args.empty_cap))) if filled else len(empties)
            n_filled = min(len(filled), n - n_empty)
            if n_filled < n - n_empty:
                n_empty = min(len(empties), n - n_filled)
            picked = rng.sample(empties, n_empty) + rng.sample(filled, n_filled)

            # Top up from whichever pool still has room, if the cap left us short.
            while len(picked) < n:
                pool = [r for r in (empties if len(filled) <= n_filled else filled)
                        if all(id(r) != id(p) for p in picked)]
                if not pool:
                    break
                picked.append(pool[rng.randrange(len(pool))])
            selected.extend(picked)
        rng.shuffle(selected)

    # `length` drives transformers' group_by_length sampler (length_column_name).
    # The collator pads every batch to its longest member, so grouping similar
    # lengths together is what stops a single long report from dragging a whole
    # batch of short answers up to ~740 tokens. Character count of prompt+answer
    # is a good proxy: the image tokens are constant across samples, so they do
    # not affect the ordering.
    def est_len(row: dict) -> int:
        answer = row.get("answer")
        answer = answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)
        return len(str(row.get("prompt", ""))) + len(answer or "")

    for row in selected:
        row["length"] = est_len(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for row in selected:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = collections.Counter(r["task_type"] for r in selected)
    print(f"\n  wrote {len(selected)} rows -> {out}")
    print("  balanced per-task mix:")
    for task, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"    {task:<36} {n:>7}  ({n/len(selected)*100:5.1f}%)")

    src_counts = collections.Counter(r.get("source", "?") for r in selected)
    print("  per-source mix:")
    for s, n in sorted(src_counts.items(), key=lambda kv: -kv[1]):
        print(f"    {s:<20} {n:>7}")

    # cost estimate, measured on this box
    rate = 5.31  # samples/s, MedGemma-1.5-4B QLoRA bs=8 on one A800
    print(f"\n  est. cost at {rate} samples/s (1 GPU): "
          f"{len(selected)*2/rate/3600:.1f} h for 2 epochs")


if __name__ == "__main__":
    main()
