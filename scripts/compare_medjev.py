#!/usr/bin/env python3
"""Put MedJev and the generative SFT baseline side by side.

Both numbers come out of the same `omnimedjev evaluate` invocation shape over the same
held-out ground truth, so this script only has to line them up -- it computes
nothing. MedJev's folded predictions are scored by that evaluator unchanged;
the extra probability columns come from `eval_medjev_metrics.py`, because a
string-based metric cannot tell a calibrated distribution from a confident
argmax and that distinction is the whole point of the architecture.

Usage:
    python scripts/compare_medjev.py \\
        --baseline datas/eval/runs-sft/checkpoint-2424/scores.json \\
        --medjev datas/eval/runs-medjev/checkpoint-2424/scores.json \\
        --calibration datas/eval/runs-medjev/checkpoint-2424/metrics.json
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path
from typing import Any

from omnimedjev.engine.evaluate import (balanced_accuracy_score, multilabel_metrics,
                                 normalize_text, parse_multilabel)

# task -> (display name, headline metric key, lower_is_better)
TASKS: tuple[tuple[str, str, str, bool], ...] = (
    ("disease_diagnosis_classification", "Classification", "balanced_accuracy", False),
    ("multi_label_classification", "Multi-label", "multi_label_f1_score", False),
    ("cell_counting", "Counting", "cell_counting_mean_absolute_error", True),
    ("regression", "Regression", "regression_mean_absolute_error", True),
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing {path} -- run the corresponding eval first")
    return json.loads(path.read_text(encoding="utf-8"))


def metrics_of(payload: dict[str, Any]) -> dict[str, float]:
    """Accept both the flat (one split) and the `mean`/`per_split` shapes."""
    for key in ("mean", "mean_metrics", "metrics"):
        block = payload.get(key)
        if isinstance(block, dict):
            return {k: float(v) for k, v in block.items() if isinstance(v, (int, float))}
    return {k: float(v) for k, v in payload.items() if isinstance(v, (int, float))}


def render(value: float | None, lower_is_better: bool) -> str:
    if value is None:
        return "—"
    return f"{value:.4f}"


def trivial_baselines(gt_path: Path) -> dict[str, float]:
    """What a model that ignores the image entirely would score.

    WHY this column exists: several of these metrics are close to saturated by
    degenerate strategies, and without the reference point a real improvement
    and a lucky constant are indistinguishable.

    * Multi-label F1 here is ~0.52 for predicting *nothing at all*, because 52%
      of the eval subset's rows are genuinely finding-free and the evaluator
      scores (empty, empty) as a perfect match.
    * Counting and regression are dominated by picking the mean.

    The constants are fitted on the *test* split, which makes them an oracle --
    a strictly stronger baseline than any real trivial model. Beating them is
    therefore sufficient, not necessary: failing to beat them means the result
    is not yet distinguishable from "predict the average".
    """
    rows = [json.loads(line) for line in gt_path.open(encoding="utf-8") if line.strip()]
    by_task: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_task[str(row.get("task_type"))].append(row)

    out: dict[str, float] = {}

    classification = by_task.get("disease_diagnosis_classification", [])
    if classification:
        labels = [str(r.get("answer")) for r in classification]
        majority = collections.Counter(labels).most_common(1)[0][0]
        out["balanced_accuracy"] = balanced_accuracy_score(
            [normalize_text(v) for v in labels], [normalize_text(majority)] * len(labels))

    multilabel = by_task.get("multi_label_classification", [])
    if multilabel:
        gold = [parse_multilabel(r.get("raw_answer")) for r in multilabel]
        out["multi_label_f1_score"] = multilabel_metrics(gold, [set()] * len(gold))["f1_score"]

    for task, key in (("cell_counting", "cell_counting_mean_absolute_error"),
                      ("regression", "regression_mean_absolute_error")):
        values = [float(r["raw_answer"]) for r in by_task.get(task, []) if r.get("raw_answer") is not None]
        if values:
            # Per-source constant, not one global mean: the regression split
            # mixes a 0-100 scale with a 0-4 one, and a single constant would
            # look far worse than any sensible trivial predictor.
            sources = collections.defaultdict(list)
            for row in by_task.get(task, []):
                if row.get("raw_answer") is not None:
                    sources[str(row.get("source"))].append(float(row["raw_answer"]))
            errors = [abs(value - statistics.mean(sources[source]))
                      for source, group in sources.items() for value in group]
            out[key] = sum(errors) / len(errors)

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--medjev", required=True, type=Path)
    ap.add_argument("--calibration", type=Path, default=None)
    ap.add_argument("--ground-truth", type=Path,
                    default=Path("datas/eval/Preprocessed-omnimed-suite/testing.jsonl"),
                    help="eval ground truth, used only to compute the trivial baselines")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    baseline = metrics_of(read_json(args.baseline))
    medjev = metrics_of(read_json(args.medjev))
    calibration = read_json(args.calibration)["families"] if args.calibration and args.calibration.exists() else {}

    lines = [
        "# MedJev vs generative SFT (MedGemma-1.5-4B)",
        "",
        "Same base model, same 29,842-row balanced training subset, same held-out",
        "ground truth, same evaluator. The only intentional difference is `max_length`",
        "(3072 for MedJev, 1024 for the baseline), which a decision state requires.",
        "",
        "| Task | Metric | Trivial (oracle) | SFT 2-epoch | MedJev | MedJev vs SFT |",
        "|---|---|---:|---:|---:|---:|",
    ]

    trivial = trivial_baselines(args.ground_truth) if args.ground_truth and args.ground_truth.exists() else {}

    for key, label, metric, lower in TASKS:
        base_value = baseline.get(metric)
        med_value = medjev.get(metric)
        delta = ""
        if base_value is not None and med_value is not None:
            change = med_value - base_value
            better = (change < 0) if lower else (change > 0)
            delta = f"{change:+.4f} {'✅' if better else '❌'}"
        arrow = " ↓" if lower else " ↑"
        lines.append(f"| {label} | `{metric}`{arrow} | {render(trivial.get(metric), lower)} | "
                     f"{render(base_value, lower)} | {render(med_value, lower)} | {delta} |")

    if calibration:
        lines += [
            "",
            "## Probability metrics (MedJev only)",
            "",
            "The baseline generates text, so it has no distribution to score here.",
            "`nll`/`brier`/`ece` are over decisions whose target has exactly one right",
            "answer; `numeric_mae` is over decisions carrying our own discretization",
            "target, where a distribution is compared by its expectation instead.",
            "",
            "| Family | Decisions | NLL ↓ | Brier ↓ | ECE ↓ | Numeric MAE ↓ |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for family, entry in sorted(calibration.items()):
            count = entry.get("decisions", entry.get("numeric_decisions", 0))
            def fmt(key: str) -> str:
                value = entry.get(key)
                return f"{value:.4f}" if isinstance(value, (int, float)) else "—"
            lines.append(f"| {family} | {count} | {fmt('nll')} | {fmt('brier')} | "
                         f"{fmt('ece')} | {fmt('numeric_mae')} |")

    report = "\n".join(lines) + "\n"
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
