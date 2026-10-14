#!/usr/bin/env python3
"""Score the generative baseline's *self-reported* confidence with MedJev's metrics.

WHY this exists: without it the calibration table has only one column, and the
obvious objection is that MedJev looks calibrated only because nothing else was
asked to report a probability. So the same prompt machinery the repo already
has (`omnimedjev/engine/infer.py --elicit_confidence`) is used to ask the generative
model for `PROBS: {...}` on choice questions and `CONFIDENCE: <0-1>` elsewhere,
and this script computes the identical metrics from that output.

**Every metric function is imported from `omnimedjev/engine/medjev_metrics.py`**, so
the two sides cannot drift apart in definition -- that is the whole point.

Two comparability notes that must travel with any number taken from here:

* On classification and multi-label the prompt asks for a probability for every
  candidate, so the records reduce to one per (row, candidate) and the unit
  matches MedJev's per-decision unit exactly. On counting and regression only a
  scalar is requested (100 ordered EF levels cannot be enumerated in a prompt),
  so those families get the top-label metrics -- ECE, MCE, Murphy, AURC -- but
  no NLL, Brier, top-k or DDx. Those cells are reported as "—", not as zero.
* The model was never *trained* to emit these numbers; this is prompted
  self-report. MedJev's distributions are trained. That asymmetry is the
  finding, not a flaw in the measurement, but it has to be stated.

Usage:
    python scripts/eval_sft_metrics.py \\
        --predictions <run>/testing_predictions.jsonl \\
        --ground-truth datas/eval/Preprocessed-omnimed-suite/testing.jsonl \\
        --output <run>/sft_metrics.json
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from omnimedjev.engine.evaluate import normalize_text, parse_multilabel
from omnimedjev.engine.medjev_metrics import (coverage_at_k, ddx_set_size, entropy,
                                       expected_calibration_error, max_calibration_error,
                                       murphy_decomposition, risk_coverage, wilson_interval)

FAMILIES = ("disease_diagnosis_classification", "multi_label_classification",
            "cell_counting", "regression")
SCORE_KEYS = ("scores", "probs", "probabilities", "class_probs", "label_scores")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def numeric(value: Any) -> float | None:
    text = str(value).replace(",", " ")
    for token in text.split():
        try:
            return float(token)
        except ValueError:
            continue
    return None


def scalar_block(confidences: Sequence[float], correct: Sequence[bool]) -> dict[str, Any]:
    """Top-label metrics from a bare (confidence, correct) pair.

    Uses the same helpers `metric_block` uses, so a scalar-confidence baseline
    and a full-distribution model are scored by identical code on the metrics
    both can produce.
    """
    if not confidences:
        return {}
    hits = sum(1 for flag in correct if flag)
    low, high = wilson_interval(hits, len(confidences))
    block: dict[str, Any] = {
        "decisions": len(confidences),
        "accuracy": hits / len(confidences),
        "accuracy_ci95": [low, high],
        "ece": expected_calibration_error(confidences, correct),
        "mce": max_calibration_error(confidences, correct),
        "mean_confidence": sum(confidences) / len(confidences),
    }
    block.update(murphy_decomposition(confidences, correct))
    block.pop("brier")                      # a top-label Brier is not the model's Brier
    block.update(risk_coverage(confidences, correct))
    return block


def distribution_block(rows: Sequence[tuple[dict[str, float], str]]) -> dict[str, Any]:
    """Same metric set as MedJev's, from a model-supplied probability vector.

    `rows` is (probabilities_by_label, gold_label). Anything the model omitted
    from its vector is treated as probability 0, and the remainder is assigned
    to a placeholder so the vector still sums to 1 -- the prompt asks for every
    candidate, so a missing one is a real omission rather than a design choice.
    """
    if not rows:
        return {}
    records = []
    for probabilities, gold in rows:
        labels = list(probabilities)
        total = sum(probabilities.values())
        if total <= 0:
            continue
        probabilities = {k: v / total for k, v in probabilities.items()}
        if gold not in probabilities:
            probabilities[gold] = 0.0
            labels.append(gold)
        gold_index = labels.index(gold)
        top = max(probabilities, key=probabilities.__getitem__)
        target = {label: (1.0 if label == gold else 0.0) for label in labels}
        records.append({
            "probabilities": probabilities,
            "gold": gold,
            "gold_index": gold_index,
            "top": top,
            "target": target,
            "target_kind": "one_hot",
            "nll": -math.log(max(probabilities[gold], 1e-30)),
            "brier": sum((probabilities[c] - target[c]) ** 2 for c in labels),
            "entropy": entropy(list(probabilities.values())),
        })
    confidences = [max(r["probabilities"].values()) for r in records]
    correct = [r["top"] == r["gold"] for r in records]
    block = scalar_block(confidences, correct)
    block["nll"] = sum(r["nll"] for r in records) / len(records)
    block["brier"] = sum(r["brier"] for r in records) / len(records)
    block["mean_normalized_entropy"] = sum(r["entropy"] for r in records) / len(records)
    for tau in (0.5, 0.9):
        block[f"ddx_set_size_at_{tau}"] = sum(
            ddx_set_size(list(r["probabilities"].values()), tau) for r in records) / len(records)
    max_k = max(len(r["probabilities"]) for r in records)
    for k in (1, 3, 5):
        if max_k < k:
            continue
        eligible = [r for r in records if len(r["probabilities"]) >= k]
        if eligible:
            block[f"top{k}_coverage"] = sum(
                1 for r in eligible if coverage_at_k(list(r["probabilities"].values()), r["gold_index"], k)
            ) / len(eligible)
    return block


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--ground-truth", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    index = {str(row["uid"]): row for row in read_jsonl(args.predictions)}
    report: dict[str, Any] = {"families": {}, "rows": 0, "with_scores": 0, "with_scalar": 0,
                              "no_signal": 0}

    per_family_scalar: dict[str, tuple[list[float], list[bool]]] = collections.defaultdict(lambda: ([], []))
    per_family_dist: dict[str, list[tuple[dict[str, float], str]]] = collections.defaultdict(list)

    for row in read_jsonl(args.ground_truth):
        task = str(row.get("task_type"))
        if task not in FAMILIES:
            continue
        record = index.get(str(row.get("uid")))
        if record is None:
            continue
        report["rows"] += 1
        prediction = str(record.get("prediction", ""))
        scores = next((record[k] for k in SCORE_KEYS if isinstance(record.get(k), dict)), None)
        confidence = record.get("confidence")
        gold_raw = row.get("raw_answer", row.get("answer"))

        if scores:
            probabilities = {normalize_text(k): float(v) for k, v in scores.items()}
            report["with_scores"] += 1
            if task == "multi_label_classification":
                # One record per FINDING, matching MedJev's per-Noul unit.
                positives = {normalize_text(v) for v in parse_multilabel(gold_raw)}
                for label, probability in probabilities.items():
                    per_family_dist[task].append(({label: probability, "__absent__": 1.0 - probability},
                                                  label if label in positives else "__absent__"))
            else:
                gold = normalize_text(gold_raw)
                per_family_dist[task].append((probabilities, gold))
            continue

        if isinstance(confidence, (int, float)):
            report["with_scalar"] += 1
            if task == "multi_label_classification":
                positives = {normalize_text(v) for v in parse_multilabel(gold_raw)}
                predicted = {normalize_text(v) for v in parse_multilabel(prediction)}
                correct = positives == predicted
            elif task in ("cell_counting", "regression"):
                a, b = numeric(prediction), numeric(gold_raw)
                correct = (a is not None and b is not None and round(a) == round(b))
            else:
                correct = normalize_text(prediction) == normalize_text(gold_raw)
            per_family_scalar[task][0].append(float(confidence))
            per_family_scalar[task][1].append(bool(correct))
            continue

        report["no_signal"] += 1

    for task in FAMILIES:
        confidences, correct = per_family_scalar.get(task, ([], []))
        block = distribution_block(per_family_dist.get(task, []))
        scalar = scalar_block(confidences, correct)
        # Prefer the richer block; record the unit so the two sides are not
        # compared at different granularities by accident.
        if block:
            merged = block
            merged["unit"] = ("per_finding" if task == "multi_label_classification" else "per_row")
            merged["source"] = "prompted PROBS vector"
        elif scalar:
            merged = scalar
            merged["unit"] = "per_row"
            merged["source"] = "prompted scalar CONFIDENCE"
        else:
            continue
        report["families"][task] = merged

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
