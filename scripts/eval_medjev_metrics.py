#!/usr/bin/env python3
"""Record everything MedJev's output structure makes measurable.

Each metric here answers a question the string-based evaluator in
`omnimedjev/engine/evaluate.py` cannot express:

* **Murphy decomposition** separates calibration (reliability) from
  discrimination (resolution). An ECE alone cannot tell a well-calibrated model
  from one that always emits the class prior, because the latter is also
  reliably calibrated.
* **Risk-coverage / AURC** answers "can it be trusted to defer?", which is the
  deployment question in medicine and the one thing a sampled answer token
  cannot express.
* **Top-k coverage and differential-diagnosis set size** describe the shortlist
  the model commits to, rather than only its top-1 pick.
* **Structural consistency** counts outputs that contradict themselves across
  tasks. For MedJev this is zero by construction; scoring the same way on the
  generative baseline turns "by construction" into a measured number instead
  of an assertion.

Writes `metrics.json` alongside a human-readable `metrics.md`.

Usage:
    python scripts/eval_medjev_metrics.py \\
        --states datas/medjev/testing.jsonl \\
        --decisions datas/eval/runs-medjev/checkpoint-2424/testing_decisions.jsonl \\
        --predictions datas/eval/runs-medjev/checkpoint-2424/testing_predictions.jsonl \\
        --ground-truth datas/eval/Preprocessed-omnimed-suite/testing.jsonl \\
        --baseline-predictions datas/eval/runs-sft/checkpoint-2424/testing_predictions.jsonl \\
        --output datas/eval/runs-medjev/checkpoint-2424/metrics.json
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from omnimedjev.engine.evaluate import parse_boxes, parse_multilabel
from omnimedjev.engine.medjev import TXL_CELL_TYPES, expected_coordinate, expected_level
from omnimedjev.engine.medjev_metrics import (coverage_at_k, ddx_set_size, entropy,
                                       expected_calibration_error, max_calibration_error,
                                       murphy_decomposition, risk_coverage, wilson_interval)

# Task families reported separately. Detection and counting both emit `count:*`
# decisions, so a qid alone is ambiguous -- `family_of` resolves it by the
# class name.
# Detection is reported only on request. Its metric was never competitive and
# the eval subset is too thin for the structural-consistency check to carry a
# claim on its own (only 5 ChestX-Det images carry both a multi-label and a
# detection answer, i.e. 7 comparison points).
FAMILIES = ("disease_diagnosis_classification", "multi_label_classification",
            "cell_counting", "regression")
DETECTION_FAMILY = "detection"


def subkind_of(qid: str) -> str:
    if qid == "diagnosis":
        return "choice"
    if qid.startswith("present:"):
        return "presence"
    if qid.startswith("count:"):
        return "count"
    if qid.startswith("detect:"):
        return "coordinate"
    return "score"


def family_of(qid: str) -> str:
    if qid == "diagnosis":
        return "disease_diagnosis_classification"
    if qid.startswith("present:"):
        return "multi_label_classification"
    if qid.startswith("detect:"):
        return "detection"
    if qid.startswith("count:"):
        class_name = qid.split(":", 1)[1]
        # `count:total` and `count:RBC/WBC/Platelets` belong to the cell-counting
        # task; `count:Nodule` and friends are the detection workflow's stage.
        return "cell_counting" if class_name in ("total", *TXL_CELL_TYPES) else "detection"
    if qid in ("ef", "grade"):
        return "regression"
    return "other"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _continuous_error(record: Mapping[str, Any]) -> float:
    """Error for a decision whose target we interpolated ourselves.

    Decided per decision, not per group: the detection family mixes presence
    (categorical), count (categorical) and coordinate (interpolated) questions,
    and the two kinds of target need different readouts.
    """
    if record["subkind"] == "coordinate":
        # Axis-normalized, so a horizontal and a vertical edge are comparable
        # despite different pixel sizes.
        return abs(expected_coordinate(record["probabilities"], 1.0)
                   - expected_coordinate(record["target"], 1.0))
    return abs(expected_level(record["probabilities"]) - expected_level(record["target"]))


def _continuous_bias(record: Mapping[str, Any]) -> float:
    if record["subkind"] == "coordinate":
        return (expected_coordinate(record["probabilities"], 1.0)
                - expected_coordinate(record["target"], 1.0))
    return expected_level(record["probabilities"]) - expected_level(record["target"])


def metric_block(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Every metric for one group of decisions."""
    if not records:
        return {}
    confidences = [max(record["probabilities"].values()) for record in records]
    correct = [record["top"] == record["gold"] for record in records]
    hits = sum(1 for flag in correct if flag)
    low, high = wilson_interval(hits, len(records))
    # Measured against the full distribution, not the top-label forecast, so it
    # differs slightly from the decomposition's binned reconstruction below.
    measured_brier = sum(record["brier"] for record in records) / len(records)

    block: dict[str, Any] = {
        "decisions": len(records),
        "accuracy": hits / len(records),
        "accuracy_ci95": [low, high],
        "nll": sum(record["nll"] for record in records) / len(records),
        "brier": measured_brier,
        "ece": expected_calibration_error(confidences, correct),
        "mce": max_calibration_error(confidences, correct),
        "mean_confidence": sum(confidences) / len(confidences),
        "mean_normalized_entropy": sum(record["entropy"] for record in records) / len(records),
    }
    decomposition = murphy_decomposition(confidences, correct)
    block["brier_decomposition_check"] = decomposition.pop("brier")
    block.update(decomposition)

    block.update(risk_coverage(confidences, correct))

    for tau in (0.5, 0.9):
        sizes = [ddx_set_size(list(record["probabilities"].values()), tau) for record in records]
        block[f"ddx_set_size_at_{tau}"] = sum(sizes) / len(sizes)

    # Top-k only where there are enough candidates for it to mean anything.
    max_k = max(len(record["probabilities"]) for record in records)
    topk_counts = collections.Counter(len(record["probabilities"]) for record in records)
    block["candidate_counts"] = dict(sorted(topk_counts.items()))
    for k in (1, 3, 5):
        if max_k < k:
            continue
        eligible = [r for r in records if len(r["probabilities"]) >= k]
        block[f"top{k}_coverage"] = (
            sum(1 for r in eligible if coverage_at_k(list(r["probabilities"].values()), r["gold_index"], k))
            / len(eligible)) if eligible else None

    # Continuous error, only for decisions whose target we interpolated.
    numeric = [r for r in records if r["target_kind"] == "discretization_interpolation"]
    if numeric:
        block["numeric_decisions"] = len(numeric)
        block["numeric_mae"] = sum(_continuous_error(r) for r in numeric) / len(numeric)
        block["numeric_bias"] = sum(_continuous_bias(r) for r in numeric) / len(numeric)
    return block


def collect(states: Mapping[str, dict], decisions: Mapping[str, dict]) -> list[dict[str, Any]]:
    """One record per answered decision, carrying everything the metrics need."""
    records: list[dict[str, Any]] = []
    for state_id, state in states.items():
        distributions = decisions.get(state_id, {}).get("distributions")
        if not distributions:
            continue
        for decision in state["decisions"]:
            qid = decision["qid"]
            probabilities = distributions.get(qid)
            if not probabilities:
                continue                       # the workflow never asked this one
            ids = [c["id"] for c in decision["candidates"]]
            probs = {cid: float(probabilities.get(cid, 0.0)) for cid in ids}
            target = decision["target"]
            gold = max(target, key=target.__getitem__)
            ordered = sorted(probs.values(), reverse=True)
            records.append({
                "qid": qid,
                "family": family_of(qid),
                "subkind": subkind_of(qid),
                "state_id": state_id,
                "source": state.get("source"),
                "probabilities": probs,
                "gold": gold,
                "gold_index": ids.index(gold),
                "top": max(probs, key=probs.__getitem__),
                "target": target,
                "target_kind": decision["target_kind"],
                "nll": -math.log(max(probs[gold], 1e-30)),
                "brier": sum((probs[cid] - float(target[cid])) ** 2 for cid in ids),
                "entropy": entropy(list(probs.values())),
            })
    return records


def structural_consistency(ground_truth: Path, predictions: Path, threshold: float = 0.5) -> dict[str, Any]:
    """Count outputs that contradict themselves across tasks for one image.

    Checked two ways, both of which MedJev's fold makes impossible and which
    the generative baseline can and does produce:

    * a finding localized in `detection` but missing from the image's
      `multi_label` answer (a box for something the model said was absent);
    * for TXL-PBC, a `cell_counting` total that disagrees with whether any
      boxes were emitted.

    Scored from prediction strings only, so the exact same function scores
    MedJev and the baseline.
    """
    index: dict[str, str] = {}
    for row in read_jsonl(predictions):
        index[str(row["uid"])] = row["prediction"]

    per_image: dict[str, dict[str, str]] = collections.defaultdict(dict)
    for row in read_jsonl(ground_truth):
        uid = str(row.get("uid"))
        if uid in index:
            per_image[str(row.get("image_path"))][str(row.get("task_type"))] = index[uid]

    boxes_without_label = labeled_without_box = 0
    count_disagreements = 0
    images_checked = comparison_points = 0
    violations: list[dict[str, Any]] = []

    for image, answers in per_image.items():
        detection = answers.get("detection")
        multilabel = answers.get("multi_label_classification")
        counting = answers.get("cell_counting")

        if detection is not None and multilabel is not None:
            images_checked += 1
            try:
                boxes = parse_boxes(detection)
            except Exception:
                boxes = []
            localized = {box.label for box in boxes if box.label}
            asserted = parse_multilabel(multilabel)
            comparison_points += len(localized | asserted)
            for label in sorted(localized - asserted):
                boxes_without_label += 1
                if len(violations) < 20:
                    violations.append({"image": image, "kind": "box_without_label", "label": label})
            for label in sorted(asserted - localized):
                labeled_without_box += 1
                if len(violations) < 20:
                    violations.append({"image": image, "kind": "label_without_box", "label": label})

        if detection is not None and counting is not None:
            images_checked += 1
            try:
                boxes = parse_boxes(detection)
            except Exception:
                boxes = []
            total = None
            for token in str(counting).replace(",", " ").split():
                if token.lstrip("-").isdigit():
                    total = int(token)
                    break
            if total is not None:
                if (total == 0) != (len(boxes) == 0):
                    count_disagreements += 1
                    if len(violations) < 20:
                        violations.append({"image": image, "kind": "count_box_disagreement",
                                           "count": total, "boxes": len(boxes)})

    total_violations = boxes_without_label + labeled_without_box + count_disagreements
    return {
        "images_checked": images_checked,
        "comparison_points": comparison_points,
        "box_without_label": boxes_without_label,
        "label_without_box": labeled_without_box,
        "count_box_disagreement": count_disagreements,
        "violations": total_violations,
        "violation_rate": total_violations / comparison_points if comparison_points else None,
        "examples": violations[:10],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--states", required=True, type=Path)
    ap.add_argument("--decisions", required=True, type=Path)
    ap.add_argument("--predictions", type=Path, default=None,
                    help="folded predictions; needed for the structural-consistency check")
    ap.add_argument("--ground-truth", type=Path,
                    default=Path("datas/eval/Preprocessed-omnimed-suite/testing.jsonl"))
    ap.add_argument("--baseline-predictions", type=Path, default=None,
                    help="a generative run's predictions, scored for consistency the same way")
    ap.add_argument("--include-detection", action="store_true",
                    help="also report the detection family and its sub-kinds")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    states = {s["state_id"]: s for s in read_jsonl(args.states)}
    decisions = {d["state_id"]: d for d in read_jsonl(args.decisions)}
    records = collect(states, decisions)

    report: dict[str, Any] = {
        "states": len(states),
        "decisions_scored": len(records),
        "families": {},
        "subkinds": {},
    }

    families = FAMILIES + ((DETECTION_FAMILY,) if args.include_detection else ())
    for family in families:
        group = [r for r in records if r["family"] == family]
        if not group:
            continue
        report["families"][family] = metric_block(group)

    # Sub-kinds matter inside detection: presence, count and coordinate are
    # three different questions and averaging them hides which one failed.
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for record in records:
        grouped[(record["family"], record["subkind"])].append(record)
    for (family, subkind), group in sorted(grouped.items()):
        if len(group) < 20:
            continue
        if family == DETECTION_FAMILY and not args.include_detection:
            continue
        report["subkinds"][f"{family}/{subkind}"] = metric_block(group)

    sources = collections.defaultdict(list)
    for record in records:
        if record["family"] == "disease_diagnosis_classification":
            sources[record["source"]].append(record)
    report["classification_by_source"] = {
        source: metric_block(group) for source, group in sorted(sources.items()) if len(group) >= 10
    }

    if args.predictions and args.predictions.exists():
        report["structural_consistency"] = {
            "medjev": structural_consistency(args.ground_truth, args.predictions)
        }
        if args.baseline_predictions and args.baseline_predictions.exists():
            report["structural_consistency"]["baseline"] = structural_consistency(
                args.ground_truth, args.baseline_predictions)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    lines = ["# MedJev metrics", "",
             f"- states: {report['states']}, decisions scored: {report['decisions_scored']}", ""]
    header = ("| group | n | acc | NLL | Brier | ECE | MCE | reliability | resolution | "
              "AURC | risk@90 | ddx@0.9 | top3 |")
    lines += [header, "|" + "---|" * 13]
    rows = list(report["families"].items()) + list(report["subkinds"].items())
    for name, m in rows:
        def g(key: str, fmt: str = "{:.4f}") -> str:
            value = m.get(key)
            return fmt.format(value) if isinstance(value, (int, float)) else "—"
        lines.append(f"| {name} | {m.get('decisions', 0)} | {g('accuracy')} | {g('nll')} | "
                     f"{g('brier')} | {g('ece')} | {g('mce')} | {g('reliability')} | {g('resolution')} | "
                     f"{g('aurc')} | {g('risk_at_90')} | {g('ddx_set_size_at_0.9', '{:.1f}')} | "
                     f"{g('top3_coverage')} |")
    if "structural_consistency" in report:
        lines += ["", "## Structural consistency", "",
                  "| run | violations | box w/o label | label w/o box | count/box mismatch | rate |",
                  "|---|---:|---:|---:|---:|---:|"]
        for name, block in report["structural_consistency"].items():
            rate = block.get("violation_rate")
            rate_text = f"{rate:.4f}" if isinstance(rate, float) else "—"
            lines.append(f"| {name} | {block['violations']} | {block['box_without_label']} | "
                         f"{block['label_without_box']} | {block['count_box_disagreement']} | "
                         f"{rate_text} |")
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:12]))
    print(f"\nwrote {args.output} and {args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
