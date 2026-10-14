#!/usr/bin/env python3
"""Compile the OmniMed-Suite annotations into MedJev decision states.

WHY this exists: the SFT pipeline stores one row per *task* -- a chest X-ray
that is both multi-label and detection becomes two independent training rows
that never meet. MedJev needs the opposite: one *state* (the image) carrying
every decision that can be asked about it, because "one state, many parallel
decisions" and "code composes the workflow" are the claims the interface rests
on. This script performs that regrouping.

One output record == one medical state == one image plus its decisions:

    {"state_id": ..., "split": ..., "source": ..., "modality": ...,
     "images": [...], "image_size": [W, H], "legacy_uids": {...},
     "decisions": [{"qid", "type", "instructions", "candidates", "target",
                    "target_kind", "legacy_uid"}, ...]}

The decision space itself -- candidate sets, question wording, caps -- lives in
`omnimedjev/engine/medjev.py`, NOT here. Inference rebuilds the detection workflow's
count and coordinate decisions on the fly, so compiler and inference must never
be allowed to drift apart. This file only maps annotations onto that contract.

Three mapping rules that are easy to get wrong and are therefore explicit:

* **Noul is Choice.** A presence question compiles to a 2-candidate Choice over
  {No, Yes}, so Boolean, Choice and Score share one cross entropy and one head.

* **`target_kind` is never implicit.** A label has exactly one right answer; an
  ejection fraction of 54.3 gets an interpolated 0.7/0.3 split that is a
  discretization artifact, not clinical uncertainty. Marking which is which
  keeps the two from being conflated in the write-up.

* **ChestX-Det's multi-label target is deduplicated.** Its `raw_answer` is the
  per-box symbol list, so 1769 of 3578 rows repeat a label (one row lists
  'Nodule' 25 times). The evaluator is immune because it compares sets, but a
  Noul target built from that list would contradict itself.

Usage:

    python scripts/build_medjev_data.py \\
        --src datas/preprocessed/Preprocessed-omnimed-suite/train.jsonl \\
        --out datas/medjev/train.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

from omnimedjev.engine.medjev import (COUNT_CAP, DETECT_COUNT_CAP, DETECTION_CLASSES, EF_MAX,
                               MAX_LOCALIZED_BOXES, RETINA_LEVELS, TXL_CELL_TYPES,
                               coordinate_decisions, count_decision, counting_decisions,
                               presence_decision, score_levels_decision)

try:
    from PIL import Image
except ImportError:  # only detection states need image dimensions
    Image = None  # type: ignore[assignment]


# "no findings" is spelled several ways upstream: chestmnist ships the literal
# string "none", other converters emit "" or "[]". Same union the evaluator's
# EMPTY_LABELS and make_train_subset.py's EMPTY_MARKERS already use, so a
# sentinel that scores as empty downstream also compiles as empty here.
EMPTY_MARKERS = {"", "[]", "{}", "null", "none", "no finding", "no findings", "n/a", "na"}

# Detection is off: it is neither trained nor evaluated.
#
# Turning this on would additionally compile `detect:*` coordinate Scores,
# `count:{finding}` decisions and the TXL-PBC presence questions. All three are
# consumed only by the detection workflow, and the coordinate head was measured
# to collapse to a near-constant box (77.7% of predictions overlapped no gold
# box at all), so they were spending roughly a quarter of a ChestX-Det state's
# loss on something nothing scores.
#
# The detection row is still READ either way: TXL-PBC's per-type box counts
# exist nowhere else, and `cell_counting` needs them.
EMIT_DETECTION = False

CLAMPED_COUNTS: collections.Counter[str] = collections.Counter()


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def as_list(value: Any) -> list[str]:
    """`raw_answer`/`choices` arrive as a real list, a JSON string, or a scalar."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [str(v) for v in parsed]
            except json.JSONDecodeError:
                pass
        return [text] if text else []
    if isinstance(value, dict):
        return [str(k) for k, v in value.items() if v]
    return [str(value)]


def positive_labels(value: Any) -> set[str]:
    """The set of labels actually asserted, with the empty sentinels dropped.

    `set()` is load-bearing for ChestX-Det, whose `raw_answer` is the per-box
    symbol list and therefore repeats a label once per box (one row lists
    'Nodule' 25 times). A Noul target built from the raw list would assert
    both "present" and "absent" for the same finding.
    """
    return {item for item in as_list(value) if item.strip().lower() not in EMPTY_MARKERS}


def extract_question(prompt: str, default: str) -> str:
    """Pull the human question out of the generative prompt.

    The upstream prompt is a three-part blob (instruction / question /
    options). MedJev supplies its own candidates, so only the middle part is
    reused -- keeping the dataset's own wording without dragging in the
    "return the option letter" scaffolding that the decision interface
    replaces.
    """
    match = re.search(r"Question:\s*(.+?)(?:\nOptions:|\nAnswer|\Z)", str(prompt or ""), flags=re.S)
    if match:
        question = " ".join(match.group(1).split())
        if question:
            return question
    return default


def image_dimensions(path: str) -> tuple[int, int]:
    if Image is None:
        raise RuntimeError("Pillow is required to compile detection states.")
    with Image.open(path) as handle:
        return handle.size


# --------------------------------------------------------------------------
# Per-family compilation
# --------------------------------------------------------------------------

def decisions_for_classification(row: dict[str, Any]) -> list[dict[str, Any]]:
    choices = as_list(row.get("choices"))
    answer = str(row.get("raw_answer") or row.get("answer") or "").strip()
    if len(choices) < 2:
        raise ValueError(f"{row.get('uid')}: classification needs >=2 choices")
    if answer not in choices:
        raise ValueError(f"{row.get('uid')}: answer {answer!r} is not among {choices}")
    from omnimedjev.engine.medjev import one_hot_target
    candidates = [{"id": name, "text": name} for name in choices]
    return [{
        "qid": "diagnosis", "type": "choice",
        "instructions": extract_question(row.get("prompt", ""), "Which class does this image belong to?"),
        "candidates": candidates,
        "target": one_hot_target(candidates, answer),
        "target_kind": "one_hot", "legacy_uid": row.get("uid"),
    }]


def decisions_for_multilabel(row: dict[str, Any]) -> list[dict[str, Any]]:
    labels = sorted(set(as_list(row.get("choices"))))
    positives = positive_labels(row.get("raw_answer"))
    unknown = positives - set(labels)
    if unknown:
        raise ValueError(f"{row.get('uid')}: positives {sorted(unknown)} are not in the candidate set")
    modality = str(row.get("modality") or "medical")
    return [presence_decision(label, modality, label in positives, row.get("uid")) for label in labels]


def decisions_for_camus(row: dict[str, Any]) -> list[dict[str, Any]]:
    value = float(row["raw_answer"])
    if not 0 <= value <= EF_MAX:
        raise ValueError(f"{row.get('uid')}: EF {value} out of range")
    return [score_levels_decision(
        "ef", "What is the left ventricular ejection fraction, in percent?",
        [str(k) for k in range(EF_MAX + 1)], value, row.get("uid"))]


def decisions_for_retina(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [score_levels_decision(
        "grade", "What is the diabetic retinopathy severity grade on the 0-4 international clinical scale?",
        [str(k) for k in range(RETINA_LEVELS)], float(row["raw_answer"]), row.get("uid"))]


def detection_class_list(source: str, multilabel: dict[str, Any] | None) -> list[str]:
    """The full candidate set for a detection state's presence stage.

    Preferred source is the joined multi-label row's `choices`, so the class
    names come from the data rather than from a constant.
    """
    if multilabel is not None and as_list(multilabel.get("choices")):
        return as_list(multilabel["choices"])
    if source not in DETECTION_CLASSES:
        raise ValueError(f"no detection candidate set known for source {source!r}")
    return DETECTION_CLASSES[source]


def group_by_image(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        path = str(row.get("image_path") or row.get("image") or "")
        if not path:
            raise ValueError(f"{row.get('uid')}: row has no image path")
        grouped[path].append(row)
    return grouped


def build_state(image_path: str, rows: list[dict[str, Any]], split: str) -> dict[str, Any] | None:
    """Turn every annotation about one image into a single decision state."""
    by_task: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_task.setdefault(str(row.get("task_type")), row)

    decisions: list[dict[str, Any]] = []
    legacy_uids: dict[str, str] = {}
    size: tuple[int, int] | None = None
    source = str(rows[0].get("source", ""))

    def note(task: str, uid: Any) -> None:
        if uid:
            legacy_uids[task] = str(uid)

    classification = by_task.get("disease_diagnosis_classification")
    if classification is not None:
        note("disease_diagnosis_classification", classification.get("uid"))
        decisions += decisions_for_classification(classification)

    multilabel = by_task.get("multi_label_classification")
    if multilabel is not None:
        note("multi_label_classification", multilabel.get("uid"))
        decisions += decisions_for_multilabel(multilabel)

    counting = by_task.get("cell_counting")
    detection = by_task.get("detection")
    boxes_by_class: dict[str, list] = {}
    if detection is not None and isinstance(detection.get("raw_answer"), dict):
        boxes_by_class = {str(k): list(v) for k, v in detection["raw_answer"].items()}

    if counting is not None:
        note("cell_counting", counting.get("uid"))
        if detection is not None:
            # The per-type breakdown lives only in the detection row's box dict,
            # so the two rows must be joined on the image to compile it.
            total = int(counting["raw_answer"])
            counts = {name: len(boxes_by_class.get(name, [])) for name in TXL_CELL_TYPES}
            counts["total"] = total
            if sum(counts[name] for name in TXL_CELL_TYPES) != total:
                raise ValueError(f"{counting.get('uid')}: per-type counts {counts} do not sum to {total}")
            if total > COUNT_CAP:
                raise ValueError(f"{counting.get('uid')}: total {total} exceeds COUNT_CAP={COUNT_CAP}")
            decisions += counting_decisions(counts, counting.get("uid"))
        else:
            # No detection row, so the per-type split is genuinely unknown.
            decisions += counting_decisions({"total": int(counting["raw_answer"])}, counting.get("uid"))

    if detection is not None and EMIT_DETECTION:
        note("detection", detection.get("uid"))
        modality = str(detection.get("modality") or "medical")
        class_list = detection_class_list(source, multilabel)
        unknown = set(boxes_by_class) - set(class_list)
        if unknown:
            raise ValueError(f"{image_path}: box classes {sorted(unknown)} are outside the candidate set")
        if multilabel is not None and positive_labels(multilabel.get("raw_answer")) != \
                {c for c in class_list if boxes_by_class.get(c)}:
            # The multi-label positives and the box dict's keys describe the same
            # finding set. If they disagreed, one qid would answer two questions.
            raise ValueError(f"{image_path}: multi-label positives disagree with the detection box keys")

        existing_qids = {d["qid"] for d in decisions}
        # Presence is emitted for EVERY class, not just the boxed ones. An image
        # whose radiograph is clean has an empty box dict, and compiling
        # "nothing present" as "no decisions at all" would both drop the row and
        # leave the detector unable to answer for that image.
        for class_name in class_list:
            qid = f"present:{class_name}"
            if qid in existing_qids:
                continue
            decisions.append(presence_decision(class_name, modality,
                                               bool(boxes_by_class.get(class_name)), detection.get("uid")))

        # The localization budget is per STATE, not per class. Per class, one
        # box-heavy finding (Nodule reaches 36 boxes) multiplies the sequence
        # into the thousands, because every localized box costs four
        # COORD_BINS-candidate Scores.
        ranked = sorted(
            (((float(b[2]) - float(b[0])) * (float(b[3]) - float(b[1]))), name, b)
            for name, blist in boxes_by_class.items() for b in blist)
        if len(ranked) > MAX_LOCALIZED_BOXES:
            ranked = ranked[len(ranked) - MAX_LOCALIZED_BOXES:]
        per_class: dict[str, list[Sequence[float]]] = collections.defaultdict(list)
        for _, name, box in ranked:
            per_class[name].append(box)

        size = image_dimensions(image_path) if ranked else None
        for class_name in sorted(per_class):
            assert size is not None
            all_boxes = boxes_by_class[class_name]
            if len(all_boxes) > DETECT_COUNT_CAP:
                CLAMPED_COUNTS[class_name] += 1
            if f"count:{class_name}" not in existing_qids:
                decisions.append(count_decision(class_name, min(len(all_boxes), DETECT_COUNT_CAP),
                                                detection.get("uid")))
            # Only the budgeted slots get emitted; the rest would carry no target.
            slots = coordinate_decisions(class_name, f"detect:{class_name}", detection.get("uid"),
                                         per_class[class_name], size)
            decisions += slots[: 4 * len(per_class[class_name])]

    regression = by_task.get("regression")
    if regression is not None:
        note("regression", regression.get("uid"))
        if source == "CAMUS":
            decisions += decisions_for_camus(regression)
            if size is None:
                size = image_dimensions(image_path)
        else:
            decisions += decisions_for_retina(regression)

    report = by_task.get("report_generation")
    if report is not None:
        # Report generation is an unbounded string. It is deliberately not a
        # decision; the uid is recorded so the generative control run can be
        # joined back to the same state.
        note("report_generation", report.get("uid"))

    if not decisions:
        return None

    qids = [d["qid"] for d in decisions]
    if len(set(qids)) != len(qids):
        duplicated = sorted({q for q in qids if qids.count(q) > 1})
        raise ValueError(f"{image_path}: duplicate decision ids {duplicated}")
    for decision in decisions:
        if not decision["target"]:
            raise ValueError(f"{image_path}:{decision['qid']}: compiled decision has no target")

    return {
        "state_id": Path(image_path).stem,
        "split": split,
        "source": source,
        "modality": str(rows[0].get("modality", "")),
        "images": [image_path],
        "image_size": list(size) if size else None,
        "legacy_uids": legacy_uids,
        "decisions": decisions,
    }


def split_of(path: Path) -> str:
    stem = path.stem.lower()
    for name in ("train", "validation", "testing", "test", "val"):
        if name in stem:
            return {"test": "testing", "val": "validation"}.get(name, name)
    raise ValueError(f"cannot infer split from {path.name}")


def cap_per_source(states: list[dict[str, Any]], per_source: int, seed: int) -> list[dict[str, Any]]:
    """Optionally cap how many states each source dataset contributes.

    Left at 0 for the head-to-head run: the training split fed in here is
    already the task-balanced 29,842-row subset the generative baseline used,
    so keeping it untouched is what makes the two runs comparable. The cap
    exists for building smaller dev splits, not for re-balancing.
    """
    if per_source <= 0:
        return states
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for state in states:
        grouped[str(state["source"])].append(state)
    picked: list[dict[str, Any]] = []
    for items in grouped.values():
        picked.extend(items if len(items) <= per_source else rng.sample(items, per_source))
    rng.shuffle(picked)
    return picked


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="a Preprocessed-omnimed-suite jsonl split")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--per-source", type=int, default=0,
                    help="cap states per source dataset (0 = keep everything)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = read_rows(args.src)
    split = split_of(args.src)
    print(f"  source: {args.src}  ({len(rows)} rows, split={split})")

    grouped = group_by_image(rows)
    print(f"  grouped into {len(grouped)} image states")

    states: list[dict[str, Any]] = []
    for image_path, group in sorted(grouped.items()):
        state = build_state(image_path, group, split)
        if state is not None:
            states.append(state)
    states = cap_per_source(states, args.per_source, args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for state in states:
            handle.write(json.dumps(state, ensure_ascii=False) + "\n")

    decision_types = collections.Counter()
    per_source = collections.Counter()
    per_kind = collections.Counter()
    for state in states:
        per_source[state["source"]] += 1
        for decision in state["decisions"]:
            decision_types[decision["type"]] += 1
            per_kind[decision["target_kind"]] += 1

    print(f"\n  wrote {len(states)} states -> {args.out}")
    print(f"  decisions: {sum(decision_types.values())}  {dict(decision_types)}")
    print(f"  target kinds: {dict(per_kind)}")
    print("  states per source:")
    for source, count in sorted(per_source.items(), key=lambda kv: -kv[1]):
        print(f"    {source:<20} {count:>7}")
    if CLAMPED_COUNTS:
        print(f"  count targets clamped to {DETECT_COUNT_CAP} ('{DETECT_COUNT_CAP} or more'): "
              f"{sum(CLAMPED_COUNTS.values())} instances, {dict(CLAMPED_COUNTS)}")


if __name__ == "__main__":
    main()
