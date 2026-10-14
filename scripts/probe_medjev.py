#!/usr/bin/env python3
"""Inference-only probes for the claims MedJev's architecture makes.

Nothing here retrains anything. Each probe perturbs the *candidate set* at
inference time and measures whether the model's answer follows the semantics
or the surface form. That is the one class of experiment a fixed classification
head physically cannot be put through: its logits are indexed by position, so
reordering the options reorders nothing, and removing one is undefined.

Probes, and the Jev element each one tests:

* **permute** (dynamic decision space) -- shuffle the candidates. A semantic
  decision returns the same distribution over the same *named* options; a
  position-indexed one follows the slot. Reported as total-variation distance
  from the unperturbed distribution plus top-1 agreement.
* **restrict_k** (dynamic decision space) -- keep a random subset of k
  candidates that still contains the answer. Accuracy on the subset is
  compared with chance (1/k). A model that has learned "which of these names
  fits" keeps finding it; one that memorised an index collapses to chance.
* **drop_gold** (complete distribution) -- delete the correct candidate. The
  remaining probability mass has to go somewhere; a distribution that was
  really comparing the options redistributes toward the runner-up, and the new
  argmax is usually the old second place.
* **add_none** (typed decisions / open-set) -- append "None of the above". The
  model was never trained with such an option, so this is a generalisation
  probe rather than a calibrated abstention, and it is reported as such: the
  question is whether P(none) rises on the images it actually gets wrong.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/probe_medjev.py \\
        --checkpoint output/medjev-lora \\
        --identity-decisions datas/eval/runs-medjev/checkpoint-2424/testing_decisions.jsonl \\
        --out-dir datas/eval/probes-medjev
"""

from __future__ import annotations

import argparse
import collections
import json

import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from peft import PeftModel
from rich.console import Console
from safetensors.torch import load_file
from transformers import AutoProcessor

from omnimedjev.engine.medjev import assert_marker_ids
from omnimedjev.engine.medjev_model import MedJevCollator, MedJevModel, build_batch
from omnimedjev.engine.train import (choose_attention_backends, choose_dtype, dtype_label,
                              load_model_with_attention_fallback, make_quant_config)

HEAD_FILE = "medjev_head.safetensors"
NONE_ID = "__none__"
NONE_TEXT = "None of the above"


# --------------------------------------------------------------------------
# Candidate-set transforms
# --------------------------------------------------------------------------

def transform(decision: Mapping[str, Any], kind: str, rng: random.Random) -> list[dict[str, str]] | None:
    """Return a new candidate list, or None if this probe does not apply.

    Candidate `id` values are preserved across every transform -- they are the
    only stable handle for matching a distribution before and after, which is
    exactly why the compiler keys candidates by id rather than by position.
    """
    candidates = [dict(c) for c in decision["candidates"]]
    k = len(candidates)
    gold = max(decision["target"], key=decision["target"].__getitem__)

    if kind == "identity":
        return candidates
    if kind == "permute":
        if k < 3:
            return None
        rng.shuffle(candidates)
        return candidates
    if kind.startswith("restrict"):
        size = int(kind.removeprefix("restrict"))
        if k < size + 2 or size < 2:
            return None
        others = [c for c in candidates if c["id"] != gold]
        keep = rng.sample(others, size - 1) + [next(c for c in candidates if c["id"] == gold)]
        rng.shuffle(keep)
        return keep
    if kind == "drop_gold":
        if k < 4:
            return None
        return [c for c in candidates if c["id"] != gold]
    if kind == "add_none":
        return candidates + [{"id": NONE_ID, "text": NONE_TEXT}]
    raise ValueError(f"unknown probe {kind!r}")


def roc_auc(scores: Sequence[float], positive: Sequence[bool]) -> float:
    """Rank-based AUC: P(a random positive scores above a random negative)."""
    pairs = sorted(zip(scores, positive), key=lambda item: item[0])
    positives = sum(1 for flag in positive if flag)
    negatives = len(positive) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    rank_sum = 0.0
    for rank, (_, flag) in enumerate(pairs, start=1):
        if flag:
            rank_sum += rank
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def total_variation(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    """Half the L1 distance between two distributions over the same ids."""
    keys = set(left) | set(right)
    return 0.5 * sum(abs(float(left.get(k, 0.0)) - float(right.get(k, 0.0))) for k in keys)


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------

def load_model(args, console: Console):
    dtype = choose_dtype(args.precision, console)
    quant = make_quant_config(not args.no_4bit, dtype)
    console.print(f"Loading {args.model} ({dtype_label(dtype)}, 4bit={quant is not None})")
    backbone = load_model_with_attention_fallback(
        args.model, choose_attention_backends(args.attn), console,
        torch_dtype=dtype, device_map=None, quantization_config=quant, trust_remote_code=True)
    if quant is not None:
        from peft import prepare_model_for_kbit_training
        backbone = prepare_model_for_kbit_training(backbone, use_gradient_checkpointing=False)
    backbone.config.use_cache = False
    checkpoint = Path(args.checkpoint)
    if (checkpoint / "adapter_config.json").exists():
        backbone = PeftModel.from_pretrained(backbone, str(checkpoint))
    model = MedJevModel(backbone, backbone.config.text_config.hidden_size)
    model.head.load_state_dict(load_file(str(checkpoint / HEAD_FILE)))
    model.to(torch.device(args.device)).eval()
    return model, dtype


@torch.no_grad()
def run(model, collator, states: Sequence[Mapping[str, Any]], device, dtype) -> list[dict[str, dict[str, float]]]:
    encoded = [collator.encode(state) for state in states]
    batch = build_batch(encoded, collator.pad_token_id, device=device)
    meta = batch.pop("meta")
    with torch.autocast("cuda", dtype=dtype):
        logits = model(**batch)["logits"].float()
    probabilities = logits.softmax(-1).cpu()
    out, cursor = [], 0
    for item in meta:
        per_question: dict[str, dict[str, float]] = {}
        for question, qid in enumerate(item["qid"]):
            ids = [str(c["id"]) for c in item["candidates"][question]]
            per_question[qid] = dict(zip(ids, probabilities[cursor + question, : len(ids)].tolist()))
        cursor += len(item["qid"])
        out.append(per_question)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--data", type=Path, default=Path("datas/medjev/testing.jsonl"))
    ap.add_argument("--identity-decisions", type=Path,
                    default=Path("datas/eval/runs-medjev/checkpoint-2424/testing_decisions.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("datas/eval/probes-medjev"))
    ap.add_argument("--probes", default="permute,restrict2,restrict4,drop_gold,add_none")
    ap.add_argument("--families", default="all",
                    help="comma-separated decision families, or 'all'")
    ap.add_argument("--model", default="models/medgemma-1.5-4b-it")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--precision", default="auto")
    ap.add_argument("--attn", default="auto")
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()

    console = Console()
    probes = [p.strip() for p in args.probes.split(",") if p.strip()]
    wanted = None if args.families == "all" else {f.strip() for f in args.families.split(",")}

    # Single-pass states only: the detection workflow asks a different question
    # set each pass, so its candidate sets are not fixed to perturb.
    all_states = [json.loads(line) for line in args.data.open(encoding="utf-8") if line.strip()]
    states = [s for s in all_states if not any(str(d["qid"]).startswith("detect:") for d in s["decisions"])]
    console.print(f"{len(states)} single-pass states of {len(all_states)}")

    identity: dict[str, dict[str, dict[str, float]]] = {}
    if args.identity_decisions.exists():
        for row in (json.loads(line) for line in args.identity_decisions.open(encoding="utf-8") if line.strip()):
            identity[row["state_id"]] = row["distributions"]
        console.print(f"loaded {len(identity)} unperturbed states from {args.identity_decisions}")

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    assert_marker_ids(processor.tokenizer)
    model, dtype = load_model(args, console)
    collator = MedJevCollator(processor, max_length=args.max_length, with_targets=False)
    device = torch.device(args.device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"checkpoint": str(args.checkpoint), "states": len(states), "probes": {}}
    started = time.perf_counter()

    for probe in probes:
        rng = random.Random(args.seed)
        # Build one transformed state per probe, keyed back to its decisions so
        # the analysis can match candidate ids across the two runs.
        perturbed, index = [], []
        orders: dict[str, dict[str, list[str]]] = {}
        for state in states:
            new_decisions, new_qids = [], []
            for decision in state["decisions"]:
                family = ("disease_diagnosis_classification" if decision["qid"] == "diagnosis" else
                          "multi_label_classification" if decision["qid"].startswith("present:") else
                          "cell_counting" if decision["qid"].startswith("count:") else "regression")
                if wanted is not None and family not in wanted:
                    continue
                candidates = transform(decision, probe, rng)
                if candidates is None:
                    continue
                new_decisions.append({**decision, "candidates": candidates, "target": {}})
                new_qids.append(decision["qid"])
                # Record the order the model actually saw. Position-bias can only
                # be measured against THIS order; the compiled state's order is a
                # property of the data, not of what was asked.
                orders.setdefault(state["state_id"], {})[decision["qid"]] = [str(c["id"]) for c in candidates]
            if new_decisions:
                perturbed.append({**state, "decisions": new_decisions})
                index.append((state["state_id"], new_qids))

        answers: list[dict[str, dict[str, float]]] = []
        for start in range(0, len(perturbed), args.batch_size):
            answers += run(model, collator, perturbed[start:start + args.batch_size], device, dtype)
        console.print(f"  {probe}: {len(perturbed)} states, "
                      f"{sum(len(q) for _, q in index)} decisions")
        summary["probes"][probe] = {"states": len(perturbed),
                                    "decisions": sum(len(q) for _, q in index),
                                    "orders": orders,
                                    "answers": {state_id: answers[i]
                                                for i, (state_id, _) in enumerate(index)}}

    (args.out_dir / "raw.json").write_text(
        json.dumps(summary, ensure_ascii=False, default=str), encoding="utf-8")

    # ---------------------------------------------------------------- analysis
    by_qid = {state["state_id"]: {d["qid"]: d for d in state["decisions"]} for state in states}
    report: dict[str, Any] = {"checkpoint": str(args.checkpoint), "states": len(states),
                              "seconds": round(time.perf_counter() - started, 1), "probes": {}}

    def gold_of(state_id: str, qid: str) -> str:
        target = by_qid[state_id][qid]["target"]
        return max(target, key=target.__getitem__)

    def base_of(state_id: str, qid: str) -> dict[str, float] | None:
        return (identity.get(state_id) or {}).get(qid)

    if "permute" in summary["probes"]:
        probe = summary["probes"]["permute"]
        orders = probe.get("orders") or {}
        distances, agreements, mean_ranks = [], 0, []
        per_family: dict[str, list[tuple[float, bool, float]]] = collections.defaultdict(list)
        total = 0
        for state_id, answers in probe["answers"].items():
            for qid, distribution in answers.items():
                base = base_of(state_id, qid)
                if not base:
                    continue
                total += 1
                distance = total_variation(distribution, base)
                agrees = max(distribution, key=distribution.__getitem__) == max(base, key=base.__getitem__)
                distances.append(distance)
                agreements += int(agrees)
                # Rank of the chosen candidate WITHIN THE ORDER THE MODEL SAW.
                # Position bias can only be read off the perturbed order: the
                # compiled order is a property of the data, not of the question
                # that was asked, so measuring against it says nothing about
                # whether the model leans on position.
                order = orders.get(state_id, {}).get(qid) or []
                rank = float("nan")
                if len(order) > 1:
                    rank = order.index(max(distribution, key=distribution.__getitem__)) / (len(order) - 1)
                    mean_ranks.append(rank)
                family = ("diagnosis" if qid == "diagnosis" else "present" if qid.startswith("present:")
                          else "count" if qid.startswith("count:") else "score")
                per_family[family].append((distance, agrees, rank))
        n = max(total, 1)
        report["probes"]["permute"] = {
            "decisions": total,
            "mean_total_variation": sum(distances) / n if distances else None,
            "max_total_variation": max(distances) if distances else None,
            "top1_agreement": agreements / n,
            # 0.5 == no preference for early or late positions.
            "mean_normalized_chosen_rank": (sum(mean_ranks) / len(mean_ranks)) if mean_ranks else None,
            "by_family": {
                family: {
                    "decisions": len(rows),
                    "mean_total_variation": sum(r[0] for r in rows) / len(rows),
                    "top1_agreement": sum(1 for r in rows if r[1]) / len(rows),
                    "mean_normalized_chosen_rank": (sum(r[2] for r in rows if r[2] == r[2]) /
                                                    max(sum(1 for r in rows if r[2] == r[2]), 1)),
                } for family, rows in sorted(per_family.items())
            },
        }

    for key in ("restrict2", "restrict4"):
        if key not in summary["probes"]:
            continue
        size = int(key.removeprefix("restrict"))
        correct = total = base_correct = base_total = 0
        for state_id, answers in summary["probes"][key]["answers"].items():
            for qid, distribution in answers.items():
                total += 1
                if max(distribution, key=distribution.__getitem__) == gold_of(state_id, qid):
                    correct += 1
                # Control: the SAME decisions scored on their full candidate set.
                # Without it the subset accuracy has no reference -- a model that
                # was already wrong on these rows cannot be said to have degraded.
                base = base_of(state_id, qid)
                if base:
                    base_total += 1
                    if max(base, key=base.__getitem__) == gold_of(state_id, qid):
                        base_correct += 1
        report["probes"][key] = {
            "decisions": total,
            "accuracy_on_subset": correct / max(total, 1),
            "accuracy_full_set_same_rows": base_correct / max(base_total, 1),
            "chance": 1 / size,
        }

    if "drop_gold" in summary["probes"]:
        runner_up, before, after, total = 0, [], [], 0
        for state_id, answers in summary["probes"]["drop_gold"]["answers"].items():
            for qid, distribution in answers.items():
                base = base_of(state_id, qid)
                if not base:
                    continue
                total += 1
                gold = gold_of(state_id, qid)
                ranked = sorted((c for c in base if c != gold), key=lambda c: -base[c])
                if not ranked:
                    continue
                # If the model was really comparing the options, deleting the
                # answer should hand its mass to the runner-up -- so the
                # runner-up's probability after should exceed before.
                before.append(float(base.get(ranked[0], 0.0)))
                after.append(float(distribution.get(ranked[0], 0.0)))
                if max(distribution, key=distribution.__getitem__) == ranked[0]:
                    runner_up += 1
        report["probes"]["drop_gold"] = {
            "decisions": total,
            "new_top_is_old_runner_up": runner_up / max(total, 1),
            "mean_p_runner_up_before": sum(before) / len(before) if before else None,
            "mean_p_runner_up_after": sum(after) / len(after) if after else None,
        }

    if "add_none" in summary["probes"]:
        none_scores, wrong = [], []
        for state_id, answers in summary["probes"]["add_none"]["answers"].items():
            base = identity.get(state_id) or {}
            for qid, distribution in answers.items():
                base_distribution = base.get(qid)
                if not base_distribution:
                    continue
                none_scores.append(float(distribution.get(NONE_ID, 0.0)))
                wrong.append(max(base_distribution, key=base_distribution.__getitem__)
                             != gold_of(state_id, qid))
        report["probes"]["add_none"] = {
            "decisions": len(none_scores),
            "mean_p_none": sum(none_scores) / max(len(none_scores), 1),
            "p_none_on_correct": (sum(s for s, w in zip(none_scores, wrong) if not w)
                                  / max(sum(1 for w in wrong if not w), 1)),
            "p_none_on_wrong": (sum(s for s, w in zip(none_scores, wrong) if w)
                                / max(sum(1 for w in wrong if w), 1)),
            "auc_p_none_predicts_error": roc_auc(none_scores, wrong),
        }

    (args.out_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    lines = ["# MedJev probes (inference only)", "",
             f"- checkpoint: `{args.checkpoint}`", f"- single-pass states: {report['states']}", ""]
    for name, block in report["probes"].items():
        lines.append(f"## {name}")
        for metric, value in block.items():
            if isinstance(value, float):
                lines.append(f"- {metric}: {value:.4f}")
            else:
                lines.append(f"- {metric}: {value}")
        lines.append("")
    (args.out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote {args.out_dir}/summary.md, summary.json, raw.json")


if __name__ == "__main__":
    main()
