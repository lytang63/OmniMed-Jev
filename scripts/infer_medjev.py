#!/usr/bin/env python3
"""Run MedJev inference and fold the resulting distributions into predictions.

Writes two files:

* `{split}_decisions.jsonl` -- every state's full distributions. This is the
  evidence for the calibration metrics; the folded predictions cannot show
  whether a probability was honest.
* `{split}_predictions.jsonl` -- `{"uid", "prediction", "task_type", "case_id"}`
  exactly as `omnimedjev infer` writes it, so `omnimedjev evaluate` scores it unchanged and
  the numbers sit directly beside the generative baseline's.

The detection workflow runs as code, not as one monolithic forward pass: ask
presence, then -- only for the findings the model reports as present -- ask how
many and where. A finding judged absent is never localized, so "no
pneumothorax" and "a pneumothorax box" cannot both appear in the output. That
consistency is structural, not a behaviour the model is asked to maintain.

Usage:
    python scripts/infer_medjev.py --checkpoint runs/medjev_lora \\
        --data datas/medjev/testing.jsonl --out-dir datas/eval/runs-medjev/checkpoint-2424
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
from peft import PeftModel
from rich.console import Console
from safetensors.torch import load_file
from transformers import AutoProcessor

from omnimedjev.engine.medjev import (assert_marker_ids, coordinate_decisions, count_decision,
                               counting_decisions, fold_state, localization_budget)
from omnimedjev.engine.medjev_model import MedJevCollator, MedJevModel, build_batch
from omnimedjev.engine.train import (choose_attention_backends, choose_dtype, dtype_label,
                              load_model_with_attention_fallback, make_quant_config)

HEAD_FILE = "medjev_head.safetensors"


def workflow(state: Mapping[str, Any], threshold: float) -> Iterator[list[dict[str, Any]]]:
    """Yield the decision lists for one state, pass by pass.

    Non-detection states are a single pass: their decisions do not depend on
    any earlier answer. Detection states are the composed workflow, and the
    generator is sent the accumulated distributions so it can decide what to
    ask next -- that dependency is the whole point of the exercise.
    """
    qids = {str(d["qid"]) for d in state["decisions"]}
    presence = [d for d in state["decisions"] if str(d["qid"]).startswith("present:")]
    # Only a state that actually carries localization decisions gets the
    # workflow. Deciding by `present:` alone would drag every multi-label state
    # (ChestMNIST is nothing but presence questions) into asking count and
    # coordinate questions it was never trained on -- three passes instead of
    # one, for answers nothing consumes.
    has_localization = any(qid.startswith("detect:") for qid in qids)
    if not presence or not has_localization:
        yield list(state["decisions"])
        return

    answers = yield [dict(d, target={}) for d in presence]
    present = [(str(d["qid"]).split(":", 1)[1], float(answers[str(d["qid"])].get("yes", 0.0)))
               for d in presence if float(answers[str(d["qid"])].get("yes", 0.0)) >= threshold]
    names = [name for name, _ in present]

    count_qids = {str(d["qid"]) for d in state["decisions"] if str(d["qid"]).startswith("count:")}
    if "count:total" in count_qids:
        # TXL-PBC: the per-type questions ARE the workflow's count stage.
        yield counting_decisions(None, None)
    else:
        yield [count_decision(name) for name in names]

    # The box budget is allocated across findings by code, not by the model,
    # and matches the per-state cap the training data was compiled under.
    budget = localization_budget(present)
    coordinates: list[dict[str, Any]] = []
    for name in names:
        slots = budget.get(name, 0)
        if slots:
            coordinates += coordinate_decisions(name, f"detect:{name}")[: 4 * slots]
    yield coordinates


def advance(generator: Iterator[list[dict[str, Any]]], answers: Mapping[str, Mapping[str, float]],
            ) -> list[dict[str, Any]] | None:
    """Drive a workflow generator to its next pass that actually asks something.

    An empty decision list means "this pass has no questions" -- which happens
    whenever the model reports no finding as present, so there is no count or
    localization to ask for. Returning that empty list as pending work would
    leave the driver loop with a pass it can never make progress on, because
    an empty list is not None but is falsy. Step over such passes here instead.

    Returns the next non-empty decision list, or None once the workflow is
    finished.
    """
    try:
        decisions = generator.send(answers)
    except StopIteration:
        return None
    while not decisions:
        try:
            decisions = generator.send(answers)
        except StopIteration:
            return None
    return decisions


@torch.no_grad()
def run_pass(model: MedJevModel, collator: MedJevCollator, decisions_by_state: Sequence[list[dict[str, Any]]],
             states: Sequence[Mapping[str, Any]], device: torch.device,
             dtype: torch.dtype) -> list[dict[str, dict[str, float]]]:
    """One forward pass over a batch; returns `{qid: {candidate_id: prob}}` per state."""
    encoded = []
    for state, decisions in zip(states, decisions_by_state):
        if not decisions:
            encoded.append(None)
            continue
        encoded.append(collator.encode({**state, "decisions": decisions}))

    live = [item for item in encoded if item is not None]
    if not live:
        return [{} for _ in states]

    # Encode each state separately, then batch them via build_batch.
    batch = build_batch(live, collator.pad_token_id, device=device)
    meta = batch.pop("meta")
    with torch.autocast("cuda", dtype=dtype):
        outputs = model(**batch)
    logits = outputs["logits"].float()

    probabilities = logits.softmax(-1).cpu()
    # Rows are questions, flattened across the batch in state order, so walk the
    # cursor forward rather than assuming one row per state.
    answers: list[dict[str, dict[str, float]]] = []
    cursor = 0
    for item in meta:
        per_question: dict[str, dict[str, float]] = {}
        for question, qid in enumerate(item["qid"]):
            ids = [str(c["id"]) for c in item["candidates"][question]]
            # Padded candidate slots carry -inf and softmax to exactly 0; slicing
            # to the real candidate count keeps them out of the distribution.
            values = probabilities[cursor + question, : len(ids)].tolist()
            per_question[qid] = dict(zip(ids, values))
        cursor += len(item["qid"])
        answers.append(per_question)

    out: list[dict[str, dict[str, float]]] = []
    live_index = 0
    for item in encoded:
        if item is None:
            out.append({})
        else:
            out.append(answers[live_index])
            live_index += 1
    return out


def load_checkpoint(args, console: Console):
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
    adapters = checkpoint / "adapter_config.json"
    if adapters.exists():
        backbone = PeftModel.from_pretrained(backbone, str(checkpoint))
        console.print(f"Attached LoRA adapters from {checkpoint}")
    else:
        console.print(f"No adapter_config.json in {checkpoint}; running the head on the bare backbone")

    head_path = checkpoint / HEAD_FILE if checkpoint.is_dir() else checkpoint
    if not head_path.exists():
        raise FileNotFoundError(f"missing decision head: {head_path}")
    model = MedJevModel(backbone, backbone.config.text_config.hidden_size)
    model.head.load_state_dict(load_file(str(head_path)))
    model.to(torch.device(args.device)).eval()
    return model, dtype


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--model", default="models/medgemma-1.5-4b-it")
    ap.add_argument("--split", default=None, help="defaults to the data file's stem")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=3072)
    ap.add_argument("--image-size", type=int, default=448)
    ap.add_argument("--presence-threshold", type=float, default=0.5)
    ap.add_argument("--precision", default="auto")
    ap.add_argument("--attn", default="auto")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    console = Console()
    split = args.split or args.data.stem
    states = [json.loads(line) for line in args.data.open(encoding="utf-8") if line.strip()]
    if args.limit:
        states = states[: args.limit]
    console.print(f"{len(states)} states from {args.data}")

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    assert_marker_ids(processor.tokenizer)

    model, dtype = load_checkpoint(args, console)
    collator = MedJevCollator(processor, image_size=args.image_size, max_length=args.max_length,
                              with_targets=False)
    device = torch.device(args.device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    decision_rows, prediction_rows = [], []
    started = time.perf_counter()

    for start in range(0, len(states), args.batch_size):
        group = states[start:start + args.batch_size]
        accumulated: list[dict[str, dict[str, float]]] = [{} for _ in group]
        generators = [workflow(state, args.presence_threshold) for state in group]
        pending: list[list[dict[str, Any]] | None] = []
        for generator in generators:
            try:
                pending.append(next(generator))
            except StopIteration:
                pending.append(None)

        # Truthiness, not `is not None`: both the loop guard and the active set
        # must agree on what counts as work, or an empty pass would satisfy the
        # guard while never appearing in `active`.
        while any(pending):
            active = [index for index, item in enumerate(pending) if item]
            answers = run_pass(model, collator, [pending[i] for i in active],
                               [group[i] for i in active], device, dtype)
            for slot, index in enumerate(active):
                accumulated[index].update(answers[slot])
            for slot, index in enumerate(active):
                pending[index] = advance(generators[index], accumulated[index])

        for state, distributions in zip(group, accumulated):
            decision_rows.append({
                "state_id": state["state_id"], "split": split, "source": state["source"],
                "legacy_uids": state.get("legacy_uids", {}),
                "distributions": distributions,
            })
            folded = fold_state(state, distributions, presence_threshold=args.presence_threshold)
            for task, prediction in folded.items():
                uid = (state.get("legacy_uids") or {}).get(task)
                if not uid:
                    continue
                prediction_rows.append({"uid": uid, "prediction": prediction,
                                        "task_type": task, "case_id": state["state_id"]})
        console.print(f"  {min(start + args.batch_size, len(states))}/{len(states)}")

    decisions_path = args.out_dir / f"{split}_decisions.jsonl"
    predictions_path = args.out_dir / f"{split}_predictions.jsonl"
    with decisions_path.open("w", encoding="utf-8") as handle:
        for row in decision_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    tasks: dict[str, int] = {}
    for row in prediction_rows:
        tasks[row["task_type"]] = tasks.get(row["task_type"], 0) + 1
    console.print(json.dumps({
        "decisions": str(decisions_path), "predictions": str(predictions_path),
        "states": len(states), "prediction_rows": len(prediction_rows), "per_task": tasks,
        "seconds": round(time.perf_counter() - started, 1),
        "seconds_per_state": round((time.perf_counter() - started) / max(len(states), 1), 3),
    }, indent=2))


if __name__ == "__main__":
    main()
