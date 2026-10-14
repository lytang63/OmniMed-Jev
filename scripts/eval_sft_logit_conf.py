#!/usr/bin/env python3
"""Read a probability distribution out of the generative baseline's logits.

WHY this exists: asking a fine-tuned generative model to also *say* a confidence
does not work. On 726 test rows the SFT model produced a parseable confidence
payload ZERO times -- appending the request to the user turn and putting it in
the system prompt both failed -- because the fine-tune (terse answer, then EOS)
overrides an inference-time instruction. The untuned base model complied only
sometimes, and the one vector it did emit was all zeros.

So the distribution has to be measured rather than requested:

    logP(c) = sum_t log p(token_t | prompt, image, token_<t)

scored for every candidate, then normalised across the candidate set. This is
the "option-token logits" baseline the MedJev design notes already list.

**Why it does not blow up memory.** Scattering logits for K sequences at once
would materialise (K, L, 262208) -- 37 GB for the 101-level CAMUS score. Instead
this reaches the inner `Gemma3Model` for the hidden states and applies the
vocabulary projection ONLY at the positions the continuation occupies, which is
the same trick MedJev uses to avoid the projection entirely. Peak cost becomes
the hidden states, not the vocabulary.

Attach these two caveats to any number this produces:

* It is a **pseudo-distribution**. Nothing trained these log-likelihoods to sum
  to 1 over our candidate set -- that normalisation is ours, not the model's.
  MedJev's distribution is trained end to end. The gap between the two columns
  is the finding; the SFT column is "what you get if you bolt a distribution
  onto a generative model afterwards".
* Multi-label is reformulated as one binary question per finding ("Is X
  present?", Yes/No), because a single semicolon-list answer has no per-finding
  likelihood to read. That reformulation matches how MedJev asks the same
  question, but it is a different prompt from the one the task metrics use.

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts/eval_sft_logit_conf.py \
        --adapter output/sft-lora/checkpoint-2828 \
        --ground-truth datas/eval/Preprocessed-omnimed-suite/testing.jsonl \
        --output datas/eval/runs-sft/checkpoint-2424/logit_metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from peft import PeftModel
from rich.console import Console
from transformers import AutoProcessor

from omnimedjev.engine.evaluate import normalize_text, parse_multilabel
from omnimedjev.engine.medjev_model import resolve_vision_language_stack
from omnimedjev.engine.train import (choose_attention_backends, choose_dtype, dtype_label,
                              load_model_with_attention_fallback, make_quant_config)
from scripts.eval_sft_metrics import distribution_block, numeric

FAMILIES = ("disease_diagnosis_classification", "multi_label_classification",
            "cell_counting", "regression")
TXL_COUNT_CAP = 32
EF_MAX = 100
RETINA_LEVELS = 5
SYSTEM_PROMPT = "You are an expert medical imaging assistant."


def softmax(values: Sequence[float]) -> list[float]:
    """Turn raw summed log-likelihoods into the distribution we report.

    Necessary because `distribution_block` normalises by the sum, which only
    works on non-negative values; the log-likelihoods are negative sums.
    """
    top = max(values)
    exps = [math.exp(v - top) for v in values]
    total = sum(exps)
    return [e / total for e in exps] if total > 0 else [1.0 / len(values)] * len(values)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def output_head(model):
    """The vocabulary projection, wherever it sits in the PEFT wrapping."""
    for candidate in (model, getattr(model, "base_model", None),
                      getattr(getattr(model, "base_model", None), "model", None)):
        if candidate is None:
            continue
        head = getattr(candidate, "lm_head", None)
        if head is not None:
            return head
        try:
            head = candidate.get_output_embeddings()
        except Exception:
            head = None
        if head is not None:
            return head
    raise TypeError("could not locate the output projection")


def question_for(row: Mapping[str, Any], finding: str | None = None) -> str:
    """The user turn, matching `infer.build_prompt`'s shape for these tasks."""
    from omnimedjev.engine.evaluate import TASK_INSTRUCTIONS
    task = str(row.get("task_type"))
    prompt = str(row.get("prompt") or row.get("question") or "").strip()
    if finding is not None:
        return f"{TASK_INSTRUCTIONS.get(task, '')}\n\nIs {finding} present in this image? Answer Yes or No."
    if "options:" in prompt.lower():
        return f"{TASK_INSTRUCTIONS.get(task, '')}\n\n{prompt}"
    choices = row.get("choices")
    choices = json.loads(choices) if isinstance(choices, str) else (choices or [])
    head = f"{TASK_INSTRUCTIONS.get(task, '')}\n\n{prompt}"
    if choices:
        head += "\n\nOptions: " + "; ".join(str(c) for c in choices)
    return head


@torch.no_grad()
def candidate_logprobs(model, processor, device, image, question: str,
                       candidates: Sequence[str]) -> list[float]:
    """Sum log p(candidate | question, image) for every candidate, one forward."""
    tokenizer = processor.tokenizer
    stack = resolve_vision_language_stack(model)
    head = output_head(model)
    # A word-initial continuation is the natural one after "\n".
    continuations = [" " + c for c in candidates]

    messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    texts = [prompt_text + c for c in continuations]

    batch = processor(text=texts, images=[[image]] * len(texts), return_tensors="pt", padding=True)
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    hidden = stack(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                   pixel_values=batch["pixel_values"], use_cache=False).last_hidden_state

    ids = batch["input_ids"]
    mask = batch["attention_mask"]
    index = []
    lengths = []
    for i, continuation in enumerate(continuations):
        cont_ids = tokenizer(continuation, add_special_tokens=False)["input_ids"]
        real = int(mask[i].sum())
        start = real - len(cont_ids)
        if start - 1 < 0:
            lengths.append(0)
            continue
        lengths.append(len(cont_ids))
        for j in range(len(cont_ids)):
            # logits[t] predicts token t+1
            index.append((i, start + j - 1, ids[i, start + j]))

    if not index:
        return [float("-inf")] * len(candidates)

    rows = torch.tensor([p[0] for p in index], device=device)
    positions = torch.tensor([p[1] for p in index], device=device)
    targets = torch.tensor([p[2] for p in index], device=device)
    selected = hidden[rows, positions]                       # (P, H) -- no vocab yet
    logits = head(selected).float()
    logp = torch.log_softmax(logits, dim=-1)
    picked = logp.gather(1, targets.unsqueeze(1)).squeeze(1)

    totals = [0.0] * len(candidates)
    cursor = 0
    for i, length in enumerate(lengths):
        for _ in range(length):
            totals[i] += float(picked[cursor])
            cursor += 1
    return totals


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/medgemma-1.5-4b-it")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir; omit to score the base model")
    ap.add_argument("--ground-truth", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--image-dir-row", default="image_path")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--precision", default="auto")
    ap.add_argument("--attn", default="auto")
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()

    from PIL import Image, ImageOps

    console = Console()
    families = {f.strip() for f in args.families.split(",") if f.strip()}

    dtype = choose_dtype(args.precision, console)
    quant = make_quant_config(not args.no_4bit, dtype)
    console.print(f"Loading {args.model} ({dtype_label(dtype)}, 4bit={quant is not None})")
    model = load_model_with_attention_fallback(
        args.model, choose_attention_backends(args.attn), console,
        torch_dtype=dtype, device_map=None, quantization_config=quant, trust_remote_code=True)
    if quant is not None:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
        console.print(f"attached adapter {args.adapter}")
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    device = torch.device(args.device)

    rows = [r for r in read_jsonl(args.ground_truth) if str(r.get("task_type")) in families]
    if args.limit:
        rows = rows[: args.limit]
    console.print(f"{len(rows)} rows")

    per_family: dict[str, list[tuple[dict[str, float], str]]] = {}
    started = time.perf_counter()
    for position, row in enumerate(rows, start=1):
        task = str(row["task_type"])
        with Image.open(str(row[args.image_dir_row])) as handle:
            image = ImageOps.exif_transpose(handle).convert("RGB")
            image = image.resize((448, 448), Image.Resampling.BICUBIC)

        if task == "disease_diagnosis_classification":
            choices = row.get("choices")
            choices = json.loads(choices) if isinstance(choices, str) else list(choices)
            gold = normalize_text(row.get("raw_answer", row.get("answer")))
            logs = candidate_logprobs(model, processor, device, image, question_for(row),
                                      [str(c) for c in choices])
            probabilities = dict(zip((normalize_text(c) for c in choices), softmax(logs)))
            per_family.setdefault(task, []).append((probabilities, gold))

        elif task == "multi_label_classification":
            choices = row.get("choices")
            choices = json.loads(choices) if isinstance(choices, str) else list(choices)
            positives = {normalize_text(v) for v in parse_multilabel(row.get("raw_answer"))}
            for finding in choices:
                logs = candidate_logprobs(model, processor, device, image,
                                          question_for(row, str(finding)), ["Yes", "No"])
                p_yes = softmax(logs)[0]           # logs = [logP(Yes), logP(No)]
                key = normalize_text(finding)
                per_family.setdefault(task, []).append(
                    ({key: p_yes, "__absent__": 1.0 - p_yes},
                     key if key in positives else "__absent__"))

        elif task == "cell_counting":
            gold = round(numeric(row.get("raw_answer")) or 0)
            logs = candidate_logprobs(model, processor, device, image, question_for(row),
                                      [str(k) for k in range(TXL_COUNT_CAP + 1)])
            probabilities = dict(zip((str(k) for k in range(TXL_COUNT_CAP + 1)), softmax(logs)))
            per_family.setdefault(task, []).append((probabilities, str(gold)))

        elif task == "regression":
            levels = EF_MAX + 1 if str(row.get("source")) == "CAMUS" else RETINA_LEVELS
            gold = round(numeric(row.get("raw_answer")) or 0)
            logs = candidate_logprobs(model, processor, device, image, question_for(row),
                                      [str(k) for k in range(levels)])
            probabilities = dict(zip((str(k) for k in range(levels)), softmax(logs)))
            per_family.setdefault(task, []).append((probabilities, str(gold)))
        else:
            continue

        if position % 20 == 0 or position == len(rows):
            console.print(f"  {position}/{len(rows)}  {time.perf_counter() - started:.0f}s")

    report: dict[str, Any] = {
        "adapter": args.adapter, "rows": len(rows),
        "seconds": round(time.perf_counter() - started, 1),
        "note": "pseudo-distribution from option log-likelihoods, normalised by us",
        "families": {},
    }
    for task, entries in per_family.items():
        block = distribution_block(entries)
        block["unit"] = ("per_finding" if task == "multi_label_classification" else "per_row")
        block["source"] = "option log-likelihood (batched forward)"
        report["families"][task] = block

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
