#!/usr/bin/env python3
"""Measure MedJev's peak memory and throughput before committing to a schedule.

WHY this is a separate step: a MedJev state carries every decision about one
image in a single sequence, so the longest states are several times the median.
The generative baseline trained at `max_length: 1024`; if MedJev needs more,
that is a real difference between the two runs and it has to be measured
rather than assumed. This script reports the peak for the worst-case batch so
`max_length` and the batch size can be chosen from evidence.

Usage:
    python scripts/benchmark_medjev.py --gpu 2 --data datas/medjev/testing.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from rich.console import Console
from transformers import AutoProcessor

from omnimedjev.engine.medjev import assert_marker_ids
from omnimedjev.engine.medjev_model import MedJevCollator, MedJevModel, resolve_vision_language_stack
from omnimedjev.engine.train import (choose_attention_backends, choose_dtype, dtype_label,
                              load_model_with_attention_fallback, make_quant_config)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/medgemma-1.5-4b-it")
    ap.add_argument("--data", default="datas/medjev/testing.jsonl")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=3072)
    ap.add_argument("--image-size", type=int, default=448)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--target-modules", default="all-linear")
    ap.add_argument("--precision", default="auto")
    ap.add_argument("--attn", default="auto")
    ap.add_argument("--steps", type=int, default=3)
    args = ap.parse_args()

    console = Console()
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats()

    states = [json.loads(line) for line in Path(args.data).open() if line.strip()]
    states.sort(key=lambda s: -len(json.dumps(s["decisions"])))
    console.print(f"Loaded {len(states)} states from {args.data}")

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    assert_marker_ids(processor.tokenizer)

    dtype = choose_dtype(args.precision, console)
    quant = make_quant_config(True, dtype)
    console.print(f"Loading {args.model} ({dtype_label(dtype)}, 4bit={quant is not None})")
    backbone = load_model_with_attention_fallback(
        args.model, choose_attention_backends(args.attn), console,
        torch_dtype=dtype, device_map=None, quantization_config=quant, trust_remote_code=True)
    if quant is not None:
        backbone = prepare_model_for_kbit_training(backbone)
    backbone.config.use_cache = False
    backbone = get_peft_model(backbone, LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", target_modules=args.target_modules, task_type="CAUSAL_LM"))
    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    hidden_size = backbone.config.text_config.hidden_size
    model = MedJevModel(backbone, hidden_size).to(device)
    stack = resolve_vision_language_stack(backbone)
    console.print(f"resolved vision-language stack: {type(stack).__name__}")

    # Only the head plus LoRA adapters should be trainable.
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    console.print(f"trainable {trainable:,} / {total:,} params ({trainable / total:.3%})")

    collator = MedJevCollator(processor, image_size=args.image_size,
                              max_length=args.max_length, with_targets=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    model.train()

    # Worst case first: the longest states, padded to their own maximum.
    lengths, times = [], []
    for step in range(args.steps):
        start = args.batch_size * step
        group = states[start:start + args.batch_size]
        if len(group) < args.batch_size:
            break
        batch = collator(group)
        width = int(batch["input_ids"].shape[1])
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items() if k != "meta"}
        torch.cuda.synchronize(); began = time.perf_counter()
        with torch.autocast("cuda", dtype=dtype):
            out = model(**batch)
            loss = out["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - began
        lengths.append(width); times.append(elapsed)
        peak = torch.cuda.max_memory_allocated() / 1e9
        console.print(f"step {step}: width={width} loss={float(loss):.4f} "
                      f"{elapsed:.2f}s peak={peak:.1f}GB")

    if times:
        peak = torch.cuda.max_memory_allocated() / 1e9
        console.print(json.dumps({
            "worst_width": max(lengths),
            "median_step_seconds": round(statistics.median(times), 2),
            "peak_allocated_gb": round(peak, 2),
            "total_gb": round(torch.cuda.get_device_properties(device).total_memory / 1e9, 1),
        }, indent=2))


if __name__ == "__main__":
    main()
