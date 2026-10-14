#!/usr/bin/env python3
"""Train MedJev: a Jev-style typed-decision interface on top of MedGemma.

Two stages, matching the two-stage schedule the MedJev notes call for and the
shape of NanoJev's own warm-up:

* `--stage head` -- backbone frozen, only the decision head trains. Cheap, and
  it gives the randomly initialised head a chance to reach a sane scale before
  it starts pushing gradients into LoRA.
* `--stage lora` -- LoRA adapters plus the head. Seeded from the Stage A head.

The Stage B settings mirror `configs/train-sft.yaml` field for field, so the run
is comparable to the generative baseline it is meant to be compared against:
same base model, same 4-bit quantisation, same LoRA rank/alpha/dropout/target
modules, same learning rate, same effective batch, same step budget. The ONLY
intentional difference is `max_length`, which this arm must raise because a
state packs every decision about one image into one sequence. That difference is
recorded in the emitted config and must be disclosed alongside the numbers.

Usage:

    python scripts/train_medjev.py --stage head \\
        --train datas/medjev/train.jsonl --eval datas/medjev/validation.jsonl \\
        --output-dir runs/medjev_head

    python scripts/train_medjev.py --stage lora --init-head runs/medjev_head \\
        --train datas/medjev/train.jsonl --eval datas/medjev/validation.jsonl \\
        --output-dir runs/medjev_lora
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from rich.console import Console
from safetensors.torch import load_file, save_file
from transformers import AutoProcessor, Trainer, TrainingArguments

from omnimedjev.engine.medjev import assert_marker_ids
from omnimedjev.engine.medjev_model import MedJevCollator, MedJevModel
from omnimedjev.engine.train import (choose_attention_backends, choose_dtype, dtype_label,
                              load_model_with_attention_fallback, make_quant_config,
                              resolve_warmup_steps, seed_everything)

HEAD_FILE = "medjev_head.safetensors"


def approximate_length(state: dict[str, Any]) -> int:
    """Character-count proxy for the encoded length, for the length sampler.

    Tokenising all ~30k states up front would cost more than it saves; the
    baseline's own `length` column is a `len(prompt) + len(answer)` proxy for
    the same reason. The image contributes a constant 256 tokens to every
    state, so it cannot affect the ordering.
    """
    total = 0
    for decision in state["decisions"]:
        total += len(decision["instructions"])
        total += sum(len(str(c["text"])) + 4 for c in decision["candidates"])
        total += 6
    return total


def load_states(path: Path) -> list[dict[str, Any]]:
    states = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    for state in states:
        state["length"] = approximate_length(state)
    return states


class MedJevTrainer(Trainer):
    """Trainer that treats the wrapper's dict output as `(loss, outputs)`."""

    def __init__(self, *args, eval_collator=None, **kwargs):
        super().__init__(*args, **kwargs)
        # The evaluation pass must use a collator that does NOT reorder options.
        # Otherwise the dev loss would vary with the augmentation draw and stop
        # being comparable between steps.
        self.eval_collator = eval_collator
        # `label_names=[]` makes `has_labels` False, so `prediction_step` only
        # computes a loss when `loss_without_labels` is true -- which requires
        # `can_return_loss`. transformers derives that by looking for a
        # `return_loss` parameter on the model's forward, and `MedJevModel.forward`
        # takes only `**batch`, so it came out False and every evaluation step
        # ran, measured nothing, and logged no `eval_loss`. Setting it here makes
        # the dev curve exist.
        self.can_return_loss = True

    def get_eval_dataloader(self, eval_dataset=None):
        if self.eval_collator is None:
            return super().get_eval_dataloader(eval_dataset)
        # The loader captures the collator when it is built, so swapping it in
        # for the duration of this call is enough.
        original = self.data_collator
        self.data_collator = self.eval_collator
        try:
            return super().get_eval_dataloader(eval_dataset)
        finally:
            self.data_collator = original

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs = {k: v for k, v in inputs.items() if k != "meta"}
        outputs = model(**inputs)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """Restore this run's own checkpoint layout.

        transformers' loader looks for `pytorch_model.bin` / `model.safetensors`,
        or -- for PEFT -- recognises the model by class and calls `load_adapter`.
        Neither applies here: the checkpoint deliberately carries no backbone
        weights (2.5B unchanged parameters, reloaded from the base model), and
        the model is a `MedJevModel` wrapper rather than a `PeftModel`, so
        `_is_peft_model` is False and the stock loader falls through to the
        sharded path and raises `Can't find a checkpoint index`.

        `build_model` has already restored the adapters and the head from this
        same directory, so this only has to avoid clobbering them. Resuming is
        still worth it: the optimizer, scheduler, RNG and global step are
        restored by the training loop independently of this method.
        """
        model = model or self.model
        head = Path(resume_from_checkpoint) / HEAD_FILE
        if head.exists():
            model.head.load_state_dict(load_file(str(head)))
        else:
            raise FileNotFoundError(f"resume checkpoint has no {HEAD_FILE}: {head}")

    def _save(self, output_dir: str | None = None, state_dict: dict | None = None) -> None:
        """Write the LoRA adapters and the head -- never the backbone.

        Deliberately does not call `super()._save`: transformers would dump the
        full model, including the tied `embed_tokens`/`lm_head` pair, which
        safetensors refuses as duplicated storage. Neither copy is wanted here
        anyway -- the base weights are reloaded from the base model, and only
        the adapters and the head differ between runs.
        """
        output_dir = output_dir or self.args.output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        if not self.model.freeze_backbone:
            self.model.backbone.save_pretrained(str(output_dir))
        # The head is a plain nn.Module rather than part of the PEFT adapter,
        # so it has to be written out separately or the checkpoint is unusable.
        head = {k: v.detach().cpu().contiguous().clone()
                for k, v in self.model.head.state_dict().items()}
        save_file(head, str(Path(output_dir) / HEAD_FILE))


def build_model(args, console: Console):
    dtype = choose_dtype(args.precision, console)
    quant = make_quant_config(not args.no_4bit, dtype)
    console.print(f"Loading {args.model} ({dtype_label(dtype)}, 4bit={quant is not None})")
    backbone = load_model_with_attention_fallback(
        args.model, choose_attention_backends(args.attn), console,
        torch_dtype=dtype, device_map=None, quantization_config=quant, trust_remote_code=True)
    hidden_size = backbone.config.text_config.hidden_size
    head_only = args.stage == "head"

    if quant is not None:
        # `prepare_model_for_kbit_training` enables gradient checkpointing by
        # default. Stage A freezes the backbone, so checkpointing there would
        # only re-run frozen layers to produce gradients nobody consumes --
        # measurably slower for no memory benefit.
        backbone = prepare_model_for_kbit_training(
            backbone, use_gradient_checkpointing=not head_only)
    backbone.config.use_cache = False

    if head_only:
        # Stage A: nothing but the head moves, so the backbone runs without a
        # graph and gradient checkpointing is pointless.
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        model = MedJevModel(backbone, hidden_size, freeze_backbone=True)
    else:
        resume = args.resume_from_checkpoint
        if resume:
            # Resume must attach the CHECKPOINT's adapters, not fresh ones.
            # `Trainer.train(resume_from_checkpoint=...)` restores the optimizer,
            # scheduler, step count and RNG, but it will not restore the weights
            # here: it looks for `model.safetensors` and this run writes the
            # PEFT-named files, so the model would silently continue with
            # step-zero adapters against a step-466 optimizer.
            backbone = PeftModel.from_pretrained(backbone, str(resume), is_trainable=True)
            console.print(f"Resumed LoRA adapters from {resume}")
        else:
            backbone = get_peft_model(backbone, LoraConfig(
                r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                bias="none", target_modules=args.target_modules, task_type="CAUSAL_LM"))
        if args.gradient_checkpointing:
            backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        model = MedJevModel(backbone, hidden_size, freeze_backbone=False)

    # The head always needs an explicit source: on resume it must come from the
    # checkpoint, otherwise it restarts from the Stage A warm-up.
    head_source = args.resume_from_checkpoint or args.init_head
    if head_source:
        source = Path(head_source)
        weights = source / HEAD_FILE if source.is_dir() else source
        if not weights.exists():
            raise FileNotFoundError(f"no {HEAD_FILE} at {weights}")
        model.head.load_state_dict(load_file(str(weights)))
        console.print(f"Initialised head from {weights}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    console.print(f"trainable {trainable:,} / {total:,} ({trainable / total:.3%})")
    return model, dtype


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, type=Path)
    ap.add_argument("--eval", type=Path, default=None)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--stage", choices=["head", "lora"], required=True)
    ap.add_argument("--init-head", default=None)
    ap.add_argument("--resume-from-checkpoint", default=None,
                    help="a checkpoint-N directory from this run; restores adapters, head, "
                         "optimizer, scheduler, RNG and the global step")
    ap.add_argument("--model", default="models/medgemma-1.5-4b-it")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--num-epochs", type=float, default=2.0)
    ap.add_argument("--head-steps", type=int, default=200,
                    help="Stage A length. The warm-up only has to reach a sane head scale "
                         "before LoRA starts, so it is bounded by steps, not epochs.")
    ap.add_argument("--per-device-train-batch-size", type=int, default=8)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--max-length", type=int, default=3072)
    ap.add_argument("--image-size", type=int, default=448)
    ap.add_argument("--resize-mode", default="square")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--target-modules", default="all-linear")
    ap.add_argument("--precision", default="auto")
    ap.add_argument("--attn", default="auto")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                    action="store_false")
    ap.add_argument("--save-steps", type=int, default=233)
    ap.add_argument("--eval-steps", type=int, default=233)
    ap.add_argument("--logging-steps", type=int, default=25)
    # 0 = keep every checkpoint. The generative baseline's sweep (ft233/466/699/932)
    # is what shows how the interface evolves with training, so pruning early
    # checkpoints would throw away half the experiment. Each one is ~230 MB.
    ap.add_argument("--save-total-limit", type=int, default=0)
    ap.add_argument("--max-eval-samples", type=int, default=128)
    ap.add_argument("--dataloader-num-workers", type=int, default=8)
    # Candidate-order augmentation. Measured failure it fixes: a model trained
    # on one fixed option order is not permutation-invariant (top-1 changed on
    # 31-95% of decisions when the options were shuffled), because it can lean
    # on position rather than reading the option names.
    ap.add_argument("--shuffle-candidates", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="tiny run for wiring checks")
    args = ap.parse_args()

    console = Console()
    if args.smoke:
        args.max_steps = 4
        args.save_steps = 2
        args.eval_steps = 2
        args.logging_steps = 1
        args.max_eval_samples = 8
        args.dataloader_num_workers = 0
        args.head_steps = 4
        console.print("Smoke mode: 4 steps, 8 eval states.")

    if args.max_steps < 0 and args.stage == "head":
        # Stage A is a warm-up, not half the training run: it only has to get
        # the randomly initialised head to a sane scale before LoRA begins.
        args.max_steps = args.head_steps
        console.print(f"Stage A: {args.max_steps} head-only steps.")

    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from datasets import Dataset
    train_states = load_states(args.train)
    train_dataset = Dataset.from_list(train_states)
    eval_dataset = None
    if args.eval is not None and args.eval.exists():
        eval_states = load_states(args.eval)[: args.max_eval_samples]
        eval_dataset = Dataset.from_list(eval_states)
    console.print(f"train states {len(train_dataset)}  eval states "
                  f"{len(eval_dataset) if eval_dataset else 0}")

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    assert_marker_ids(processor.tokenizer)

    model, dtype = build_model(args, console)
    collator = MedJevCollator(processor, image_size=args.image_size, resize_mode=args.resize_mode,
                              max_length=args.max_length, with_targets=True,
                              shuffle_candidates=args.shuffle_candidates, seed=args.seed)
    eval_collator = MedJevCollator(processor, image_size=args.image_size, resize_mode=args.resize_mode,
                                   max_length=args.max_length, with_targets=True,
                                   shuffle_candidates=False)
    if args.shuffle_candidates:
        console.print("Candidate-order augmentation ON for training (eval stays in the data's own order).")

    # transformers 5.x dropped `warmup_ratio`; `resolve_warmup_steps` is the
    # helper the SFT path already uses to turn the ratio into a step count, so
    # reusing it keeps MedJev's schedule identical to the baseline's.
    warmup_steps = resolve_warmup_steps(
        None, args.warmup_ratio, len(train_dataset),
        args.per_device_train_batch_size, args.gradient_accumulation_steps,
        args.num_epochs, args.max_steps,
    )
    console.print(f"warmup_steps={warmup_steps}")

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="linear",
        warmup_steps=warmup_steps,
        weight_decay=0.0,
        max_grad_norm=0.3,
        optim="paged_adamw_8bit" if not args.no_4bit else "adamw_torch_fused",
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        gradient_checkpointing=args.gradient_checkpointing and args.stage == "lora",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        train_sampling_strategy="group_by_length",
        length_column_name="length",
        # On resume, don't replay-and-discard the batches already consumed: with
        # a length-grouped sampler the skip would not land on the same data
        # anyway, and it makes a restart slow in proportion to how far in it was.
        ignore_data_skip=True,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        label_names=[],
        report_to=[],
        seed=args.seed,
    )

    trainer = MedJevTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        eval_collator=eval_collator,
    )

    started = time.perf_counter()
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))

    (args.output_dir / "medjev_config.json").write_text(json.dumps({
        **vars(args),
        "train": str(args.train), "eval": str(args.eval), "output_dir": str(args.output_dir),
        "resolved_dtype": dtype_label(dtype),
        "train_states": len(train_dataset),
        "eval_states": len(eval_dataset) if eval_dataset else 0,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "training_seconds": round(time.perf_counter() - started, 1),
        "max_gpu_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }, indent=2, default=str), encoding="utf-8")
    console.print(f"done -> {args.output_dir}")


if __name__ == "__main__":
    main()
