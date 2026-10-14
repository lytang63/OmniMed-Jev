"""Model wrapper and collator shared by MedJev training and inference.

Two things live here, both of which have to be identical between training and
inference or the checkpoint will not mean what the evaluation thinks it means:

* `MedJevModel` -- MedGemma backbone plus the shared decision head, with the
  backbone's own vocabulary projection deliberately bypassed.
* `collate_states` -- turns a list of compiled decision states into the padded
  tensor batch the head consumes.

**Why `backbone.model(...)` and not `backbone(...)`:** Gemma3ForConditionalGeneration
projects every position through a 262,208-way vocabulary head. A 3000-token
sequence would allocate ~1.6 GB of logits per sample before the backward pass
even starts. MedJev never needs a token distribution, so it calls the inner
`Gemma3Model`, which returns `last_hidden_state` and stops there.

Load model weights with the helpers in `omnimedjev/engine/train.py` rather than
re-deriving precision, attention-backend fallback and 4-bit configuration;
those are the settings the generative baseline ran with and the comparison
depends on them matching.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from omnimedjev.engine.medjev import MedJevHead, encode_state
from omnimedjev.engine.train import get_image_paths, load_image


def resolve_vision_language_stack(model: nn.Module) -> nn.Module:
    """Find the inner `Gemma3Model` inside whatever PeftModel wrapping is present.

    The wrapping depth is not fixed: bare, it is `model.model`; with LoRA
    attached it becomes `model.base_model.base_model` while `model.model`
    still resolves to the *outer* Gemma3ForConditionalGeneration. Rather than
    hard-code a chain that shifts with the peft version, walk the `model` /
    `base_model` links and identify the target by what it holds.
    """
    seen: set[int] = set()
    frontier = [model]
    while frontier:
        candidate = frontier.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        if hasattr(candidate, "language_model") and hasattr(candidate, "vision_tower"):
            return candidate
        for attribute in ("base_model", "model"):
            child = getattr(candidate, attribute, None)
            if isinstance(child, nn.Module):
                frontier.append(child)
    raise TypeError("could not locate the Gemma3Model inside the loaded model")


class MedJevModel(nn.Module):
    """Backbone + shared decision head. Loss is computed only when targets are given."""

    def __init__(self, backbone: nn.Module, hidden_size: int, *, use_query: bool = True,
                 use_set_head: bool = True, freeze_backbone: bool = False):
        super().__init__()
        self.backbone = backbone
        self.head = MedJevHead(hidden_size, use_query=use_query, use_set_head=use_set_head)
        self.freeze_backbone = freeze_backbone
        # Trainer reaches for `.config` in several places (dtype checks, logging).
        self.config = getattr(backbone, "config", None)

    def gradient_checkpointing_enable(self, *args, **kwargs):
        """Delegate to the backbone.

        `Trainer` calls this on the model it is given whenever
        `gradient_checkpointing=True`; without the delegation it would fail on
        this wrapper, which is a plain nn.Module rather than a PreTrainedModel.
        """
        return self.backbone.gradient_checkpointing_enable(*args, **kwargs)

    def gradient_checkpointing_disable(self, *args, **kwargs):
        return self.backbone.gradient_checkpointing_disable(*args, **kwargs)

    def embed(self, batch: Mapping[str, Any]) -> torch.Tensor:
        """One backbone pass; returns the fused last hidden state.

        The image soft tokens are already expanded into `input_ids` by the
        processor, so this is a single forward over the whole state.

        When the backbone is frozen (head warm-up) the pass runs without
        building a graph -- there is nothing upstream to differentiate into,
        and skipping it removes the bulk of the activation memory.

        The ambient grad mode is respected: `torch.enable_grad()` here would
        silently override a caller's `torch.no_grad()` and make inference build
        a full autograd graph, which is both far slower and much larger.
        """
        stack = resolve_vision_language_stack(self.backbone)
        want_graph = torch.is_grad_enabled() and not self.freeze_backbone
        with torch.enable_grad() if want_graph else torch.no_grad():
            outputs = stack(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                use_cache=False,
            )
            return outputs.last_hidden_state

    def forward(self, **batch: Any) -> dict[str, torch.Tensor]:
        hidden = self.embed(batch)
        logits = self.head(
            hidden,
            batch["key_sample"], batch["key_pos"],
            batch["query_sample"], batch["query_pos"],
            batch["valid"],
        )
        result = {"logits": logits}
        if "target" in batch:
            from omnimedjev.engine.medjev import masked_ce
            result["loss"] = masked_ce(logits, batch["target"], batch["valid"])
        return result


def build_batch(encoded: Sequence[Mapping[str, Any]], pad_token_id: int,
                device: torch.device | None = None, *, with_meta: bool = True) -> dict[str, Any]:
    """Pad a list of `encode_state` outputs into one batch.

    Questions are flattened across the batch: row `q` of the returned tensors
    is a question, and `query_sample[q]` says which sample it belongs to. That
    keeps the head's work proportional to the number of questions rather than
    to `batch x max_questions`.

    `with_meta=False` drops the per-state metadata, which cannot be moved to a
    device and would otherwise be passed straight into `model(**batch)`.
    """
    batch_size = len(encoded)
    width = max(item["input_ids"].shape[0] for item in encoded)
    max_candidates = max(max(item["cand_count"]) for item in encoded)

    input_rows = torch.full((batch_size, width), pad_token_id, dtype=torch.long)
    attention_rows = torch.zeros((batch_size, width), dtype=torch.long)
    pixel_values = []
    for index, item in enumerate(encoded):
        length = item["input_ids"].shape[0]
        input_rows[index, :length] = item["input_ids"]
        attention_rows[index, :length] = 1
        pixel_values.append(item["pixel_values"])

    key_sample, key_pos, query_sample, query_pos, valid, target = [], [], [], [], [], []
    for index, item in enumerate(encoded):
        for question, keys in enumerate(item["keys_by_question"]):
            query_sample.append(index)
            query_pos.append(item["query_pos"][question])
            padded = torch.full((max_candidates,), -1, dtype=torch.long)
            padded[:len(keys)] = torch.tensor(keys, dtype=torch.long)
            key_pos.append(padded)
            key_sample.append(index)  # one batch index per QUESTION, not per candidate slot
            mask = torch.zeros(max_candidates, dtype=torch.bool)
            mask[:len(keys)] = True
            valid.append(mask)
            if "target" in item:
                decision = item["target"][question]
                ids = [c["id"] for c in item["candidates"][question]]
                row = torch.zeros(max_candidates, dtype=torch.float32)
                for slot, candidate_id in enumerate(ids):
                    row[slot] = float(decision.get(candidate_id, 0.0))
                target.append(row)

    batch: dict[str, Any] = {
        "input_ids": input_rows,
        "attention_mask": attention_rows,
        "pixel_values": torch.stack(pixel_values),
        "key_sample": torch.tensor(key_sample, dtype=torch.long),
        "key_pos": torch.stack(key_pos),
        "query_sample": torch.tensor(query_sample, dtype=torch.long),
        "query_pos": torch.tensor(query_pos, dtype=torch.long),
        "valid": torch.stack(valid),
    }
    if target:
        batch["target"] = torch.stack(target)
    if device is not None:
        batch = {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items()}
    if with_meta:
        batch["meta"] = list(encoded)
    return batch


class MedJevCollator:
    """`encode_state` plus padding, wired for a dataloader.

    Keeps the decision targets so the loss can be computed; the inference path
    builds the same batch without them (`with_targets=False`) so training and
    inference share one encoder.

    `shuffle_candidates` implements candidate-order augmentation: every time a
    state is drawn, the options are reordered. It is the fix for a measured
    failure -- a model trained on one fixed order is not permutation-invariant
    (see `scripts/probe_medjev.py`), because it can lean on position instead of
    reading the option names.

    Only `choice` and `noul` decisions are reordered. A `score` decision's
    candidates are ordered levels, and that order is what defines each level's
    index -- shuffling it would scramble an ordinal scale rather than test
    anything. The level text is self-contained, so order-invariance there is
    arguably meaningful too, but it would be a different (and riskier)
    experiment, not this one.
    """

    def __init__(self, processor, *, image_size: int = 896, resize_mode: str = "square",
                 max_images_per_sample: int = 1, max_length: int = 4096, with_targets: bool = True,
                 shuffle_candidates: bool = False, seed: int = 0):
        self.processor = processor
        self.image_size = image_size
        self.resize_mode = resize_mode
        self.max_images_per_sample = max_images_per_sample
        self.max_length = max_length
        self.with_targets = with_targets
        self.shuffle_candidates = shuffle_candidates
        # Advanced per draw, so the same state is seen in a different order each
        # epoch. Seeding per (state, question) instead would fix one order for
        # the whole run, which is the thing being fixed.
        self._rng = random.Random(seed)
        self.pad_token_id = processor.tokenizer.pad_token_id
        if self.pad_token_id is None:
            raise ValueError("processor tokenizer has no pad_token_id")

    def _prepare(self, state: Mapping[str, Any]) -> list[dict[str, Any]]:
        """The decisions this draw will use, possibly with options reordered.

        The text the model reads and the metadata used to build the target MUST
        come from the same list: `encode_state` lays the markers out in this
        order and `build_batch` maps each target row by looking the candidate id
        up in it. Reordering one and not the other would silently mis-label the
        batch, so the list is built once here and threaded through both.
        """
        decisions = state["decisions"]
        if not self.shuffle_candidates:
            return list(decisions)
        prepared = []
        for decision in decisions:
            if str(decision["type"]) not in ("choice", "noul"):
                prepared.append(decision)
                continue
            candidates = list(decision["candidates"])
            self._rng.shuffle(candidates)
            prepared.append({**decision, "candidates": candidates})
        return prepared

    def encode(self, state: Mapping[str, Any]) -> dict[str, Any]:
        paths = get_image_paths(state)[: self.max_images_per_sample]
        images = [load_image(path, self.image_size, self.resize_mode) for path in paths]
        if not images:
            raise ValueError(f"{state.get('state_id')}: state has no images")
        decisions = self._prepare(state)
        # `encode_state` reads `state["decisions"]`, so hand it the same list
        # that the target rows below are built from.
        encoded = encode_state({**state, "decisions": decisions}, self.processor, images, self.max_length)
        encoded["state_id"] = state["state_id"]
        encoded["legacy_uids"] = state.get("legacy_uids", {})
        encoded["image_size"] = state.get("image_size")
        # Question metadata is always carried: inference needs it to interpret
        # the distributions, and training needs it for the target rows.
        encoded["candidates"] = [list(d["candidates"]) for d in decisions]
        encoded["qid"] = [str(d["qid"]) for d in decisions]
        encoded["type"] = [str(d["type"]) for d in decisions]
        encoded["instructions"] = [str(d["instructions"]) for d in decisions]
        if self.with_targets:
            encoded["target"] = [dict(d["target"]) for d in decisions]
            encoded["target_kind"] = [str(d["target_kind"]) for d in decisions]
        return encoded

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        encoded = [self.encode(state) for state in features]
        return build_batch(encoded, self.pad_token_id)
