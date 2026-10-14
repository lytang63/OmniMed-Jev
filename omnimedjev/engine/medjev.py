"""MedJev: Jev-style typed probabilistic decisions on top of MedGemma.

This module is the single shared core behind every MedJev script. It holds four
things and nothing else:

1. the *encoding* that lays a whole medical state -- one image plus every
   decision question about it -- into a single token sequence;
2. the *decision head* that turns marker hidden states into per-question
   distributions;
3. the *unified loss* (one cross entropy for Choice, Noul and Score alike);
4. the *folding rules* that turn distributions back into the legacy prediction
   strings `omnimedjev evaluate` already knows how to score.

Deliberately NOT here: model loading, training loops, HTTP serving. Those live
in the scripts.

Why markers instead of one forward per candidate (the NanoJev way): MedGemma
expands every image into 256 soft tokens, so repeating the prefix once per
candidate would run the vision tower K times per question. Laying the whole
state out once costs a single vision forward and is what makes "14 findings in
one pass" true rather than aspirational.
"""

from __future__ import annotations

import json
import math
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import nn

# Single-token, context-stable markers. Verified against the local
# GemmaTokenizer (vocab 262144): both keep the same id whether or not they are
# preceded by whitespace. This matters because the tokenizer merges some
# symbols into different tokens depending on context -- e.g. "▶" is 241175
# alone but 160450 after a space -- which would silently shift every position.
# `assert_marker_ids` re-checks at runtime so a tokenizer swap fails loudly.
MARKER_KEY = "⟩"    # ⟩  candidate end; its hidden state is the key
MARKER_QUERY = "✦"  # ✦  question end;  its hidden state is the query
MARKER_KEY_ID = 241287
MARKER_QUERY_ID = 244193

ALL_TYPES = ("choice", "noul", "score")


# --------------------------------------------------------------------------
# The decision space
# --------------------------------------------------------------------------
# These constants and constructors ARE the interface contract. They live here,
# next to the encoder, rather than in the data compiler, because inference has
# to rebuild part of a state on the fly: the detection workflow asks presence
# first, then constructs the count and coordinate decisions from what came
# back. If the two ever disagreed the model would be evaluated on questions it
# was never trained to answer, so there is exactly one definition of each.
#
# Every value is fixed across train/dev/test -- a decision's candidate set must
# never depend on which split it came from.

NO_YES = [{"id": "no", "text": "No"}, {"id": "yes", "text": "Yes"}]
COUNT_CAP = 32           # TXL-PBC cell totals run 4..32
DETECT_COUNT_CAP = 8     # ChestX-Det boxes per class: 99th percentile 5
COORD_BINS = 12          # bins per image axis; the expectation gives sub-bin resolution
MAX_LOCALIZED_BOXES = 3  # largest-first, per STATE
EF_MAX = 100             # CAMUS ejection fraction is a percentage
RETINA_LEVELS = 5        # ordinal grades 0..4

# Detection candidate sets. The converters keep these only inside the prompt
# text; `convert_datasets.py` holds the authoritative copies.
DETECTION_CLASSES: dict[str, list[str]] = {
    "ChestX-Det": ["Atelectasis", "Calcification", "Cardiomegaly", "Consolidation", "Diffuse Nodule",
                   "Effusion", "Emphysema", "Fibrosis", "Fracture", "Mass", "Nodule",
                   "Pleural Thickening", "Pneumothorax"],
    "TXL-PBC": ["WBC", "RBC", "Platelets"],
}
TXL_CELL_TYPES = ["RBC", "WBC", "Platelets"]


def one_hot_target(candidates: Sequence[Mapping[str, str]], chosen_id: str) -> dict[str, float]:
    ids = [str(c["id"]) for c in candidates]
    if chosen_id not in ids:
        raise ValueError(f"target id {chosen_id!r} is not among candidates {ids}")
    return {cid: (1.0 if cid == chosen_id else 0.0) for cid in ids}


def _decision(qid: str, typ: str, instructions: str, candidates: list[dict[str, str]],
              legacy_uid: str | None, target: dict[str, float], target_kind: str) -> dict[str, Any]:
    return {"qid": qid, "type": typ, "instructions": instructions, "candidates": candidates,
            "target": target, "target_kind": target_kind, "legacy_uid": legacy_uid}


def presence_decision(class_name: str, modality: str, present: bool,
                      legacy_uid: str | None = None) -> dict[str, Any]:
    """A Noul is a 2-candidate Choice over {No, Yes}.

    Boolean, Choice and Score therefore share one head, one softmax and one
    cross entropy -- there is no separate sigmoid path to keep calibrated.
    """
    return _decision(
        f"present:{class_name}", "noul",
        f"Is {class_name} present in this {modality} image?",
        [dict(c) for c in NO_YES], legacy_uid,
        one_hot_target(NO_YES, "yes" if present else "no"), "one_hot")


def count_decision(class_name: str, count: int | None = None,
                   legacy_uid: str | None = None) -> dict[str, Any]:
    candidates = [{"id": str(k), "text": str(k)} for k in range(DETECT_COUNT_CAP + 1)]
    target = one_hot_target(candidates, str(count)) if count is not None else {}
    return _decision(f"count:{class_name}", "choice",
                     f"How many {class_name} instances are present?",
                     candidates, legacy_uid, target, "one_hot")


def coordinate_candidates() -> list[dict[str, str]]:
    """Bin centres as percentages.

    No ordinal index is injected: the level's own text carries its position, so
    moving a level changes which slot it occupies without changing how it reads.
    """
    centers = [(k + 0.5) / COORD_BINS for k in range(COORD_BINS)]
    return [{"id": f"c{k}", "text": f"{c * 100:.0f}%"} for k, c in enumerate(centers)]


def coordinate_decisions(class_name: str, prefix: str, legacy_uid: str | None = None,
                         boxes: Sequence[Sequence[float]] | None = None,
                         size: tuple[int, int] | None = None) -> list[dict[str, Any]]:
    """`MAX_LOCALIZED_BOXES` box slots, four edge Scores each.

    With `boxes` given (compilation) the targets are filled in; without it
    (inference) the same decisions are emitted targetless so the model is
    asked exactly the questions it was trained on.
    """
    candidates = coordinate_candidates()
    decisions: list[dict[str, Any]] = []
    for index in range(MAX_LOCALIZED_BOXES):
        for edge, direction in (("left", "horizontal"), ("top", "vertical"),
                                ("right", "horizontal"), ("bottom", "vertical")):
            target: dict[str, float] = {}
            kind = "discretization_interpolation"
            if boxes is not None and size is not None and index < len(boxes):
                x1, y1, x2, y2 = (float(v) for v in boxes[index][:4])
                value, axis_size = {
                    "left": (x1, size[0]), "top": (y1, size[1]),
                    "right": (x2, size[0]), "bottom": (y2, size[1]),
                }[edge]
                target = _interpolate_coordinate(value / axis_size)
                kind = "discretization_interpolation"
            decisions.append(_decision(
                f"{prefix}:box{index}:{edge}", "score",
                f"Where is the {edge} edge of {class_name} instance {index + 1} "
                f"along the {direction} axis?",
                [dict(c) for c in candidates], legacy_uid, target, kind))
    return decisions


def localization_budget(present: Sequence[tuple[str, float]],
                        max_slots: int = MAX_LOCALIZED_BOXES) -> dict[str, int]:
    """Split the per-state box budget across the findings reported present.

    Training keeps at most `MAX_LOCALIZED_BOXES` boxes for a whole state,
    ranked largest-first, so the coordinate questions a model ever saw were
    bounded by that. Inference must respect the same budget: asking about
    three boxes for each of thirteen findings would produce a state several
    times longer than anything trained on, and would be answering a different
    question than the one the checkpoint was fit to.

    Findings are taken in descending confidence, so when the budget cannot
    cover everything it goes where the model is most sure. Ordering by
    probability keeps this a code decision; the model's only job is to report
    which findings are present.
    """
    ranked = [name for name, probability in
              sorted(present, key=lambda item: item[1], reverse=True)]
    if not ranked:
        return {}
    # `per_class` must never guarantee every finding a slot: with thirteen
    # findings present and a budget of three, handing each one a slot would
    # allocate thirteen and blow past both the budget and the trained length.
    per_class = max(1, max_slots // len(ranked))
    budget: dict[str, int] = {}
    remaining = max_slots
    for name in ranked:
        if remaining <= 0:
            break
        take = min(per_class, remaining)
        budget[name] = take
        remaining -= take
    # Any leftover goes to the most confident findings, still one slot at a time.
    for name in ranked:
        if remaining <= 0:
            break
        budget[name] = budget.get(name, 0) + 1
        remaining -= 1
    return {name: slots for name, slots in budget.items() if slots > 0}


def _interpolate_coordinate(position: float) -> dict[str, float]:
    """Split one observed coordinate across the two bins it straddles.

    This is discretization, NOT measured uncertainty, and the decisions that
    carry it are tagged `discretization_interpolation` so the two are never
    conflated in a write-up.
    """
    position = min(max(position, 0.0), 1.0)
    scaled = position * COORD_BINS
    low = min(int(scaled), COORD_BINS - 1)
    high = min(low + 1, COORD_BINS - 1)
    target = {f"c{k}": 0.0 for k in range(COORD_BINS)}
    if low == high:
        target[f"c{low}"] = 1.0
    else:
        target[f"c{low}"] = 1.0 - (scaled - low)
        target[f"c{high}"] = scaled - low
    return target


def counting_decisions(counts: Mapping[str, int] | None = None,
                       legacy_uid: str | None = None) -> list[dict[str, Any]]:
    """TXL-PBC's count questions: the total, plus one per cell type.

    The per-type questions do double duty -- they are the counting task's
    per-type targets *and* the detection workflow's count stage, so "how many
    RBCs are present" is one decision serving two consumers.

    A type absent from `counts` emits NO question. The per-type breakdown
    exists only in the detection row, so an image without one genuinely does
    not know its RBC/WBC/platelet split; defaulting those to 0 would be
    fabricating labels rather than declining to ask.
    """
    span = [{"id": str(k), "text": str(k)} for k in range(COUNT_CAP + 1)]
    decisions: list[dict[str, Any]] = []
    if counts is None or "total" in counts:
        decisions.append(_decision(
            "count:total", "choice", "How many cells are present in this blood smear image?",
            [dict(c) for c in span], legacy_uid,
            one_hot_target(span, str(counts["total"])) if counts else {}, "one_hot"))
    for name in TXL_CELL_TYPES:
        if counts is not None and name not in counts:
            continue
        decisions.append(_decision(
            f"count:{name}", "choice", f"How many {name}s are present in this blood smear image?",
            [dict(c) for c in span], legacy_uid,
            one_hot_target(span, str(counts[name])) if counts else {}, "one_hot"))
    return decisions


def score_levels_decision(qid: str, instructions: str, levels: Sequence[str],
                          value: float | None, legacy_uid: str | None) -> dict[str, Any]:
    """A Score over explicitly described ordered levels (ordinal or numeric)."""
    candidates = [{"id": f"L{k}", "text": str(text)} for k, text in enumerate(levels)]
    target: dict[str, float] = {}
    kind = "one_hot"
    if value is not None:
        low = int(round(value))
        if not 0 <= low < len(candidates):
            raise ValueError(f"{qid}: value {value} is outside the declared levels")
        target = {c["id"]: 0.0 for c in candidates}
        fraction = float(value) - low
        if abs(fraction) < 1e-9 or low + 1 >= len(candidates):
            target[f"L{low}"] = 1.0
        else:
            target[f"L{low}"] = 1.0 - fraction
            target[f"L{low + 1}"] = fraction
            kind = "discretization_interpolation"
    return _decision(qid, "score", instructions, candidates, legacy_uid, target, kind)


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------

def assert_marker_ids(tokenizer) -> None:
    """Fail loudly if the markers do not tokenize to exactly one stable id each.

    The whole architecture assumes marker positions can be found by scanning
    for a token id. If that assumption breaks the failure mode is silent
    misalignment, so it is checked rather than trusted.
    """
    for text, expected in ((MARKER_KEY, MARKER_KEY_ID), (MARKER_QUERY, MARKER_QUERY_ID)):
        for context in (text, " " + text, "a" + text, "\n" + text):
            ids = tokenizer.encode(context, add_special_tokens=False)
            if expected not in ids:
                raise ValueError(
                    f"marker {text!r} is not a stable single token in context {context!r} "
                    f"(got {ids}, expected to contain {expected})"
                )


def decision_block(decisions: Sequence[Mapping[str, Any]]) -> str:
    """Render the decision questions as the text that follows the image.

    Layout, one block per question:

        Q1: <instructions>
        - <candidate text> ⟩
        - <candidate text> ⟩
        ✦
        Q2: ...

    Candidates are grouped into questions purely by position -- the keys
    between a question header and its `✦` belong to that question. That is the
    same positional contract NanoJev uses for Score level indices: no ordinal
    index or adjacent level is ever injected into a candidate's own text, so
    moving a candidate changes which slot it occupies but not how it reads.
    """
    lines: list[str] = []
    for index, decision in enumerate(decisions, start=1):
        typ = str(decision["type"])
        if typ not in ALL_TYPES:
            raise ValueError(f"Unknown decision type: {typ!r}")
        lines.append(f"Q{index}: {str(decision['instructions']).strip()}")
        candidates = decision["candidates"]
        if len(candidates) < 2:
            raise ValueError(f"Q{index}: a decision needs at least 2 candidates")
        for candidate in candidates:
            lines.append(f"- {str(candidate['text']).strip()} {MARKER_KEY}")
        lines.append(MARKER_QUERY)
    return "\n".join(lines)


def state_text(state: Mapping[str, Any]) -> str:
    """The non-image part of the state: modality and source dataset.

    Kept short on purpose. Evaluation information is never appended -- the
    model sees exactly what a deployed system would have.
    """
    parts = []
    if state.get("source"):
        parts.append(str(state["source"]))
    if state.get("modality"):
        parts.append(str(state["modality"]))
    return " | ".join(parts)


def encode_state(state: Mapping[str, Any], processor, images: Sequence[Any], max_length: int) -> dict[str, Any]:
    """Tokenize one state plus its decisions and locate every marker.

    Returns a dict with the model inputs and three parallel arrays describing
    where the decision markers landed:

        query_pos[s]      token position of the s-th question's `✦`
        key_pos[s]        list of token positions of that question's `⟩`
        cand_count[s]     number of candidates for that question

    The marker counts are asserted against the decision specification: a
    mismatch means a marker merged into a neighbouring token, which must fail
    here rather than mis-align the head downstream.
    """
    body = decision_block(state["decisions"])
    header = state_text(state)
    # Everything goes in the user turn. We never generate, so there is no
    # assistant turn to open -- the decision markers are just positions we read.
    content: list[dict[str, Any]] = [{"type": "image"} for _ in images]
    content.append({"type": "text", "text": f"{header}\n\n{body}\n" if header else f"{body}\n"})
    text = processor.apply_chat_template(
        [{"role": "user", "content": content}], add_generation_prompt=False, tokenize=False
    )
    batch = processor(text=[text], images=[list(images)], return_tensors="pt")
    input_ids = batch["input_ids"][0]

    ids = input_ids.tolist()
    query_pos = [i for i, value in enumerate(ids) if value == MARKER_QUERY_ID]
    key_pos = [i for i, value in enumerate(ids) if value == MARKER_KEY_ID]

    expected_questions = len(state["decisions"])
    expected_keys = sum(len(d["candidates"]) for d in state["decisions"])
    if len(query_pos) != expected_questions or len(key_pos) != expected_keys:
        raise ValueError(
            f"{state.get('state_id')}: marker count mismatch -- found "
            f"{len(query_pos)} query / {len(key_pos)} key markers but the state declares "
            f"{expected_questions} question(s) / {expected_keys} candidate(s). "
            "A marker token merged with its neighbours; do not train on this row."
        )

    # Assign keys to questions by position: everything before the s-th `✦`
    # and after the (s-1)-th belongs to question s.
    keys_by_question: list[list[int]] = []
    cursor = 0
    for boundary in query_pos:
        group = [p for p in key_pos[cursor:] if p < boundary]
        cursor += len(group)
        keys_by_question.append(group)
    if cursor != len(key_pos):
        raise ValueError(f"{state.get('state_id')}: key markers found after the last query marker")

    length = int(input_ids.shape[0])
    if max_length and length > max_length:
        raise ValueError(
            f"{state.get('state_id')}: encoded length {length} exceeds max_length={max_length}. "
            "Inputs are never silently truncated."
        )

    return {
        "input_ids": input_ids,
        "attention_mask": batch["attention_mask"][0],
        "pixel_values": batch["pixel_values"][0],
        "query_pos": query_pos,
        "keys_by_question": keys_by_question,
        "cand_count": [len(d["candidates"]) for d in state["decisions"]],
        "type": [str(d["type"]) for d in state["decisions"]],
    }


# --------------------------------------------------------------------------
# Head
# --------------------------------------------------------------------------

class MedJevHead(nn.Module):
    """Shared scorer: one set of weights for every question, every task family.

    There is deliberately no per-task head. `derma_head`, `path_head`,
    `retina_head` and friends are exactly what would make this ordinary
    multi-task learning instead of a System-One decision interface.

    Three additive terms produce each candidate's score:

        z[s,k] = W_s h_key[s,k]                  per-candidate scalar
               + <q_s, k_s,k> / sqrt(d)          question conditioning
               + set-attention delta             candidates see each other

    The q.k term is what carries the question: without it two questions that
    offer identical candidate texts would be scored by a context-free scalar.
    """

    def __init__(self, hidden_size: int, *, use_query: bool = True, use_set_head: bool = True,
                 attn_dim: int = 128, attn_heads: int = 4):
        super().__init__()
        self.use_query = use_query
        self.use_set_head = use_set_head
        self.norm = nn.LayerNorm(hidden_size)
        # Nonzero init so the first step is not dead.
        self.scalar = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.scalar.weight, std=0.02)
        nn.init.zeros_(self.scalar.bias)
        if use_query:
            self.query_proj = nn.Linear(hidden_size, attn_dim)
            self.key_proj = nn.Linear(hidden_size, attn_dim)
        if use_set_head:
            self.set_project = nn.Linear(hidden_size + 1, attn_dim)
            self.set_attention = nn.MultiheadAttention(attn_dim, attn_heads, dropout=0.0, batch_first=True)
            self.set_output = nn.Linear(attn_dim, 1)
            # Only the final residual projection starts at zero.
            nn.init.zeros_(self.set_output.weight)
            nn.init.zeros_(self.set_output.bias)

    def forward(self, hidden: torch.Tensor, key_sample: torch.Tensor, key_pos: torch.Tensor,
                query_sample: torch.Tensor, query_pos: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        """Score every candidate of every question in the batch.

        hidden        (B, L, H) last hidden states from the backbone
        key_sample    (Q,)      which batch element each question lives in
        key_pos       (Q, K)    candidate marker positions inside that element
        query_sample  (Q,)      batch element of each question's query marker
        query_pos     (Q,)      position of that query marker
        valid         (Q, K)    bool; false for padded candidate slots

        Returns (Q, K) logits, padded slots filled with -inf.
        """
        safe_pos = key_pos.clamp(min=0)
        h_key = hidden[key_sample.unsqueeze(1), safe_pos]           # (Q, K, H)
        h_key = self.norm(h_key)
        z = self.scalar(h_key).squeeze(-1).float()                  # (Q, K)

        if self.use_query:
            h_q = self.norm(hidden[query_sample, query_pos])        # (Q, H)
            q = self.query_proj(h_q).float()                        # (Q, d)
            k = self.key_proj(h_key).float()                        # (Q, K, d)
            z = z + (q.unsqueeze(1) * k).sum(-1) / math.sqrt(k.shape[-1])

        if self.use_set_head:
            count = valid.sum(-1).clamp(min=1).float()
            log_k = count.log()[:, None, None].expand(-1, key_pos.shape[1], 1)
            u = self.set_project(torch.cat([h_key, log_k.to(h_key.dtype)], dim=-1))
            mixed, _ = self.set_attention(u, u, u, key_padding_mask=~valid, need_weights=False)
            delta = self.set_output(torch.tanh(u + mixed)).squeeze(-1).float()
            z = z + delta

        return z.masked_fill(~valid, float("-inf"))


def masked_ce(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """The single loss behind every decision type.

    `-sum_k q_k log p_k`, averaged over valid questions. Choice, Noul
    (`C = {False, True}`) and Score all reduce to this; the target is dense so
    a one-hot and an interpolated soft target take the same path.
    """
    logp = torch.log_softmax(logits.float(), dim=-1)
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    per_question = -(safe_target * torch.where(valid, logp, torch.zeros_like(logp))).sum(-1)
    return per_question.sum() / valid.any(-1).sum().clamp(min=1)


# --------------------------------------------------------------------------
# Folding: distributions -> the prediction strings `omnimedjev evaluate` scores
# --------------------------------------------------------------------------
# This is the workflow layer, and it is deliberately plain code rather than
# model output. A complex task is a set of narrow typed decisions stitched
# together deterministically, and that is what makes the consistency claims
# hold by construction rather than by hoping the model complies -- `absent`
# cannot produce a box, because there is no code path that emits one.

def _argmax_id(distribution: Mapping[str, float]) -> str:
    return max(distribution, key=distribution.__getitem__)


def _level_index(candidate_id: str) -> int:
    """Recover the numeric level from a candidate id ("L7" or "7")."""
    text = str(candidate_id)
    return int(text[1:]) if text.startswith("L") else int(text)


def expected_level(distribution: Mapping[str, float]) -> float:
    """The probability-weighted level: `sum_k k * p_k`.

    This is why a numeric prediction is a distribution and not a scalar -- the
    expectation is the MAE-optimal point estimate, and the spread around it is
    available for free.
    """
    return sum(_level_index(cid) * float(p) for cid, p in distribution.items())


def expected_coordinate(distribution: Mapping[str, float], axis_size: float) -> float:
    """Expected value along one axis, in pixels."""
    centers = {f"c{k}": (k + 0.5) / COORD_BINS for k in range(COORD_BINS)}
    return sum(centers[cid] * float(p) for cid, p in distribution.items()) * axis_size


def fold_state(state: Mapping[str, Any], distributions: Mapping[str, Mapping[str, float]],
               *, presence_threshold: float = 0.5) -> dict[str, str]:
    """Turn one state's distributions into per-task legacy prediction strings.

    Returns a mapping of task family -> prediction string, keyed the way
    `omnimedjev evaluate` expects. Several tasks may be produced from one set of
    decisions: ChestX-Det's presence questions answer the multi-label task and
    open the detection workflow, and TXL-PBC's per-type counts serve the
    counting task and the detection workflow's count stage.
    """
    legacy = state.get("legacy_uids") or {}
    size = state.get("image_size")
    result: dict[str, str] = {}

    diagnosis = distributions.get("diagnosis")
    if diagnosis and "disease_diagnosis_classification" in legacy:
        result["disease_diagnosis_classification"] = _argmax_id(diagnosis)

    presence = {qid.split(":", 1)[1]: dist for qid, dist in distributions.items()
                if qid.startswith("present:")}

    if "count:total" in distributions and "cell_counting" in legacy:
        result["cell_counting"] = str(int(round(expected_level(distributions["count:total"]))))

    if diagnosis is None and not presence and "regression" in legacy:
        for qid in ("ef", "grade"):
            if qid in distributions:
                result["regression"] = f"{expected_level(distributions[qid]):.4f}"
                break

    if presence:
        positives = sorted(label for label, dist in presence.items()
                           if float(dist.get("yes", 0.0)) >= presence_threshold)
        if "multi_label_classification" in legacy:
            # The evaluator splits on ";" and treats "" as the empty label set,
            # which is exactly how the ground truth is spelled.
            result["multi_label_classification"] = "; ".join(positives)

    if "detection" in legacy:
        result["detection"] = json.dumps(_detect_boxes(presence, distributions, size, presence_threshold))

    return result


def _detect_boxes(presence: Mapping[str, Mapping[str, float]],
                  distributions: Mapping[str, Mapping[str, float]],
                  size: Sequence[int] | None,
                  presence_threshold: float) -> dict[str, list[list[float]]]:
    """Presence -> count -> coordinates, composed in code.

    A class whose presence probability is below threshold is skipped entirely,
    so its box list is absent rather than empty. That is the structural
    consistency the interface buys: "no pneumothorax" and "a pneumothorax box"
    cannot both appear, because nothing generates the box.
    """
    boxes: dict[str, list[list[float]]] = {}
    if size is None:
        return boxes
    width, height = float(size[0]), float(size[1])
    present = {name: float(dist.get("yes", 0.0)) for name, dist in presence.items()
               if float(dist.get("yes", 0.0)) >= presence_threshold}
    # Same per-state budget the questions were allocated under, so the fold can
    # never emit a box whose coordinates were never asked for.
    budget = localization_budget(list(present.items()))
    for class_name in present:
        slots = budget.get(class_name, 0)
        if not slots:
            continue
        count_dist = distributions.get(f"count:{class_name}")
        count = int(round(expected_level(count_dist))) if count_dist else slots
        count = max(0, min(count, slots))
        instances = []
        for index in range(count):
            coords = []
            for edge, axis_size in (("left", width), ("top", height),
                                    ("right", width), ("bottom", height)):
                axis_dist = distributions.get(f"detect:{class_name}:box{index}:{edge}")
                coords.append(expected_coordinate(axis_dist, axis_size) if axis_dist else 0.0)
            x1, y1, x2, y2 = coords
            # Emit a well-formed box direction or the IoU matcher discards it.
            instances.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
        if instances:
            boxes[class_name] = instances
    return boxes
