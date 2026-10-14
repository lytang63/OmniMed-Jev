# Metrics

Every quantity reported in the paper is produced by one pure-function module,
[`omnimedjev/engine/medjev_metrics.py`](../omnimedjev/engine/medjev_metrics.py),
so a metric means the same thing on both sides of the comparison. The
generative arm's distribution is reconstructed from its logits
([`scripts/eval_sft_logit_conf.py`](../scripts/eval_sft_logit_conf.py)) and then
scored by those same functions.

`scripts/eval_medjev_metrics.py` writes the full set to `metrics.json` /
`metrics.md` for a checkpoint.

## The reported probability

For all calibration metrics the "reported probability" is the probability the
model assigns to its **top candidate**:

- `Choice` — the class it selects,
- `Noul` — the finding it considers present,
- `Score` — the annotated value.

It is a probability about the decision, not the probability that a disease is
present or that a measurement is clinically acceptable. Those are different
quantities, defined by a downstream rule.

## Calibration

**ECE** bins decisions by reported confidence and averages the gap between
confidence and accuracy:

```
ECE = Σ_b (n_b / n) · |acc(b) − conf(b)|
```

It answers the question a clinician asks of a reported probability: when the
model says it is right with probability *p*, is it right about *p* of the time?

**MCE** is the worst bin rather than the average, and is the reason ECE is never
read alone: ECE can hide a model that is overconfident in one region of
confidence space and underconfident in another.

## Reliability vs. resolution

The Brier score decomposes into a calibration term, a discrimination term and an
irreducible constant:

```
BS = Σ_b (n_b/n)(conf_b − acc_b)²  −  Σ_b (n_b/n)(acc_b − ȳ)²  +  ȳ(1 − ȳ)
     \_________________________/      \____________________/      \______/
            reliability                    resolution            uncertainty
```

with `ȳ` the base correctness rate.

- **Reliability** asks whether the reported numbers are probabilities.
- **Resolution** asks whether the ranking separates correct from incorrect
  decisions.

Both must be reported together, because either one alone is uninformative: a
model that always emits the class prior has reliability error near zero and
resolution exactly zero, and a model that ranks well but reports unusable
numbers has high resolution *and* high reliability error. That pair is what
separates the two failure modes, and it is why the paper's central claim is
about reliability rather than accuracy.

## Selective prediction

Ordering decisions by confidence and answering only the most confident fraction
gives a risk–coverage curve:

```
risk(c) = error among the c most confident decisions
AURC    = ∫₀¹ risk(c) dc
```

This is the quantity a deferral policy actually consumes: it measures whether
low confidence marks the decisions worth handing to a clinician. `risk@k` is
read off the same curve at a fixed coverage.

A favourable AURC is a statement about the quality of an *ordering*, not about
safety: a model can rank well and still be miscalibrated for a subpopulation.

## Decision-space summaries

- **`DDx@0.9`** — the number of top-ranked candidates needed to carry 0.9 of the
  probability mass. It shrinks as the distribution concentrates.
- **top-k coverage** — whether the annotated candidate is inside the top *k*.
- **normalized entropy** — the shape of the distribution; descriptive only.

`DDx@0.9` and top-k coverage are interpretable but threshold-dependent, and they
inherit whatever error the candidate grid introduces. Note that a shorter
shortlist is *not* the same claim as the annotated candidate being inside a
shortlist of a given size — the two should not be conflated.

## Intervals

Proportions are reported with **Wilson intervals**, which stay inside [0, 1]
where several of these quantities sit. Model-to-model differences are reported
with **paired bootstrap intervals** (20,000 resamples of rows, seed 20260925),
because the two models score the same rows and an unpaired standard error would
overstate the uncertainty.

## Soundness checks worth running

`scripts/compare_medjev.py` cross-checks a checkpoint's task metrics against the
generative arm's, and `scripts/probe_medjev.py` re-scores the same states with
the candidate set perturbed — permuted, restricted to a sub-list, or with the
annotated candidate removed. Those probes are only defined because the candidate
set is part of the input.
