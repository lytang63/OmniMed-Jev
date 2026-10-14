"""Probability metrics for MedJev, beyond a single accuracy number.

WHY this module exists separately from `omnimedjev/engine/evaluate.py`: every metric
there is computed from a prediction *string*, so `p=0.51` and `p=0.99` score
identically. Everything here reads the distribution directly.

The metrics fall into three groups, and the distinction matters when writing
them up:

* **Calibration** -- is the stated probability true? ECE, MCE, and the Murphy
  decomposition, which separates *reliability* (are the probabilities honest)
  from *resolution* (does the model actually discriminate). A model that
  always outputs the class prior is perfectly reliable and has zero
  resolution, so an ECE alone cannot distinguish it from a good model.

* **Selective prediction** -- can the model be trusted to know when to defer?
  Risk-coverage curves, AURC, and selective risk at a target coverage. This is
  the metric that matters for deployment ("hand it to a radiologist when
  unsure") and the generative baseline structurally cannot produce it: a
  sampled answer token carries no decision confidence.

* **Decision-set shape** -- how many candidates does the model actually
  consider? Top-k coverage and the differential-diagnosis set size, which is
  what a clinician reads off such a model: not "the answer" but "the shortlist
  it is committing to".

All functions are pure and take plain Python lists, so they are testable
without a checkpoint.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

DEFAULT_BINS = 15


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

def _bin_members(confidences: Sequence[float], bins: int) -> list[list[int]]:
    members: list[list[int]] = [[] for _ in range(bins)]
    for index, value in enumerate(confidences):
        slot = min(int(value * bins), bins - 1)
        members[slot].append(index)
    return members


def expected_calibration_error(confidences: Sequence[float], correct: Sequence[bool],
                               bins: int = DEFAULT_BINS) -> float:
    """Sample-weighted mean gap between confidence and accuracy (ECE).

    Bins by stated confidence, then compares the average confidence in each bin
    with the observed accuracy there. Zero for a perfectly calibrated model;
    an overconfident one is penalised by the size of the gap.
    """
    if not confidences:
        return float("nan")
    total = len(confidences)
    error = 0.0
    for members in _bin_members(confidences, bins):
        if not members:
            continue
        mean_confidence = sum(confidences[i] for i in members) / len(members)
        mean_accuracy = sum(1.0 for i in members if correct[i]) / len(members)
        error += (len(members) / total) * abs(mean_confidence - mean_accuracy)
    return error


def max_calibration_error(confidences: Sequence[float], correct: Sequence[bool],
                          bins: int = DEFAULT_BINS) -> float:
    """The largest single-bin confidence/accuracy gap.

    ECE can hide a badly miscalibrated region behind well-calibrated ones; MCE
    is the worst case, which is what matters if a downstream threshold sits in
    that region.
    """
    if not confidences:
        return float("nan")
    worst = 0.0
    for members in _bin_members(confidences, bins):
        if not members:
            continue
        mean_confidence = sum(confidences[i] for i in members) / len(members)
        mean_accuracy = sum(1.0 for i in members if correct[i]) / len(members)
        worst = max(worst, abs(mean_confidence - mean_accuracy))
    return worst


def murphy_decomposition(confidences: Sequence[float], correct: Sequence[bool],
                         bins: int = DEFAULT_BINS) -> dict[str, float]:
    """Split the Brier score into reliability, resolution and uncertainty.

        Brier = reliability - resolution + uncertainty

    * **reliability** (lower better) is the calibration term: 0 for a model
      whose stated confidence matches its accuracy.
    * **resolution** (higher better) is the discrimination term: how far the
      per-bin accuracy moves away from the base rate. A model that always
      emits the prior has resolution 0 and cannot be told from a useful one by
      calibration alone.
    * **uncertainty** is a property of the data, not the model.

    Computed on the top-label forecast: the confidence in the chosen candidate
    against whether that candidate was right. That is the standard reference
    forecast for a multi-class distribution and is stated here so the numbers
    are not mistaken for a per-candidate decomposition.
    """
    if not confidences:
        return {"reliability": float("nan"), "resolution": float("nan"),
                "uncertainty": float("nan"), "brier": float("nan")}
    total = len(confidences)
    base_rate = sum(1.0 for flag in correct if flag) / total
    reliability = resolution = 0.0
    for members in _bin_members(confidences, bins):
        if not members:
            continue
        weight = len(members) / total
        mean_confidence = sum(confidences[i] for i in members) / len(members)
        mean_accuracy = sum(1.0 for i in members if correct[i]) / len(members)
        reliability += weight * (mean_confidence - mean_accuracy) ** 2
        resolution += weight * (mean_accuracy - base_rate) ** 2
    uncertainty = base_rate * (1.0 - base_rate)
    return {
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "brier": reliability - resolution + uncertainty,
        "base_rate": base_rate,
    }


# --------------------------------------------------------------------------
# Selective prediction
# --------------------------------------------------------------------------

def risk_coverage(confidences: Sequence[float], correct: Sequence[bool]) -> dict[str, float]:
    """Risk (error rate) as a function of how much the model is allowed to answer.

    Sort by confidence, keep the most confident fraction `c`, and measure the
    error rate among them. A model that knows when it does not know shows a
    steeply rising curve.

    * `aurc` -- area under that curve; lower is better.
    * `e_aurc` -- excess over the best achievable ordering, which is the
      oracle that sorts by correctness. Scale-free, unlike AURC.
    * `risk_at_*` -- the error rate if only the top 80/90/95% are answered.
    """
    if not confidences:
        return {"aurc": float("nan"), "e_aurc": float("nan")}
    order = sorted(range(len(confidences)), key=lambda i: -confidences[i])
    errors = 0
    curve = []
    for rank, index in enumerate(order, start=1):
        errors += 0 if correct[index] else 1
        curve.append(errors / rank)          # risk when covering `rank` items
    n = len(order)
    aurc = sum(curve) / n

    # The oracle ordering puts every correct item first, so its curve is the
    # running error rate of a sorted-by-correctness list.
    total_errors = sum(1 for flag in correct if not flag)
    oracle, seen = [], 0
    for rank in range(1, n + 1):
        if rank <= n - total_errors:
            oracle.append(0.0)               # correct item, risk still 0
        else:
            seen += 1
            oracle.append(seen / rank)
    best = sum(oracle) / n

    out = {"aurc": aurc, "best_aurc": best, "e_aurc": aurc - best}
    for coverage in (0.5, 0.8, 0.9, 0.95):
        keep = max(1, int(round(n * coverage)))
        out[f"risk_at_{int(coverage * 100)}"] = curve[keep - 1]
    return out


def coverage_at_k(sorted_probabilities: Sequence[float], gold_index: int, k: int) -> bool:
    """Was the correct candidate among the top `k`? (k=1 is plain accuracy.)"""
    ranked = sorted(range(len(sorted_probabilities)),
                    key=lambda i: -sorted_probabilities[i])
    return gold_index in ranked[:k]


def ddx_set_size(probabilities: Sequence[float], tau: float = 0.9) -> int:
    """How many candidates it takes to accumulate probability `tau`.

    The nearest thing to what a clinician reads off the model: not the single
    best answer but the shortlist it is committing to. A model firing
    everything at one candidate gives 1; a uniform one needs all of them.
    """
    if not probabilities:
        return 0
    ordered = sorted(probabilities, reverse=True)
    total = sum(ordered)
    if total <= 0:
        return len(ordered)
    running = 0.0
    for index, value in enumerate(ordered, start=1):
        running += value / total
        if running >= tau:
            return index
    return len(ordered)


def entropy(probabilities: Sequence[float], normalize: bool = True) -> float:
    """Shannon entropy of a distribution, optionally as a fraction of the maximum.

    Normalised entropy is `1.0` for a uniform distribution over the candidates
    and `0.0` for a point mass, which makes decisions with different candidate
    counts comparable.
    """
    values = [p for p in probabilities if p > 0]
    if not values:
        return float("nan")
    raw = -sum(p * math.log(p) for p in values)
    if not normalize:
        return raw
    k = len(probabilities)
    return raw / math.log(k) if k > 1 else 0.0


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval -- the honest error bar on a proportion.

    Used instead of the normal approximation because several of these metrics
    sit near 0 or 1, where that approximation breaks down.
    """
    if total == 0:
        return (float("nan"), float("nan"))
    phat = successes / total
    denominator = 1 + z * z / total
    centre = (phat + z * z / (2 * total)) / denominator
    half = z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def mean(values: Iterable[float]) -> float:
    items = [v for v in values]
    return sum(items) / len(items) if items else float("nan")
