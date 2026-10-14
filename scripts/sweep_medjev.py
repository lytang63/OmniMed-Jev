#!/usr/bin/env python3
"""Score every saved MedJev checkpoint through the identical path.

WHY a sweep rather than a single final number: one score cannot distinguish
"the decision interface works" from "it needed more steps than the generative
baseline got". The generative run was swept the same way (ft233 / ft466 / ...),
so this puts the two training curves on the same axes.

Every checkpoint goes through exactly the pipeline used for the headline
result -- `infer_medjev.py` folds the distributions into prediction strings,
`omnimedjev evaluate` scores those with the unchanged string evaluator, and
`eval_medjev_metrics.py` records what the string evaluator cannot see -- so
nothing here is a special case.

Layout (one self-contained directory per checkpoint, so a run is reproducible
from its own folder and the sweep can be resumed):

    datas/eval/runs-medjev/
        checkpoint-233/
            testing_predictions.jsonl   folded predictions (omnimedjev evaluate input)
            testing_decisions.jsonl     full distributions
            scores.json, details.json   omnimedjev evaluate output
            metrics.json, metrics.md    probability metrics
            logs/                       per-stage stdout
        ...
        summary.json, summary.md

Detection is deliberately not scored here; see `eval_medjev_metrics.py`.

Usage:
    python scripts/sweep_medjev.py --checkpoints output/medjev-lora --gpus 0
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Detection is excluded on request: its metric never became competitive and the
# eval subset cannot support the workflow-consistency claim either way.
TASKS = ("disease_diagnosis_classification", "cell_counting",
         "multi_label_classification", "regression")

# task -> (display name, headline metric key, lower_is_better)
HEADLINES = (
    ("disease_diagnosis_classification", "Classification", "balanced_accuracy", False),
    ("multi_label_classification", "Multi-label", "multi_label_f1_score", False),
    ("cell_counting", "Counting", "cell_counting_mean_absolute_error", True),
    ("regression", "Regression", "regression_mean_absolute_error", True),
)

# Extra columns pulled from metrics.json, reported per family.
CALIBRATION_ROWS = (
    ("disease_diagnosis_classification", "Classification"),
    ("multi_label_classification", "Multi-label"),
    ("cell_counting", "Counting"),
    ("regression", "Regression"),
)


def step_of(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)", path.name)
    return int(match.group(1)) if match else 10 ** 9


def run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=PROJECT_ROOT, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"failed ({result.returncode}): {' '.join(command)}\n  see {log}")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metrics_of(scores_path: Path) -> dict[str, float]:
    """Headline metrics out of an `omnimedjev evaluate` scores.json (either shape)."""
    payload = read_json(scores_path)
    for key in ("mean", "mean_metrics", "metrics"):
        block = payload.get(key)
        if isinstance(block, dict):
            return {k: float(v) for k, v in block.items() if isinstance(v, (int, float))}
    return {k: float(v) for k, v in payload.items() if isinstance(v, (int, float))}


def find_checkpoints(root: Path) -> list[Path]:
    """Resolve "a directory of checkpoints" vs "one checkpoint directory".

    Decided by looking for `checkpoint-*` CHILDREN first, not by testing the
    directory itself for `adapter_config.json`: at the end of training
    `save_model` writes the adapter to the run's own root, so the root and its
    `checkpoint-*` children both look like checkpoints and the naive test
    returns the root alone -- silently skipping every saved step.
    """
    children = sorted((p for p in root.iterdir()
                       if p.is_dir() and re.fullmatch(r"checkpoint-\d+", p.name)
                       and (p / "adapter_config.json").exists()),
                      key=step_of)
    if children:
        return children
    if (root / "adapter_config.json").exists():
        return [root]
    return []


def process(checkpoint: Path, args, workdir: Path, gpu: str) -> list[str]:
    """Run one checkpoint end to end, skipping stages already on disk.

    Returns the stages that actually ran, so a caller can tell a cached
    checkpoint from a fresh one.
    """
    run_dir = workdir / checkpoint.name
    run_dir.mkdir(parents=True, exist_ok=True)
    logs = run_dir / "logs"
    stages: list[str] = []

    predictions = run_dir / "testing_predictions.jsonl"
    if args.force or not predictions.exists():
        run([sys.executable, "-u", "scripts/infer_medjev.py",
             "--checkpoint", str(checkpoint), "--data", str(args.data),
             "--out-dir", str(run_dir), "--device", f"cuda:{gpu}",
             "--model", args.model], logs / "infer.log")
        stages.append("infer")

    scores = run_dir / "scores.json"
    if args.force or not scores.exists():
        # `omnimedjev` accepts JSON for --custom_args, so no YAML needs to be written.
        evaluate_args = run_dir / "evaluate_args.json"
        evaluate_args.write_text(json.dumps({
            "split": "testing",
            "predictions": str(predictions.resolve()),
            "eval_output_dir": str(run_dir.resolve()),
            "allow_missing_predictions": True,
            "iou_threshold": 0.5,
            "skip_green_score": True,
            "skip_crimson_score": True,
        }, indent=2), encoding="utf-8")
        run([sys.executable, "-u", "-m", "omnimedjev", "-n", "medjev", "-d", "omnimed-suite",
             "--config", "local", "--root_dir", str(PROJECT_ROOT),
             "--input_dir", "datas/stub-input", "--output_dir", "datas/eval",
             "--custom_args", str(evaluate_args), "evaluate", *TASKS],
            logs / "evaluate.log")
        stages.append("evaluate")

    metrics = run_dir / "metrics.json"
    if args.force or not metrics.exists():
        run([sys.executable, "-u", "scripts/eval_medjev_metrics.py",
             "--states", str(args.states), "--decisions", str(run_dir / "testing_decisions.jsonl"),
             "--predictions", str(predictions), "--ground-truth", str(args.ground_truth),
             "--output", str(metrics)], logs / "metrics.log")
        stages.append("metrics")
    return stages


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", required=True, type=Path,
                    help="directory of checkpoint-N subdirs, or a single checkpoint")
    ap.add_argument("--data", type=Path, default=Path("datas/medjev/testing.jsonl"))
    ap.add_argument("--states", type=Path, default=Path("datas/medjev/testing.jsonl"))
    ap.add_argument("--ground-truth", type=Path,
                    default=Path("datas/eval/Preprocessed-omnimed-suite/testing.jsonl"))
    ap.add_argument("--workdir", type=Path, default=Path("datas/eval/runs-medjev"))
    ap.add_argument("--model", default="models/medgemma-1.5-4b-it")
    ap.add_argument("--gpus", default="0",
                    help="comma-separated CUDA devices to spread the sweep over. A single "
                         "inference uses ~9 GB and leaves the GPU mostly idle, so several "
                         "checkpoints run concurrently at no cost.")
    ap.add_argument("--force", action="store_true", help="redo stages that already have output")
    args = ap.parse_args()
    devices = [d.strip() for d in str(args.gpus).split(",") if d.strip()]
    if not devices:
        raise SystemExit("--gpus needs at least one device")

    checkpoints = find_checkpoints(args.checkpoints)
    if not checkpoints:
        raise SystemExit(f"no checkpoints with medjev_head.safetensors under {args.checkpoints}")
    args.workdir.mkdir(parents=True, exist_ok=True)
    print(f"scoring {len(checkpoints)} checkpoint(s) into {args.workdir} "
          f"across cuda:{','.join(devices)}")

    # One worker per device. The stages are subprocesses, so threads are only
    # orchestrating -- there is no Python-level work to contend on.
    failures: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        running = {}
        for index, checkpoint in enumerate(checkpoints):
            gpu = devices[index % len(devices)]
            running[pool.submit(process, checkpoint, args, args.workdir, gpu)] = (checkpoint.name, gpu)
        for future in as_completed(running):
            name, gpu = running[future]
            try:
                stages = future.result()
                print(f"  {name} done on cuda:{gpu}"
                      f"{' (' + ', '.join(stages) + ')' if stages else ' (all cached)'}", flush=True)
            except Exception as exc:                       # keep going; report at the end
                print(f"  {name} FAILED on cuda:{gpu}: {exc}", flush=True)
                failures.append((name, str(exc)))
    if failures:
        print(f"\n{len(failures)} checkpoint(s) failed; their summaries are omitted.")
        for name, error in failures:
            print(f"  {name}: {error}")

    failed = {name for name, _ in failures}
    rows = []
    for checkpoint in checkpoints:
        if checkpoint.name in failed:
            continue
        run_dir = args.workdir / checkpoint.name
        if not (run_dir / "scores.json").exists() or not (run_dir / "metrics.json").exists():
            continue
        row = {"checkpoint": checkpoint.name, "step": step_of(checkpoint),
               "headline": metrics_of(run_dir / "scores.json"),
               "metrics": read_json(run_dir / "metrics.json")}
        rows.append(row)

    summary = {row["checkpoint"]: {"step": row["step"], "headline": row["headline"],
                                   "families": row["metrics"]["families"]} for row in rows}
    (args.workdir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    lines = ["# MedJev checkpoint sweep", "",
             "Same base model, same training data, same evaluator as the generative baseline.",
             "Detection is not scored (see `eval_medjev_metrics.py`).", "",
             "## Task metrics", "",
             "| step | " + " | ".join(name for _, name, _, _ in HEADLINES) + " |",
             "|---:|" + "---:|" * len(HEADLINES)]
    for row in rows:
        cells = []
        for _, _, metric, _ in HEADLINES:
            value = row["headline"].get(metric)
            cells.append(f"{value:.4f}" if value is not None else "—")
        lines.append(f"| {row['step']} | " + " | ".join(cells) + " |")

    lines += ["", "## Probability metrics", "",
              "| step | family | n | acc | ECE | MCE | reliability | resolution | AURC | ddx@0.9 | top3 |",
              "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        for family, label in CALIBRATION_ROWS:
            block = row["metrics"]["families"].get(family)
            if not block:
                continue
            def g(key, fmt="{:.4f}"):
                value = block.get(key)
                return fmt.format(value) if isinstance(value, (int, float)) else "—"
            lines.append(f"| {row['step']} | {label} | {block.get('decisions', 0)} | {g('accuracy')} | "
                         f"{g('ece')} | {g('mce')} | {g('reliability')} | {g('resolution')} | "
                         f"{g('aurc')} | {g('ddx_set_size_at_0.9', '{:.1f}')} | {g('top3_coverage')} |")

    report = "\n".join(lines) + "\n"
    (args.workdir / "summary.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote {args.workdir / 'summary.md'} and summary.json")


if __name__ == "__main__":
    main()
