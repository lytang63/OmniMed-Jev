#!/usr/bin/env python3
"""Score every saved SFT checkpoint, mirroring `sweep_medjev.py`'s layout.

WHY the mirroring matters: the two sweeps are meant to be read side by side, so
a checkpoint must live at the same relative path on both sides. `omnimedjev infer`
insists on writing `{output_dir}/{experiment}-infer/`, which would give the SFT
side a different shape, so this drives the CLI and then files the artefacts into
`{workdir}/{checkpoint}/` the way the MedJev sweep does.

Resumable: a checkpoint whose `scores.json` already exists is skipped, so this
can be run now for the checkpoints that exist and re-run later to pick up the
rest without redoing anything.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/sweep_sft.py \
        --checkpoints output/sft-lora --gpus 0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Scratch directory for the per-checkpoint config files this sweep writes.
# Override with OMNIMED_TMPDIR when the system temp directory is too small.
TMPDIR = Path(os.environ.get("OMNIMED_TMPDIR") or tempfile.gettempdir())

TASKS = ("disease_diagnosis_classification", "cell_counting",
         "multi_label_classification", "regression")


def step_of(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)", path.name)
    return int(match.group(1)) if match else 10 ** 9


def run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=PROJECT_ROOT, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"failed ({result.returncode}): {' '.join(command)}\n  see {log}")


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


def process(checkpoint: Path, args, workdir: Path, gpu: str) -> str:
    run_dir = workdir / checkpoint.name
    if (run_dir / "scores.json").exists() and not args.force:
        return "cached"
    run_dir.mkdir(parents=True, exist_ok=True)
    logs = run_dir / "logs"
    TMPDIR.mkdir(parents=True, exist_ok=True)
    experiment = f"sft-{checkpoint.name}"

    infer_config = TMPDIR / f"infer-{experiment}.yaml"
    infer_config.write_text(
        f"split: testing\n"
        f"model_name_or_path: {args.model}\n"
        f"adapter_path: {checkpoint}\n"
        f"image_size: 448\nresize_mode: square\nmax_images_per_sample: 1\n"
        f"batch_size: 16\nmax_new_tokens: 256\ntemperature: 0.0\n"
        f"elicit_confidence: false\n", encoding="utf-8")
    run([sys.executable, "-u", "-m", "omnimedjev", "-n", experiment, "-d", "omnimed-suite",
         "--config", "local", "--root_dir", str(PROJECT_ROOT),
         "--input_dir", "datas/stub-input", "--output_dir", str(args.data_dir),
         "--custom_args", str(infer_config), "infer", *TASKS], logs / "infer.log")

    predictions = args.data_dir / f"{experiment}-infer" / "testing_predictions.jsonl"
    if not predictions.exists():
        raise FileNotFoundError(f"no predictions at {predictions}")

    eval_config = TMPDIR / f"eval-{experiment}.yaml"
    eval_config.write_text(
        f"split: testing\n"
        f"predictions: {predictions}\n"
        f"eval_output_dir: {run_dir}\n"
        f"allow_missing_predictions: true\niou_threshold: 0.5\n"
        f"skip_green_score: true\nskip_crimson_score: true\n", encoding="utf-8")
    run([sys.executable, "-u", "-m", "omnimedjev", "-n", experiment, "-d", "omnimed-suite",
         "--config", "local", "--root_dir", str(PROJECT_ROOT),
         "--input_dir", "datas/stub-input", "--output_dir", str(args.data_dir),
         "--custom_args", str(eval_config), "evaluate", *TASKS], logs / "evaluate.log")

    # File the artefacts the way the MedJev sweep does, then drop the CLI's dirs.
    shutil.copy(predictions, run_dir / "testing_predictions.jsonl")
    for stale in (args.data_dir / f"{experiment}-infer", args.data_dir / f"{experiment}-eval"):
        shutil.rmtree(stale, ignore_errors=True)
    return "done"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", required=True, type=Path)
    ap.add_argument("--data-dir", type=Path, default=Path("datas/preprocessed"),
                    help="holds Preprocessed-<dataset>/; inference reads its testing split")
    ap.add_argument("--workdir", type=Path, default=Path("datas/eval/runs-sft"))
    ap.add_argument("--model", default="models/medgemma-1.5-4b-it")
    ap.add_argument("--gpus", default="0",
                    help="comma-separated devices; a single inference leaves the GPU mostly idle")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    devices = [d.strip() for d in str(args.gpus).split(",") if d.strip()]

    checkpoints = find_checkpoints(args.checkpoints)
    if not checkpoints:
        raise SystemExit(f"no checkpoints under {args.checkpoints}")
    args.workdir.mkdir(parents=True, exist_ok=True)
    print(f"scoring {len(checkpoints)} SFT checkpoint(s) into {args.workdir}")

    failures = []
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        running = {pool.submit(process, c, args, args.workdir, devices[i % len(devices)]): c.name
                   for i, c in enumerate(checkpoints)}
        for future in as_completed(running):
            name = running[future]
            try:
                print(f"  {name}: {future.result()}", flush=True)
            except Exception as exc:
                print(f"  {name} FAILED: {exc}", flush=True)
                failures.append(name)

    rows = []
    for checkpoint in checkpoints:
        if checkpoint.name in failures:
            continue
        path = args.workdir / checkpoint.name / "scores.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        block = payload.get("mean") or payload.get("mean_metrics") or payload.get("metrics") or payload
        rows.append((step_of(checkpoint), {k: float(v) for k, v in block.items()
                                           if isinstance(v, (int, float))}))
    if rows:
        heads = (("balanced_accuracy", "Classification"), ("multi_label_f1_score", "Multi-label"),
                 ("cell_counting_mean_absolute_error", "Counting"),
                 ("regression_mean_absolute_error", "Regression"))
        lines = ["# SFT checkpoint sweep", "", "| step | " + " | ".join(h[1] for h in heads) + " |",
                 "|---:|" + "---:|" * len(heads)]
        for step, metrics in sorted(rows):
            lines.append(f"| {step} | " + " | ".join(
                f"{metrics[h[0]]:.4f}" if h[0] in metrics else "—" for h in heads) + " |")
        (args.workdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
    if failures:
        print(f"\nfailed: {failures}")


if __name__ == "__main__":
    main()
