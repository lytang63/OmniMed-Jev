import gc
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from rich.console import Console

from omnimedjev.engine.evaluate import (
    MODEL_ID,
    TASK_INSTRUCTIONS,
    as_text,
    load_converted_split,
    maybe_json_load,
    normalize_task_list,
    normalize_task_name,
    normalize_text,
    predictions_out_path_for_split,
    resolve_eval_splits,
    split_confidence,
    write_json,
    write_jsonl,
)
from omnimedjev.vars import ExpConfig

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from PIL import Image, ImageOps
except Exception:  # pragma: no cover
    Image = None
    ImageOps = None

try:
    from peft import PeftModel
except Exception:  # pragma: no cover
    PeftModel = None

try:
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
except Exception:  # pragma: no cover
    AutoModelForImageTextToText = None
    AutoProcessor = None
    BitsAndBytesConfig = None


def infer(config: ExpConfig, tasks: Sequence[str], use_wandb: bool, smoke_test: bool, *, console: Console = Console(),
          **kwargs) -> None:
    """
    This is a template entrypoint for inference. You MUST NOT change its signature, but you may add functions and
    classes to this file.

    All your logs MUST be sent to the provided console. Your implementation MUST support WandB logging and it MUST ONLY
    be enabled if :param use_wandb is `True`.

    :param config: experiment configuration
    :param tasks: the tasks to evaluate on
    :param use_wandb: whether to use wandb for logging
    :param smoke_test: whether to run in smoke test mode
    :param console: the console for logging
    :param kwargs: custom arguments
    """
    kwargs = dict(kwargs)
    if smoke_test:
        console.print("Smoke test mode: limiting inference samples and generation length.")
        kwargs.setdefault("max_samples", 4)
        kwargs.setdefault("batch_size", 1)
        kwargs.setdefault("image_size", 512)
        kwargs.setdefault("max_new_tokens", 32)

    selected_tasks = normalize_task_list(tasks or kwargs.get("tasks") or TASK_INSTRUCTIONS)
    splits = resolve_eval_splits(kwargs)
    output_dir = Path(
        kwargs.get("infer_output_dir")
        or kwargs.get("predictions_output_dir")
        or kwargs.get("eval_output_dir")
        or Path(config.output_dir) / f"{config.experiment_name}-infer"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    split_results = {}
    model_bundle = None
    wandb_run = start_wandb_run(config, output_dir, selected_tasks, splits, kwargs) if use_wandb else None
    completed = False
    try:
        if wandb_run is not None:
            wandb_run.log({"infer/status": "started", "infer/num_splits": len(splits)})
        if len(splits) > 1:
            model_bundle = load_model_and_processor(config, kwargs)
        for split in splits:
            split_results[split] = infer_one_split(
                config,
                selected_tasks,
                split,
                output_dir,
                console,
                kwargs,
                model_bundle,
                wandb_run=wandb_run,
            )

        details = {
            "splits": splits,
            "tasks": selected_tasks,
            "model_variant": selected_model_variant(config, kwargs),
            "split_results": split_results,
            "num_predictions": sum(int(result["num_predictions"]) for result in split_results.values()),
        }
        details_path = Path(kwargs.get("infer_details_json") or output_dir / "inference_details.json")
        write_json(details_path, details)
        console.print(f"Saved inference details to {details_path}")

        if wandb_run is not None:
            wandb_run.log({"infer/status": "completed", "infer/num_predictions": details["num_predictions"]})
            for split, result in split_results.items():
                wandb_run.log({f"infer/{split}/num_predictions": result["num_predictions"]})
            log_wandb_file(wandb_run, details_path, "inference-details")
        completed = True
    finally:
        if model_bundle is not None:
            del model_bundle
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        if wandb_run is not None:
            if not completed:
                wandb_run.log({"infer/status": "failed"})
            wandb_run.finish(exit_code=0 if completed else 1)


def infer_one_split(
    config: ExpConfig,
    selected_tasks: Sequence[str],
    split: str,
    output_dir: Path,
    console: Console,
    kwargs: Mapping[str, Any],
    model_bundle: tuple[Any, Any, str] | None = None,
    wandb_run: Any | None = None,
) -> dict[str, Any]:
    console.print(f"Loading {split} rows from {config.preprocessed_dataset_dir}")
    rows = load_converted_split(Path(config.preprocessed_dataset_dir), split, optional_int(kwargs.get("max_samples")))
    rows = filter_inference_rows(rows, selected_tasks)
    if not rows:
        raise RuntimeError(f"No rows found for split={split!r} and tasks={selected_tasks}")
    console.print(f"Generating predictions for {len(rows)} row(s) across {', '.join(selected_tasks)}")
    if wandb_run is not None:
        wandb_run.log({f"infer/{split}/status": "started", f"infer/{split}/num_rows": len(rows)})

    prediction_records = generate_predictions(config, rows, console, dict(kwargs), model_bundle=model_bundle, split=split, wandb_run=wandb_run)
    predictions_out = predictions_out_path_for_split(kwargs.get("predictions_out"), output_dir, split)
    write_jsonl(predictions_out, prediction_records)
    console.print(f"Saved generated predictions to {predictions_out}")
    if wandb_run is not None:
        wandb_run.log({f"infer/{split}/status": "completed", f"infer/{split}/num_predictions": len(prediction_records)})
        log_wandb_file(wandb_run, predictions_out, f"predictions-{split}")
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "split": split,
        "predictions": str(predictions_out),
        "num_predictions": len(prediction_records),
        "num_rows": len(rows),
        "model_variant": selected_model_variant(config, kwargs),
    }


def optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    out = int(value)
    return out if out > 0 else None


def start_wandb_run(config: ExpConfig, output_dir: Path, selected_tasks: Sequence[str], splits: Sequence[str], kwargs: Mapping[str, Any]):
    import wandb

    wandb_dir = Path(kwargs.get("wandb_dir") or output_dir / "wandb")
    wandb_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_DIR", str(wandb_dir))
    os.environ.setdefault("WANDB_MODE", str(kwargs.get("wandb_mode", "online")))
    os.environ.setdefault("WANDB_PROJECT", str(kwargs.get("wandb_project", "omnimed-jev")))
    if kwargs.get("wandb_entity"):
        os.environ.setdefault("WANDB_ENTITY", str(kwargs["wandb_entity"]))
    if kwargs.get("wandb_tags"):
        os.environ.setdefault("WANDB_TAGS", str(kwargs["wandb_tags"]))
    os.environ.pop("WANDB_DISABLED", None)

    return wandb.init(
        project=str(kwargs.get("wandb_project", os.environ["WANDB_PROJECT"])),
        name=str(kwargs.get("wandb_run_name", f"{config.experiment_name}-infer")),
        dir=str(wandb_dir),
        config={
            "splits": list(splits),
            "tasks": list(selected_tasks),
            "model_variant": selected_model_variant(config, kwargs),
            "model_name_or_path": str(kwargs.get("model_name_or_path", MODEL_ID)),
            "batch_size": int(kwargs.get("batch_size", 1)),
            "max_new_tokens": int(kwargs.get("max_new_tokens", 256)),
        },
        settings=wandb.Settings(init_timeout=int(os.environ.get("WANDB_INIT_TIMEOUT", "300"))),
    )


def log_wandb_file(wandb_run: Any, path: Path, artifact_name: str) -> None:
    try:
        import wandb

        artifact = wandb.Artifact(artifact_name, type="inference-output")
        artifact.add_file(str(path))
        wandb_run.log_artifact(artifact)
    except Exception:
        try:
            wandb_run.save(str(path))
        except Exception:
            pass


def filter_inference_rows(rows: Sequence[dict[str, Any]], tasks: Sequence[str]) -> list[dict[str, Any]]:
    selected = set(tasks)
    out = []
    for row in rows:
        task = normalize_task_name(row.get("task_type", ""))
        if task not in selected:
            continue
        row = dict(row)
        row["task_type"] = task
        out.append(row)
    return out


def get_image_paths(row: Mapping[str, Any]) -> list[str]:
    images = maybe_json_load(row.get("images"))
    if isinstance(images, list):
        paths = [str(path).strip() for path in images if str(path).strip()]
        if paths:
            return paths
    if isinstance(images, str) and images.strip():
        return [images.strip()]
    for key in ("image_path", "image", "volume_path"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return [value.strip()]
    raise KeyError(f"No image path found for uid={row.get('uid')}")


def load_image(path: str, image_size: int, resize_mode: str):
    if Image is None or ImageOps is None:
        raise RuntimeError("Pillow is required for inference.")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
    if image_size and image_size > 0:
        resample = Image.Resampling.BICUBIC
        if resize_mode == "square":
            image = image.resize((image_size, image_size), resample)
        elif resize_mode == "longest":
            image.thumbnail((image_size, image_size), resample)
        elif resize_mode != "none":
            raise ValueError(f"Unknown resize_mode: {resize_mode}")
    return image


def row_images(row: Mapping[str, Any], image_size: int, resize_mode: str, max_images_per_sample: int):
    paths = get_image_paths(row)
    if max_images_per_sample > 0:
        paths = paths[:max_images_per_sample]
    return [load_image(path, image_size, resize_mode) for path in paths]


def confidence_instruction(task: str, choices: Sequence[Any]) -> str:
    """Ask for a ranked signal without disturbing the plain-text answer.

    AUC and COCO AP both need a *continuous* score, which free text does not
    carry. Rather than switch to constrained decoding (which would change what
    the generative baseline is), the prompt asks for the scores on a separate
    line. The first line stays the ordinary answer, so every existing metric
    keeps working; only the new confidence-based ones read the extra line.
    """
    if task in ("disease_diagnosis_classification", "multi_label_classification") and choices:
        options = ", ".join(f'"{normalize_text(c)}": <probability 0-1>' for c in choices)
        return ("Then, on a new line, write your probability for EVERY candidate as "
                f'PROBS: {{{options}}}')
    if task == "detection":
        return ("Give every box five numbers: [x1, y1, x2, y2, confidence], where confidence is "
                "your 0-1 certainty for that box.")
    return 'Then, on a new line, write "CONFIDENCE: <a number between 0 and 1>".'


def build_prompt(row: Mapping[str, Any], elicit_confidence: bool = False) -> str:
    task = normalize_task_name(row.get("task_type", ""))
    prompt = as_text(row.get("prompt") or row.get("question") or "")
    choices = maybe_json_load(row.get("choices"))
    if not isinstance(choices, list):
        choices = []
    parts = [TASK_INSTRUCTIONS.get(task, "Answer the medical imaging question using the provided image.")]
    if prompt:
        parts.append(prompt)
    if choices and "options:" not in prompt.lower():
        parts.append("Options: " + "; ".join(str(choice) for choice in choices))
    if elicit_confidence:
        parts.append(confidence_instruction(task, choices))
    return "\n\n".join(parts)


def make_generation_messages(num_images: int, prompt: str, system_prompt: str) -> list[dict[str, Any]]:
    content = [{"type": "image"} for _ in range(num_images)]
    content.append({"type": "text", "text": prompt})
    return [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}, {"role": "user", "content": content}]


def load_model_and_processor(config: ExpConfig, kwargs: Mapping[str, Any]):
    missing = []
    base_model = should_infer_base_model(kwargs)
    adapter_path = resolve_adapter_path(config, kwargs)
    if torch is None:
        missing.append("torch")
    if AutoModelForImageTextToText is None or AutoProcessor is None or BitsAndBytesConfig is None:
        missing.append("transformers")
    if not base_model and adapter_path and PeftModel is None:
        missing.append("peft")
    if missing:
        raise RuntimeError("Missing inference dependencies: " + ", ".join(sorted(set(missing))))

    device = str(kwargs.get("device", "auto"))
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Pass device=cpu for a slow CPU-only dry run.")
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else (torch.float16 if device == "cuda" else torch.float32)
    load_in_4bit = bool(kwargs.get("load_in_4bit", device == "cuda"))
    quant_config = None
    if load_in_4bit and device == "cuda":
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)

    model_name = str(kwargs.get("model_name_or_path", MODEL_ID))
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        quantization_config=quant_config,
        trust_remote_code=True,
    )
    if adapter_path:
        if PeftModel is None:
            raise RuntimeError("peft is required to infer with a fine-tuned adapter. Pass --base_model to infer without it.")
        model = PeftModel.from_pretrained(model, adapter_path)
    if device == "cpu":
        model.to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(adapter_path or model_name, trust_remote_code=True, use_fast=True)
    processor.tokenizer.padding_side = "left" if int(kwargs.get("batch_size", 1)) > 1 else "right"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    return model, processor, device


def should_infer_base_model(kwargs: Mapping[str, Any]) -> bool:
    for key in ("base_model", "base_model_only", "infer_base_model", "evaluate_base_model", "no_adapter"):
        if parse_bool(kwargs.get(key, False)):
            return True
    return False


def resolve_adapter_path(config: ExpConfig, kwargs: Mapping[str, Any]) -> str | None:
    if should_infer_base_model(kwargs):
        return None
    adapter_path = kwargs.get("adapter_path")
    if adapter_path:
        return str(adapter_path)
    candidate = Path(kwargs.get("model_output_dir") or Path(config.output_dir) / f"{config.experiment_name}-medgemma15-lora") / "final"
    return str(candidate) if candidate.exists() else None


def selected_model_variant(config: ExpConfig, kwargs: Mapping[str, Any]) -> str:
    if should_infer_base_model(kwargs):
        return "base"
    return "adapter" if resolve_adapter_path(config, kwargs) else "base"


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def generate_predictions(
    config: ExpConfig,
    rows: Sequence[dict[str, Any]],
    console: Console,
    kwargs: dict[str, Any],
    model_bundle: tuple[Any, Any, str] | None = None,
    split: str = "inference",
    wandb_run: Any | None = None,
) -> list[dict[str, Any]]:
    model, processor, device = model_bundle or load_model_and_processor(config, kwargs)
    image_size = int(kwargs.get("image_size", 896))
    resize_mode = str(kwargs.get("resize_mode", "square"))
    max_images_per_sample = int(kwargs.get("max_images_per_sample", 1))
    batch_size = max(1, int(kwargs.get("batch_size", 1)))
    system_prompt = str(kwargs.get("system_prompt", "You are an expert medical imaging assistant."))
    elicit_confidence = bool(kwargs.get("elicit_confidence", False))
    predictions = []
    log_every = max(1, int(kwargs.get("log_every", 25)))
    next_log_at = log_every

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]
        texts = []
        batch_images = []
        for row in batch_rows:
            images = row_images(row, image_size, resize_mode, max_images_per_sample)
            messages = make_generation_messages(len(images), build_prompt(row, elicit_confidence), system_prompt)
            if hasattr(processor, "apply_chat_template"):
                text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            else:
                text = processor.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            texts.append(text)
            batch_images.append(images)
        inputs = processor(text=texts, images=batch_images, return_tensors="pt", padding=True)
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        generation_kwargs = {
            "max_new_tokens": int(kwargs.get("max_new_tokens", 256)),
            "do_sample": float(kwargs.get("temperature", 0.0)) > 0,
            "top_p": float(kwargs.get("top_p", 1.0)),
            "pad_token_id": processor.tokenizer.pad_token_id,
        }
        if generation_kwargs["do_sample"]:
            generation_kwargs["temperature"] = float(kwargs.get("temperature", 0.0))
        with torch.no_grad():
            generated = model.generate(**inputs, **generation_kwargs)
        prompt_len = inputs["input_ids"].shape[1]
        decoded = processor.tokenizer.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)
        for row, prediction in zip(batch_rows, decoded):
            answer, scores, confidence = split_confidence(prediction.strip())
            record = {"uid": row["uid"], "prediction": answer,
                      "task_type": row.get("task_type", ""), "case_id": row.get("case_id", "")}
            if scores:
                record["scores"] = scores
            if confidence is not None:
                record["confidence"] = confidence
            predictions.append(record)
        done = min(start + len(batch_rows), len(rows))
        if done == len(rows) or done >= next_log_at:
            console.print(f"Generated {done}/{len(rows)} predictions")
            if wandb_run is not None:
                wandb_run.log({
                    "infer/progress": done / len(rows),
                    "infer/generated_predictions": done,
                    f"infer/{split}/progress": done / len(rows),
                    f"infer/{split}/generated_predictions": done,
                })
            while next_log_at <= done:
                next_log_at += log_every
    return predictions
