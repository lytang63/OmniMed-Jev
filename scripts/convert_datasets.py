#!/usr/bin/env python3
"""Convert the source datasets into the layout `omnimedjev/engine/preprocess.py` expects.

The converter writes, per split:

    <out>/<split>/<modality>/<dataset>/
        <name>_questions_<split>.json      the questions, with `ImageName` RELATIVE
        images/                            the decoded PNGs

`ImageName` stays relative to its own dataset directory, which is what makes the
tree relocatable -- nothing downstream stores an absolute path.

Output tree (datas/converted/omnimed-suite/):

    MedMNIST+ 224 (12 subsets)   -> classification / multi-label / ordinal regression
    ChestX-Det                   -> multi-label (box annotations are carried but unused)
    TXL-PBC                      -> cell counting
    CAMUS                        -> regression on ejection fraction
    IU X-Ray                     -> report generation (outside the paper's scope)

Usage:
    python scripts/convert_datasets.py medmnist chestx camus txlpbc
    python scripts/convert_datasets.py --list
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "datas" / "raw"
EXT = ROOT / "datas" / "extracted"
OUT = ROOT / "datas" / "converted" / "omnimed-suite"
MANIFESTS = ROOT / "datas" / "converted" / "manifests"

SPLIT_DIR = {"train": "training", "val": "validation_public", "test": "testing"}

# What the model should SAY when the ground truth has no findings.
#
# WHY not the empty string (the obvious first choice, and what this converter
# originally emitted):
#
#   1. An empty generation is indistinguishable from a generation failure — a
#      truncated response, a stop-token misfire, a loader bug. Scoring cannot
#      tell "the model correctly reported nothing" from "the model produced
#      nothing at all", and neither can a human reading the output.
#   2. It gives the model a degenerate optimum. If "" is the correct target for
#      the majority of multi-label rows (ChestMNIST is 53.9% all-zero), always
#      emitting "" is a low-loss strategy. Measured: an early adapter returned
#      "" for every regression, multi-label and detection sample.
#   3. The engine already has a convention for this. `EMPTY_LABELS` in
#      evaluate.py contains {"", "none", "no finding", "no findings", "n/a",
#      "na", "null", "[]"}, so `parse_multilabel("none")` and
#      `parse_boxes("none")` both correctly yield the empty set.
#
# WHY "none" and specifically NOT "normal": the source annotations do not mean
# normal. NIH's own FAQ on ChestX-ray14 states that an all-zero label vector
# means none of the 14 listed patterns were found, and explicitly warns that
# "'No finding' is not equal to 'normal'" — the image may contain pathologies
# outside the listed 14, or uncertain findings within them, because the labels
# are NLP-mined from reports and "hard to exclude" language is labelled
# No-finding. ChestX-Det is the same shape: three radiologists found none of
# the 13 categories, which is not a statement of health.
#
# "none" is the shortest phrasing that is both in EMPTY_LABELS and does not
# assert more than the annotation supports.
NONE_ANSWER = "none"

# --------------------------------------------------------------------------- #
# image writing
# --------------------------------------------------------------------------- #


def _save_one(job: tuple[np.ndarray, str]) -> int:
    arr, dest = job
    path = Path(dest)
    if path.exists() and path.stat().st_size > 0:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(arr)
    if image.mode not in ("L", "RGB"):
        image = image.convert("RGB")
    image.save(path, "PNG", optimize=False, compress_level=1)
    return 1


def write_images(jobs: list[tuple[np.ndarray, str]], workers: int = 16) -> int:
    """Materialise images with a process pool.

    Single-threaded PNG writes on this filesystem run at ~51 files/s (the cost
    is metadata latency, not encoding); 16 workers reach ~300 files/s. 32
    workers is *slower* than 16 because of filesystem contention.
    """
    if not jobs:
        return 0
    if workers <= 1:
        return sum(_save_one(j) for j in jobs)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return sum(pool.map(_save_one, jobs, chunksize=32))


def write_split_jsons(out_dir: Path, split: str, suffix: str, dataset: str, records: list[dict]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{dataset}_questions_{suffix}.json"
    path.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def report(tag: str, path: Path, records: list[dict], n_images: int) -> None:
    print(f"  [{tag}] {len(records):>7} records, {n_images:>7} images -> {path.relative_to(ROOT)}")


# --------------------------------------------------------------------------- #
# MedMNIST+ 224
# --------------------------------------------------------------------------- #

MEDMNIST_SPEC: dict[str, tuple[str, str]] = {
    # subset: (modality, kind)
    "pathmnist": ("Pathology", "classification"),
    "chestmnist": ("Radiology", "multilabel"),
    "dermamnist": ("Dermatology", "classification"),
    "octmnist": ("Ophthalmology", "classification"),
    "pneumoniamnist": ("Radiology", "classification"),
    "retinamnist": ("Ophthalmology", "ordinal"),
    "breastmnist": ("Ultrasound", "classification"),
    "bloodmnist": ("Hematology", "classification"),
    "tissuemnist": ("Histology", "classification"),
    "organamnist": ("CT", "classification"),
    "organcmnist": ("CT", "classification"),
    "organsmnist": ("CT", "classification"),
}

# Verbatim from medmnist/info.py upstream — do not paraphrase, these strings are
# what the model must reproduce and what normalize_text compares.
MEDMNIST_LABELS: dict[str, dict[int, str]] = {
    "pathmnist": {0: "adipose", 1: "background", 2: "debris", 3: "lymphocytes", 4: "mucus",
                  5: "smooth muscle", 6: "normal colon mucosa", 7: "cancer-associated stroma",
                  8: "colorectal adenocarcinoma epithelium"},
    "chestmnist": {0: "atelectasis", 1: "cardiomegaly", 2: "effusion", 3: "infiltration",
                   4: "mass", 5: "nodule", 6: "pneumonia", 7: "pneumothorax",
                   8: "consolidation", 9: "edema", 10: "emphysema", 11: "fibrosis",
                   12: "pleural", 13: "hernia"},
    "dermamnist": {0: "actinic keratoses and intraepithelial carcinoma",
                   1: "basal cell carcinoma", 2: "benign keratosis-like lesions",
                   3: "dermatofibroma", 4: "melanoma", 5: "melanocytic nevi",
                   6: "vascular lesions"},
    "octmnist": {0: "choroidal neovascularization", 1: "diabetic macular edema",
                 2: "drusen", 3: "normal"},
    "pneumoniamnist": {0: "normal", 1: "pneumonia"},
    "retinamnist": {0: "0", 1: "1", 2: "2", 3: "3", 4: "4"},
    "breastmnist": {0: "malignant", 1: "normal, benign"},
    "bloodmnist": {0: "basophil", 1: "eosinophil", 2: "erythroblast",
                   3: "immature granulocytes(myelocytes, metamyelocytes and promyelocytes)",
                   4: "lymphocyte", 5: "monocyte", 6: "neutrophil", 7: "platelet"},
    "tissuemnist": {0: "Collecting Duct, Connecting Tubule", 1: "Distal Convoluted Tubule",
                    2: "Glomerular endothelial cells", 3: "Interstitial endothelial cells",
                    4: "Leukocytes", 5: "Podocytes", 6: "Proximal Tubule Segments",
                    7: "Thick Ascending Limb"},
    "organamnist": {0: "bladder", 1: "femur-left", 2: "femur-right", 3: "heart",
                    4: "kidney-left", 5: "kidney-right", 6: "liver", 7: "lung-left",
                    8: "lung-right", 9: "pancreas", 10: "spleen"},
    "organcmnist": {0: "bladder", 1: "femur-left", 2: "femur-right", 3: "heart",
                    4: "kidney-left", 5: "kidney-right", 6: "liver", 7: "lung-left",
                    8: "lung-right", 9: "pancreas", 10: "spleen"},
    "organsmnist": {0: "bladder", 1: "femur-left", 2: "femur-right", 3: "heart",
                    4: "kidney-left", 5: "kidney-right", 6: "liver", 7: "lung-left",
                    8: "lung-right", 9: "pancreas", 10: "spleen"},
}


def _medmnist_record(base: dict, kind: str, target: Any, labels: dict[int, str]) -> dict:
    """Build one record from a raw MedMNIST target."""
    if kind == "multilabel":
        present = [labels[j] for j, flag in enumerate(np.asarray(target).ravel()) if flag]
        return {
            **base,
            "TaskType": "multi_label_classification",
            "Question": "Which of the listed findings are present? Answer with the findings separated by semicolons, or the single word \"none\" if none are present.",
            "Choices": [labels[k] for k in sorted(labels)],
            # "none" rather than "" — see NO_EMPTY_ANSWER below. The engine's
            # EMPTY_LABELS already treats "none"/"no findings" as an empty set.
            "Answer": present if present else NONE_ANSWER,
        }
    if kind == "ordinal":
        return {
            **base,
            "TaskType": "regression",
            "Question": "What is the diabetic retinopathy severity grade on the 0-4 international clinical scale? Answer with the integer only.",
            "Answer": int(np.asarray(target).ravel()[0]),
        }
    cls = int(np.asarray(target).ravel()[0])
    return {
        **base,
        "TaskType": "disease_diagnosis_classification",
        "Question": "Which class does this image belong to?",
        "Choices": [labels[k] for k in sorted(labels)],
        "Answer": labels[cls],
    }


def _iter_npz(subset: str):
    """Yield (split, index, ndarray_image, target) from the Zenodo .npz."""
    data = np.load(RAW / "medmnistplus" / f"{subset}_224.npz")
    for mn_split, split in (("train", "train"), ("val", "val"), ("test", "test")):
        if f"{mn_split}_images" not in data:
            continue
        for i, (arr, target) in enumerate(zip(data[f"{mn_split}_images"], data[f"{mn_split}_labels"])):
            yield split, i, arr, target


def _iter_parquet(subset: str):
    """Yield (split, index, ndarray_image, target) from the ModelScope mirror.

    The mirror stores HuggingFace parquet with an `image` column (WebP bytes
    decoded by PIL) and a `label` column, split across train/validation/test
    shards. Same records as the .npz, different container.
    """
    from datasets import Dataset

    base = RAW / "medmnistplus_ms" / f"{subset}_224" / "data"
    for shard in sorted(base.glob("*.parquet")):
        split = shard.name.split("-", 1)[0]
        if split == "validation":
            split = "val"
        if split not in ("train", "val", "test"):
            continue
        dataset = Dataset.from_parquet(str(shard))
        for i, row in enumerate(dataset):
            image = np.asarray(row["image"].convert("RGB"))
            yield split, i, image, row["label"]


def _npz_is_readable(path: Path) -> bool:
    """A truncated .npz from a failed download is not a zip file at all.

    Checking only for existence means one leftover corrupt file aborts the whole
    conversion, so validate by actually opening the central directory.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            return archive.testzip() is None
    except Exception:
        return False


def convert_medmnist(workers: int) -> None:
    print("=== MedMNIST+ 224 ===")
    for subset, (modality, kind) in MEDMNIST_SPEC.items():
        npz_path = RAW / "medmnistplus" / f"{subset}_224.npz"
        has_npz = npz_path.exists() and _npz_is_readable(npz_path)
        has_parquet = (RAW / "medmnistplus_ms" / f"{subset}_224" / "data").is_dir()
        if npz_path.exists() and not has_npz:
            print(f"  [warn] {subset}: {npz_path.name} is present but unreadable (truncated download); "
                  f"{'falling back to the ModelScope parquet' if has_parquet else 'nothing to fall back on'}")
        if not has_npz and not has_parquet:
            print(f"  [skip] {subset}: neither a readable Zenodo .npz nor the ModelScope parquet is present")
            continue
        source = "npz" if has_npz else "parquet"
        labels = MEDMNIST_LABELS[subset]

        # Shards restart their per-split index at 0, so index by (split, running
        # counter) instead of the shard-local index to keep uids unique.
        counters: dict[str, int] = {}
        staged: dict[str, list[tuple[dict, tuple[np.ndarray, str]]]] = {}
        for split, _shard_index, arr, target in (_iter_npz(subset) if has_npz else _iter_parquet(subset)):
            n = counters.get(split, 0)
            counters[split] = n + 1
            out_dir = OUT / SPLIT_DIR[split] / modality / subset
            name = f"{subset}_{split}_{n:06d}.png"
            base = {"ID": f"{subset}_{split}_{n:06d}", "ImageName": f"images/{name}"}
            record = _medmnist_record(base, kind, target, labels)
            staged.setdefault(split, []).append((record, (arr, str(out_dir / "images" / name))))

        for split, pairs in staged.items():
            out_dir = OUT / SPLIT_DIR[split] / modality / subset
            records = [r for r, _ in pairs]
            n = write_images([job for _, job in pairs], workers)
            path = write_split_jsons(out_dir, split, split, subset, records)
            report(f"{subset}/{split} [{source}]", path, records, n)


# --------------------------------------------------------------------------- #
# ChestX-Det
# --------------------------------------------------------------------------- #

CHESTX_CLASSES = ["Atelectasis", "Calcification", "Cardiomegaly", "Consolidation",
                  "Diffuse Nodule", "Effusion", "Emphysema", "Fibrosis", "Fracture",
                  "Mass", "Nodule", "Pleural Thickening", "Pneumothorax"]


def convert_chestx(workers: int) -> None:
    print("=== ChestX-Det ===")
    src_dir = EXT / "chestx-det"
    ann_dir = RAW / "chestx-det"
    if not src_dir.exists():
        print(f"  [skip] extract {src_dir} first")
        return

    # The official release has only train/test. Carve a fixed 10% of official
    # train out as the public validation split so every baseline shares one
    # manifest. The rule is a deterministic crc32 hash of the file name rather
    # than a seeded shuffle, so it stays stable across Python versions.
    for official in ("train", "test"):
        ann = json.loads((ann_dir / f"ChestX_Det_{official}.json").read_text())
        if official == "train":
            val_names = {r["file_name"] for i, r in enumerate(ann)
                         if zlib.crc32(r["file_name"].encode()) % 10 == 0}
            subsets = {"train": [r for r in ann if r["file_name"] not in val_names],
                       "val":   [r for r in ann if r["file_name"] in val_names]}
        else:
            subsets = {"test": ann}

        for split, rows in subsets.items():
            out_dir = OUT / SPLIT_DIR[split] / "Radiology" / "ChestX-Det"
            records, jobs = [], []
            for r in rows:
                stem = Path(r["file_name"]).stem
                img_src = src_dir / official / r["file_name"]
                if not img_src.exists():
                    continue
                dest = out_dir / "images" / f"{stem}.png"
                arr = np.asarray(Image.open(img_src).convert("RGB"))
                jobs.append((arr, str(dest)))
                rel = f"images/{stem}.png"

                # (a) multi-label classification over the 13 findings
                records.append({
                    "ID": f"chestxdet_{split}_{stem}_labels",
                    "ImageName": rel,
                    "TaskType": "multi_label_classification",
                    "Question": "Which thoracic abnormalities are present? Answer with the findings separated by semicolons, or the single word \"none\" if none are present.",
                    "Choices": CHESTX_CLASSES,
                    "Answer": list(r["syms"]) if r["syms"] else NONE_ANSWER,
                })

                # (b) class-aware detection; see the module docstring for the shape.
                boxes_by_class: dict[str, list[list[float]]] = {}
                for label, box in zip(r["syms"], r["boxes"]):
                    boxes_by_class.setdefault(label, []).append([round(float(v), 1) for v in box])
                records.append({
                    "ID": f"chestxdet_{split}_{stem}_boxes",
                    "ImageName": rel,
                    "TaskType": "detection",
                    "Question": ("Return one bounding box per finding as a JSON object mapping the "
                                 "finding name to a list of [x1, y1, x2, y2] boxes, e.g. "
                                 '{"Nodule": [[10, 20, 60, 80]]}. Answer with the single word '
                                 '"none" if nothing is found. '
                                 "Class list: " + "; ".join(CHESTX_CLASSES)),
                    "Answer": boxes_by_class if boxes_by_class else NONE_ANSWER,
                })

            n = write_images(jobs, workers)
            suffix = {"train": "train", "val": "val", "test": "test"}[split]
            path = write_split_jsons(out_dir, split, suffix, "ChestX-Det", records)
            report(f"ChestX-Det/{split}", path, records, n)

    # Freeze the exact validation manifest next to the converted data.
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    ann = json.loads((ann_dir / "ChestX_Det_train.json").read_text())
    val = sorted(r["file_name"] for r in ann if zlib.crc32(r["file_name"].encode()) % 10 == 0)
    mpath = MANIFESTS / "chestx-det_validation_manifest.json"
    mpath.write_text(json.dumps({"seed_rule": "crc32(file_name) % 10 == 0",
                                 "source": "ChestX_Det_train.json",
                                 "num_val": len(val), "files": val}, indent=1))
    print(f"  [manifest] {mpath.relative_to(ROOT)}  ({len(val)} val files)")


# --------------------------------------------------------------------------- #
# NeurIPS22 CellSeg -> counting
# --------------------------------------------------------------------------- #


def convert_txlpbc(workers: int) -> None:
    """TXL-PBC (KeenForgeAI/TXL-PBC-corrected) -> counting + detection.

    Replaces NeurIPS22 CellSeg. Chosen because its YOLO labels make the counting
    target unambiguous — the cell count is just the number of label lines, no
    instance-mask thresholding or pseudo-labelling involved — and it yields a
    detection task from the same annotations for free.

    Layout: images/{train,val,test}/*.png + labels/{train,val,test}/*.txt, with
    labels in YOLO form `<class> <cx> <cy> <w> <h>`, all four normalised to
    [0,1]. Class ids: 0=WBC, 1=RBC, 2=Platelets.
    """
    print("=== TXL-PBC (cell counting + detection) ===")
    src = RAW / "txl-pbc"
    if not (src / "images").is_dir():
        print(f"  [skip] {src}/images not found")
        return

    classes = ["WBC", "RBC", "Platelets"]
    for split in ("train", "val", "test"):
        img_dir, lbl_dir = src / "images" / split, src / "labels" / split
        if not img_dir.is_dir():
            print(f"  [skip] {split}")
            continue
        out_dir = OUT / SPLIT_DIR[split] / "Hematology" / "TXL-PBC"
        records, jobs = [], []
        for img_path in sorted(img_dir.glob("*.png")):
            label_path = lbl_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
            width = height = None
            with Image.open(img_path) as probe:
                width, height = probe.size
                arr = np.asarray(probe.convert("RGB"))
            rel = f"images/{img_path.name}"
            jobs.append((arr, str(out_dir / rel)))

            boxes_by_class: dict[str, list[list[float]]] = {}
            for line in label_path.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls_id, cx, cy, bw, bh = (float(parts[0]), *[float(v) for v in parts[1:5]])
                # YOLO is centre/size normalised; the engine wants absolute xyxy.
                x1 = (cx - bw / 2) * width
                y1 = (cy - bh / 2) * height
                x2 = (cx + bw / 2) * width
                y2 = (cy + bh / 2) * height
                name = classes[int(cls_id)] if 0 <= int(cls_id) < len(classes) else str(int(cls_id))
                boxes_by_class.setdefault(name, []).append(
                    [round(max(0.0, x1), 1), round(max(0.0, y1), 1),
                     round(min(float(width), x2), 1), round(min(float(height), y2), 1)])

            total_cells = sum(len(v) for v in boxes_by_class.values())
            records.append({
                "ID": f"txlpbc_{split}_{img_path.stem}_count",
                "ImageName": rel,
                "TaskType": "cell_counting",
                "Question": "How many cells are present in this blood smear image? Answer with the integer only.",
                "Answer": total_cells,
            })
            records.append({
                "ID": f"txlpbc_{split}_{img_path.stem}_boxes",
                "ImageName": rel,
                "TaskType": "detection",
                "Question": ("Return one bounding box per cell as a JSON object mapping the cell type to a "
                             "list of [x1, y1, x2, y2] boxes in pixels, e.g. {\"RBC\": [[10, 20, 40, 60]]}. "
                             "Answer with the single word \"none\" if nothing is found. "
                             "Cell types: " + "; ".join(classes)),
                "Answer": boxes_by_class if boxes_by_class else NONE_ANSWER,
            })

        n = write_images(jobs, workers)
        path = write_split_jsons(out_dir, split, split, "TXL-PBC", records)
        report(f"TXL-PBC/{split}", path, records, n)


def convert_camus(workers: int) -> None:
    """CAMUS (MedOtter/CAMUS) -> regression on ejection fraction.

    Replaces IUGC 2025. CAMUS ships an official patient-level 400/50/50 split,
    a genuinely continuous target (`ef`, in percent) rather than a quantised
    grade, and the same 2D ultrasound modality IUGC used.

    Only the `ed_es` subset is used: one row per (patient, view, phase) with a
    single ED or ES frame. The `half_sequence` subset is full cardiac cine loops
    (~1.7 GB) and is not needed for a single-image regression target.
    """
    print("=== CAMUS (EF regression) ===")
    src = RAW / "camus"
    if not (src / "ed_es").is_dir():
        print(f"  [skip] {src}/ed_es not found")
        return

    from datasets import Dataset

    for shard in sorted((src / "ed_es").glob("*.parquet")):
        split = shard.name.split("-", 1)[0]
        if split not in ("train", "validation", "test"):
            continue
        mapped = "val" if split == "validation" else split
        out_dir = OUT / SPLIT_DIR[mapped] / "Cardiology" / "CAMUS"
        dataset = Dataset.from_parquet(str(shard))
        records, jobs = [], []
        for row in dataset:
            image = row["image"]
            if row.get("ef") is None:
                continue
            uid = f"{row['patient_id']}_{row['view']}_{row['phase']}"
            rel = f"images/{uid}.png"
            jobs.append((np.asarray(image.convert("RGB")), str(out_dir / rel)))
            records.append({
                "ID": f"camus_{mapped}_{uid}",
                # view/phase are part of the question because EF is measured on a
                # specific frame of a specific view; hiding them would make the
                # target ambiguous.
                "ImageName": rel,
                "TaskType": "regression",
                "Question": (f"This is a {row['view']} echocardiographic view at end-"
                             f"{'diastole' if row['phase'] == 'ED' else 'systole'}. "
                             "What is the left ventricular ejection fraction (EF) in percent? "
                             "Answer with the number only."),
                "Answer": float(row["ef"]),
            })
        n = write_images(jobs, workers)
        path = write_split_jsons(out_dir, mapped, mapped, "CAMUS", records)
        report(f"CAMUS/{mapped}", path, records, n)


# --------------------------------------------------------------------------- #
# IU X-Ray -> report generation
# --------------------------------------------------------------------------- #


def convert_iuxray(workers: int) -> None:
    print("=== IU X-Ray ===")
    src = EXT / "iu-xray" / "IU-Xray"
    reports_dir, images_dir = src / "raw" / "reports", src / "raw" / "images"
    if not reports_dir.exists():
        print(f"  [skip] extract {src} first")
        return

    import xml.etree.ElementTree as ET

    def split_for(patient: str) -> str:
        """Patient-disjoint 70/10/20 split.

        The original IU X-Ray release ships no split at all. The report-generation
        literature universally uses R2Gen's 2069/296/590, but that is a *fixed
        list of case ids*, not reproducible from any rule. We reproduce the
        proportions instead, keyed on the patient (uId) so no patient straddles
        a boundary, using crc32 for cross-version determinism.

        Consequence, and it must be stated in any write-up: numbers produced on
        this split are internally comparable but are NOT comparable to
        R2Gen-published numbers. Swapping in R2Gen's annotation.json is a
        one-line change if strict literature alignment is required.
        """
        bucket = zlib.crc32(patient.encode()) % 10
        return "val" if bucket == 0 else ("test" if bucket in (1, 2) else "train")

    buckets: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    images_by_split: dict[str, list[tuple[str, str]]] = {"train": [], "val": [], "test": []}

    for xml_path in sorted(reports_dir.glob("*.xml")):
        root = ET.parse(xml_path).getroot()
        patient = next((u.attrib["id"] for u in root.findall(".//uId")
                        if u.attrib.get("id", "").startswith("CXR")), None)
        if not patient:
            continue
        texts = {a.attrib.get("Label"): (a.text or "").strip()
                 for a in root.findall(".//AbstractText")}
        findings, impression = texts.get("FINDINGS", ""), texts.get("IMPRESSION", "")
        if not findings and not impression:
            continue
        # NOTE: must not be called `report` — the module already defines a
        # report() helper, and a local of that name shadows it.
        report_text = "\n".join(part for part in
                                (f"FINDINGS: {findings}" if findings else "",
                                 f"IMPRESSION: {impression}" if impression else "") if part)

        image_ids = [p.attrib["id"] for p in root.findall(".//parentImage") if p.attrib.get("id")]
        image_ids = [i for i in image_ids if (images_dir / f"{i}.png").exists()]
        if not image_ids:
            continue

        split = split_for(patient)
        rels = [f"images/{image_id}.png" for image_id in image_ids]
        for src_path, rel in zip(image_ids, rels):
            images_by_split[split].append(
                (str(images_dir / f"{src_path}.png"),
                 str(OUT / SPLIT_DIR[split] / "Radiology" / "IU-Xray" / rel)))
        # ONE record per study, listing every view. Emitting a record per view
        # would repeat the same reference text N times and inflate the reported
        # sample count for multi-view studies, of which there are many
        # (1.89 images/study on average).
        buckets[split].append({
            "ID": f"iuxray_{patient}",
            "ImageName": rels if len(rels) > 1 else rels[0],
            "TaskType": "report_generation",
            "Question": "Generate the radiology report (Findings and Impression) for this chest radiograph.",
            "Answer": report_text,
        })

    total_images = 0
    for split, split_records in buckets.items():
        if not split_records:
            continue
        out_dir = OUT / SPLIT_DIR[split] / "Radiology" / "IU-Xray"
        jobs = [(np.asarray(Image.open(s).convert("RGB")), d)
                for s, d in images_by_split[split]]
        n = write_images(jobs, workers)
        total_images += n
        suffix = {"train": "train", "val": "val", "test": "test"}[split]
        path = write_split_jsons(out_dir, split, suffix, "IU-Xray", split_records)
        report(f"IU-Xray/{split}", path, split_records, n)
    print(f"  total images materialised: {total_images}")


def convert_iugc(workers: int) -> None:
    print("=== IUGC 2025 ===")
    src = EXT / "iugc2025"
    if not src.exists():
        print(f"  [skip] extract {src} first")
        return
    print("  (structure inspection required)")
    for p in sorted(src.rglob("*"))[:20]:
        print("   ", p.relative_to(src))


CONVERTERS = {
    "medmnist": convert_medmnist,
    "chestx": convert_chestx,
    "txlpbc": convert_txlpbc,
    "camus": convert_camus,
    "iuxray": convert_iuxray,
}


# What each converter needs on disk before it can do anything. Checked up front
# so that a requested converter whose download is missing fails loudly: each
# converter skips its missing input and returns normally, which otherwise
# reports success after converting nothing and leaves a silently incomplete
# suite.
REQUIRED_INPUT = {
    "medmnist": (RAW / "medmnistplus", RAW / "medmnistplus_ms"),
    "chestx": (EXT / "chestx-det",),
    "txlpbc": (RAW / "txl-pbc" / "images",),
    "camus": (RAW / "camus" / "ed_es",),
    "iuxray": (EXT / "iu-xray" / "IU-Xray",),
    "iugc": (EXT / "iugc2025",),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="*", choices=sorted(CONVERTERS), help="which converters to run")
    ap.add_argument("--list", action="store_true", help="list converters and exit")
    ap.add_argument("--workers", type=int, default=16, help="image-writer processes (default 16; 32 is slower)")
    args = ap.parse_args()
    if args.list or not args.datasets:
        for name in sorted(CONVERTERS):
            print(f"  {name}")
        return
    missing = [name for name in args.datasets
               if not any(p.exists() for p in REQUIRED_INPUT.get(name, ()))]
    if missing:
        raise SystemExit(
            "no input found for: " + ", ".join(missing)
            + "\n  Download and unpack them first, e.g."
            + "\n    ./scripts/download_datasets.sh " + " ".join(missing)
            + f"\n  Expected under {RAW} / {EXT}."
        )
    for name in args.datasets:
        CONVERTERS[name](args.workers)


if __name__ == "__main__":
    main()
