# data

The image files and model weights are **not** in this repository. What is here
is the partition itself — enough to rebuild the exact splits from the public
source datasets, and enough to reproduce the reported numbers without training
anything.

```
data/
  splits/
    training_uids.txt                 23,558 uids
    validation_uids.txt                  600 uids
    testing_uids.txt                     726 uids
    testing_states.jsonl                 726 decision states  (2.2 MB)
    validation_states.jsonl              600 decision states  (1.5 MB)
    testing_sft.jsonl                    726 generative-arm rows (1.6 MB)
    chestx-det_validation_manifest.json  the ChestX-Det hold-out rule + 289 files
  figure_data/
    fig2_dynamics.json                    frozen numbers behind Figures 2-3
    fig3_reliability_resolution.json      and the probe results, each with the
    probe_results.json                    sha256 of the artefact it was read from
```

## What the uids are

A uid identifies one question about one image:

```
<source-split>:<source>:<question-file>:<case>:<task_type>:<case>
```

for example

```
testing:organamnist:organamnist_questions_test:organamnist_test_000074:disease_diagnosis_classification:organamnist_test_000074
```

The first field is the split of the **source dataset**, not of this suite: it
says which of the source's own train/validation/test files the row came from.
Join the two arms on this string — `testing_uids.txt` is exactly the set of uids
the decision arm and the generative arm are both scored on.

## The splits

| | states | where they come from | how they were built |
|---|---:|---|---|
| train | 23,558 | every source dataset's own train split | the task-balanced subset (below), restricted to the four decision families |
| validation | 600 | every source dataset's own validation split | 40 states per source, seeded draw — this is the dev set used to pick the comparison checkpoint |
| test | 726 | every source dataset's own test split | held out; the same 726 states for both arms |

The split was **constructed here**, not adopted from a published benchmark, so
neither arm can profit from having been tuned against a widely used test set.
Two consequences worth stating:

- The train file is the balanced subset minus `detection` and
  `report_generation`, which is exactly 29,842 − 3,614 − 2,670 = 23,558 rows.
  One row per image, so 23,558 rows are 23,558 states.
- `validation` is the decision arm's dev set. In the generative arm the dev loss
  is computed on the first `max_eval_samples: 128` rows of the preprocessed
  validation file (`omnimedjev/engine/train.py`, `load_splits`), not on these
  600 states.

## Rebuilding the splits from scratch

The full command sequence is in [`../docs/REPRODUCE.md`](../docs/REPRODUCE.md).
The part that decides *which rows* are the partition is this:

```bash
source scripts/env.sh

./scripts/download_datasets.sh                      # -> datas/raw/
python scripts/convert_datasets.py medmnist chestx camus txlpbc
./scripts/run_preprocess.sh start                   # -> datas/preprocessed/

# the task-balanced draw over that pool
python scripts/make_train_subset.py \
    --source datas/preprocessed/Preprocessed-omnimed-suite/train.jsonl \
    --out datas/balanced/train.jsonl --per-task 10000 --empty-cap 0.25   # -> 29,842 rows

# keep the released training rows: the uids in data/splits/training_uids.txt.
# This is the step that drops detection and report_generation -- 29,842 - 3,614
# - 2,670 = 23,558 -- and joining on `uid` is exact, not approximate.
python - <<'PY'
import json, pathlib
uids = set(pathlib.Path("data/splits/training_uids.txt").read_text().split())
src = pathlib.Path("datas/balanced/train.jsonl")
dst = pathlib.Path("datas/sft/Preprocessed-omnimed-suite/train.jsonl")
dst.parent.mkdir(parents=True, exist_ok=True)
kept = 0
with src.open() as fin, dst.open("w") as fout:
    for line in fin:
        if json.loads(line)["uid"] in uids:
            fout.write(line); kept += 1
print(f"wrote {kept} rows")                          # -> 23,558
PY

# the decision states, and the evaluator tree that carries the 726 test rows
python scripts/build_medjev_data.py \
    --src datas/sft/Preprocessed-omnimed-suite/train.jsonl --out datas/medjev/train.jsonl
python scripts/build_medjev_data.py \
    --src datas/preprocessed/Preprocessed-omnimed-suite/validation.jsonl \
    --out datas/medjev/validation.jsonl --per-source 40
python scripts/make_eval_subset.py --out datas/eval
python scripts/build_medjev_data.py \
    --src datas/eval/Preprocessed-omnimed-suite/testing.jsonl --out datas/medjev/testing.jsonl
```

`build_medjev_data.py` groups rows by image, so a chest radiograph carrying
fourteen annotated findings becomes **one** state with fourteen `Noul`
decisions rather than fourteen rows. Because the four decision families
contribute exactly one row per image, 23,558 rows are 23,558 states.

## Using the shipped states without rebuilding

The states in `data/splits/` are loadable as they are. Their image paths are
relative to the repository root:

```json
"images": ["datas/converted/omnimed-suite/testing/CT/organamnist/images/organamnist_test_000074.png"]
```

So materialise the converted tree at `datas/converted/omnimed-suite/` (step 1–2
above, plus the image decode) and run from the repository root — then

```bash
python scripts/infer_medjev.py \
    --checkpoint output/medjev-lora \
    --data data/splits/testing_states.jsonl \
    --out-dir datas/eval/checkpoint-final
```

works unchanged. If your tree lives somewhere else, rewrite the
`datas/converted/omnimed-suite/` prefix in `images` — that prefix is the only
path component stored in the file.

## A note on the ChestX-Det hold-out

`chestx-det_validation_manifest.json` records how 289 images were carved out of
`ChestX_Det_train.json` as a validation split: `crc32(file_name) % 10 == 0`. It
is shipped because it is the only split decision here that is not simply "use
the source dataset's own split", and it is not recoverable from the other files.

## Licensing of the source data

The splits are derived from public datasets and inherit their terms. Check each
source's licence before redistributing anything built on top of them — MedMNIST+
is CC BY 4.0, CAMUS and TXL-PBC have their own terms, and ChestX-Det is
distributed by its authors. See the dataset table in the top-level README.
