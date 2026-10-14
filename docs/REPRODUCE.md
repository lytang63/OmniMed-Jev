# Reproducing the results

Everything below is the sequence actually run to produce the paper. The two
arms differ in exactly one configuration item (`max_length`) and in nothing
else; the loader, the four task families and the test rows are shared.

## 0. Environment

```bash
git clone https://github.com/lytang63/OmniMed-Jev && cd OmniMed-Jev
conda create -n omnimedjev python=3.12 -y && conda activate omnimedjev

# torch: install the CUDA build matching your driver, then
pip install -e .

# optional, and worth it: flash-attn as a prebuilt wheel
./scripts/install_flash_attn.sh
```

Set `CONDA_ENV=omnimedjev` before sourcing `scripts/env.sh` if you want the
launchers to activate the environment for you; otherwise activate it yourself
and leave `CONDA_ENV` unset. `scripts/env.sh` also redirects `TMPDIR`,
`HF_HOME`, `PIP_CACHE_DIR` and friends under `$HOME/.cache`, because the
checkpoints and the datasets do not fit comfortably on a small system disk.

## 1. Weights and data

Two command-line tools are needed by the downloaders and are **not** installed
by `pip install -e .`:

- `aria2c` — the Zenodo/ModelScope fetches (`apt install aria2` / `brew install aria2`)
- `huggingface-cli` — the two Hugging Face datasets (`pip install huggingface_hub`)

```bash
./scripts/download_models.sh            # -> models/medgemma-1.5-4b-it
./scripts/download_datasets.sh          # -> datas/raw/  (7 groups)

# ChestX-Det and IU X-Ray arrive as zip/tgz archives; unpack them into
# datas/extracted/<name>/ so the converter can read the images.
mkdir -p datas/extracted
for z in datas/raw/chestx-det/*.zip; do unzip -q -o "$z" -d datas/extracted/chestx-det; done
mkdir -p datas/extracted/iu-xray && tar -xzf datas/raw/iu-xray/NLMCXR_png.tgz -C datas/extracted/iu-xray

python scripts/convert_datasets.py medmnist chestx camus txlpbc
```

### Building the splits

`preprocess` is asynchronous, so wait for it before continuing (check with
`./scripts/run_preprocess.sh status`).

```bash
./scripts/run_preprocess.sh start       # -> datas/preprocessed/Preprocessed-omnimed-suite/

# 1. the task-balanced draw over that pool
python scripts/make_train_subset.py \
    --source datas/preprocessed/Preprocessed-omnimed-suite/train.jsonl \
    --out datas/balanced/train.jsonl --per-task 10000 --empty-cap 0.25   # -> 29,842 rows

# 2. keep the released training rows -- the four decision families. The uid list
#    in data/splits/ is the authoritative record of exactly which rows those are.
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
print(f"wrote {kept} rows")             # -> 23,558
PY

# 3. the validation split the generative arm dev-slices from (--per-task 0 copies
#    the rows through unchanged)
python scripts/make_train_subset.py \
    --source datas/preprocessed/Preprocessed-omnimed-suite/validation.jsonl \
    --out datas/sft/Preprocessed-omnimed-suite/validation.jsonl --per-task 0

# 4. the evaluator's tree, which also supplies the 726 held-out test rows
python scripts/make_eval_subset.py --out datas/eval

# 5. the decision arm's states
python scripts/build_medjev_data.py \
    --src datas/sft/Preprocessed-omnimed-suite/train.jsonl       --out datas/medjev/train.jsonl
python scripts/build_medjev_data.py \
    --src datas/preprocessed/Preprocessed-omnimed-suite/validation.jsonl \
    --out datas/medjev/validation.jsonl --per-source 40
python scripts/build_medjev_data.py \
    --src datas/eval/Preprocessed-omnimed-suite/testing.jsonl    --out datas/medjev/testing.jsonl
```

Step 2 is what removes `detection` and `report_generation`: the decision arm
cannot express either, so the released runs drop them, which is why 29,842
answer-level rows become 23,558 states.

`data/README.md` covers the dataset layout and how to join the two arms on
`uid`. The exact partition is shipped in `data/splits/`, so steps 1–5 can be
skipped entirely if all you need is the released train/dev/test sets.

`convert_datasets.py` fails immediately if a requested dataset's input is
missing, so a partially-downloaded suite cannot be silently converted.

### Hardware

The reference machine was a single A800-80GB per arm, no scheduler. Wall clock:
**12.86 h** for the generative arm, **13.95 h** for the decision arm (Stage B
peak memory 17.18 GB). Both are single-GPU runs; the sweep for one arm fits on
one card too, and `--gpus` takes a comma-separated device list.

## 2. Training

The generative arm:

```bash
source scripts/env.sh
CUDA_VISIBLE_DEVICES=0 ./scripts/run_train_sft.sh
```

The decision arm, two stages — the head first on a frozen backbone, then the
head together with LoRA adapters seeded from that head:

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/run_train_medjev.sh
```

### Configuration

| Item | Generative SFT | OmniMed-Jev |
|---|---|---|
| Backbone | `models/medgemma-1.5-4b-it` | same |
| Quantisation | 4-bit NF4, double quantisation, bf16 compute | same |
| LoRA rank / alpha / dropout | 16 / 16 / 0.05 | same |
| LoRA target modules | `all-linear` (including the vision tower) | same |
| Learning rate | 2e-4 | same |
| Schedule / warmup | linear / 111 steps (ratio 0.03) | same |
| Optimiser | `paged_adamw_8bit` | same |
| Weight decay / grad clipping | 0.0 / 0.3 | same |
| Precision | bf16 | same |
| Effective batch | 8 × 4 = 32 | same |
| Epochs / total steps | 5 / 3,685 | 5 / 3,685 |
| Sampling | `group_by_length` | same |
| Seed | 42 | same |
| Gradient checkpointing | on | on (Stage B) |
| Option-order augmentation | `shuffle_options: true` | `shuffle_candidates: true` |
| Checkpoint saves | `save_steps = 404` (10 checkpoints) | same |
| **`max_length`** | **1,024** | **3,072** |

`max_length` is the only configuration difference, and it is a consequence of
the interface rather than a tuning choice: a state packs every decision about
one image into a single sequence, and the longest observed decision block is
2,707 tokens, which does not fit in 1,024.

Stage A trains 1,057,410 parameters (the head) for 200 steps; Stage B adds
39,555,202 of the backbone's 2,529,778,162 parameters, i.e. 1.564%.

Augmentation is on during training and **off** at evaluation time, so the dev
loss stays comparable across steps. On the decision arm only `Choice` and
`Noul` candidates are permuted; `Score` grids are left in place because position
*is* the ordinal index there. On the generative arm only the tasks carrying
choices (classification and multi-label) are permuted — the exact counterpart.

## 3. Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/run_full_eval.sh
```

That scores every checkpoint of both arms, runs the decision-space probes, and
extracts the generative arm's reconstructed distribution from its logits. Every
stage skips work that already has output, so it is safe to re-run.

Running the pieces by hand:

```bash
# one checkpoint: infer -> fold -> score -> probability metrics
python scripts/sweep_medjev.py --checkpoints output/medjev-lora --gpus 0
python scripts/sweep_sft.py    --checkpoints output/sft-lora   --gpus 0

# decision-space probes (permute / restrict / drop-gold / add-none)
python scripts/probe_medjev.py --checkpoint output/medjev-lora/checkpoint-2424 \
    --out-dir datas/eval/probes-medjev/checkpoint-2424

# the generative arm's distribution, read from its own logits
python scripts/eval_sft_logit_conf.py \
    --adapter output/sft-lora/checkpoint-2424 \
    --ground-truth data/splits/testing_sft.jsonl \
    --output datas/eval/runs-sft/checkpoint-2424/logit_metrics.json

# side by side
python scripts/compare_medjev.py \
    --baseline datas/eval/runs-sft/checkpoint-2424/scores.json \
    --medjev    datas/eval/runs-medjev/checkpoint-2424/scores.json \
    --calibration datas/eval/runs-medjev/checkpoint-2424/metrics.json
```

### Recovering a distribution from the generative arm

A generative model emits an answer string, not a distribution, so the comparison
requires a reconstruction. Every reported generative probability comes from
reading the model's own next-token logits: each candidate's continuation is
scored as a whole (summed log-probability, prefixed with a space), and the
scores are normalised across the candidate set. All candidates are scored in one
batched forward pass.

Multi-label has no per-finding likelihood in a semicolon-joined answer, so it is
reformulated as one binary question per finding ("Is `<finding>` present in this
image? Answer Yes or No."), asked once per finding, and the softmax is taken
over the `Yes` / `No` continuations.

Note that this reconstruction is *generous*: nothing in training made those
log-likelihoods sum to one, so it is a pseudo-distribution, whereas the decision
arm's distribution is the training objective itself. Two prompting-based
alternatives were tried first and both failed — the configuration files for
those attempts (`configs/infer-sft-conf.yaml`, `configs/infer-sft-conf2.yaml`)
and the untuned-backbone probe are kept as a record.

### Checkpoint selection

The comparison is made at **step 2424**, the decision arm's best-validation
checkpoint; both arms are evaluated at that same step. The generative arm's
decision-quality numbers exist only at 2424; its task metrics exist at all ten.

## 4. Figures

The numbers plotted in Figures 2–3, and the probe results, are frozen in
`data/figure_data/`, each with the sha256 of the artefact it was read from, so
they can be checked without re-running anything.

## 5. Statistical protocol

- Every difference is computed on the two models' **per-row** predictions,
  because both score the same rows; an unpaired comparison would overstate the
  uncertainty.
- Proportions carry **Wilson intervals**; differences carry **paired bootstrap**
  intervals (20,000 paired resamples of rows, freshly seeded generator with seed
  20260925, quoted to two decimals).
- Classification, multi-label and regression differences are **not**
  distinguishable from zero at 95%. The only task-level difference that is, is
  the counting MAE — and it favours the generative arm.

## 6. Things that are not like-for-like

These are reported rather than hidden, because each one is a respect in which
the two arms differ beyond the interface:

1. **Supervision density.** Factoring the annotations by decision semantics
   turns 23,558 answer-level targets into 155,729 decision-level targets. No new
   labels are collected, but the decision arm sees far more signals per image.
2. **An extra trained component.** The decision arm has a decision head; the
   generative arm has none.
3. **Post-training budget.** The generative baseline received instruction tuning
   after supervised fine-tuning, a strictly larger budget than the decision arm
   received. If that asymmetry matters, it favours the baseline.
4. **`max_length`**, as above, follows from the interface.

Consequently what is measured is decision modelling *as implemented* here, not
the interface in isolation. The ablation that would separate density from
interface — down-sampling the multi-label decisions to the baseline's count and
re-measuring the classification crossover — has not been run.
