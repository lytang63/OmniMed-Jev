#!/usr/bin/env bash
# The released OmniMed-Jev run: candidate-order augmentation, 5 epochs.
#
# WHY the augmentation: trained without it, the model is NOT
# permutation-invariant -- with the options shuffled, top-1 changed on 31% of
# classification decisions and 95% of counting decisions, because training only
# ever showed one fixed order. This run adds per-draw candidate reordering for
# choice/noul decisions. (`Score` grids are never permuted: position *is* the
# ordinal index there.)
#
# 5 epochs with save_steps=404 places a checkpoint at step 1616, so a single run
# both trains the released model and leaves a checkpoint comparable with the
# generative baseline's epoch budget.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

python3 -u scripts/train_medjev.py --stage head --head-steps 200 \
  --train datas/medjev/train.jsonl --eval datas/medjev/validation.jsonl \
  --output-dir output/medjev-head --shuffle-candidates

python3 -u scripts/train_medjev.py --stage lora --init-head output/medjev-head \
  --train datas/medjev/train.jsonl --eval datas/medjev/validation.jsonl \
  --output-dir output/medjev-lora --save-total-limit 0 \
  --num-epochs 5 --save-steps 404 --shuffle-candidates
