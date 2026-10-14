#!/usr/bin/env bash
# The generative SFT baseline, matched to the OmniMed-Jev run.
#
# Same 23,558 training states, same 5 epochs / 3,685 steps, same option-order
# augmentation; the difference is the interface: this arm emits answer tokens,
# the decision arm emits a probability distribution over the same candidates.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

exec python3 -u -m omnimedjev \
  -n sft -d omnimed-suite --config local \
  --root_dir "$PWD" \
  --input_dir datas/stub-input \
  --output_dir datas/sft \
  --custom_args configs/train-sft.yaml \
  train --num_epochs 5 --learning_rate 2e-4
