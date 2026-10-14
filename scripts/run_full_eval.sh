#!/usr/bin/env bash
# After both training runs finish: score every checkpoint, then run the
# decision-space probes and the logit-confidence baseline.
#
# Every step is resumable. `sweep_medjev.py` and `sweep_sft.py` both skip a
# checkpoint whose `scores.json` already exists, so re-running this after a
# partial sweep only pays for the new checkpoints.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"

MEDJEV_FINAL=$(ls -d output/medjev-lora/checkpoint-* | sed 's#.*/checkpoint-##' | sort -n | tail -1)
SFT_FINAL=$(ls -d output/sft-lora/checkpoint-* | sed 's#.*/checkpoint-##' | sort -n | tail -1)
echo "MedJev final = $MEDJEV_FINAL ; SFT final = $SFT_FINAL"

echo "=== MedJev: sweep (cached) ==="
python3 -u scripts/sweep_medjev.py --checkpoints output/medjev-lora \
    --gpus 0,0 --workdir datas/eval/runs-medjev

echo "=== SFT: sweep (cached) ==="
python3 -u scripts/sweep_sft.py --checkpoints output/sft-lora \
    --gpus 0,0 --workdir datas/eval/runs-sft

echo "=== MedJev: permute probes across training ==="
for STEP in 404 1616 "$MEDJEV_FINAL"; do
    CKPT="output/medjev-lora/checkpoint-$STEP"
    DEC="datas/eval/runs-medjev/checkpoint-$STEP/testing_decisions.jsonl"
    [ -d "$CKPT" ] && [ -f "$DEC" ] || continue
    [ -f "datas/eval/probes-medjev/checkpoint-$STEP/summary.json" ] && continue
    python3 -u scripts/probe_medjev.py --checkpoint "$CKPT" --identity-decisions "$DEC" \
        --out-dir "datas/eval/probes-medjev/checkpoint-$STEP"
done

echo "=== SFT: distribution read from logits ==="
if [ -f "datas/eval/runs-sft/checkpoint-$SFT_FINAL/logit_metrics.json" ]; then
    echo "already computed, skipping"
else
python3 -u scripts/eval_sft_logit_conf.py \
    --adapter "output/sft-lora/checkpoint-$SFT_FINAL" \
    --ground-truth data/splits/testing_sft.jsonl \
    --output "datas/eval/runs-sft/checkpoint-$SFT_FINAL/logit_metrics.json"
fi

echo "=== ALL DONE ==="
