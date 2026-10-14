#!/bin/bash
# Download MedGemma weights from the ModelScope mirror into ./models/.
#
# WHY ModelScope and not Hugging Face: google/medgemma-* is `gated: auto` on HF,
# so it needs an account that has accepted Google's Health AI Developer
# Foundations terms plus a token. ModelScope carries the identical official
# repos ungated, and serves them without a proxy.
#
# Usage: ./scripts/download_models.sh [model-name ...]
#   Default: both medgemma-1.5-4b-it and medgemma-4b-it
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${PROJECT_ROOT}/models"
API="https://modelscope.cn/api/v1/models"
REVISION="${REVISION:-master}"

DEFAULT_MODELS=("google/medgemma-1.5-4b-it" "google/medgemma-4b-it")
if [[ $# -gt 0 ]]; then
  MODELS=("$@")
else
  MODELS=("${DEFAULT_MODELS[@]}")
fi

mkdir -p "$MODELS_DIR"

for model in "${MODELS[@]}"; do
  target="${MODELS_DIR}/$(basename "$model")"
  mkdir -p "$target"
  echo "=== ${model} -> ${target} ==="

  # ModelScope's file-list endpoint returns JSON; pull out every file path.
  mapfile -t files < <(
    curl -sS --noproxy '*' --max-time 60 --retry 5 --retry-delay 3 \
      "${API}/${model}/repo/files?Revision=${REVISION}&Recursive=true" |
      python3 -c '
import json, sys
d = json.load(sys.stdin)
for f in d.get("Data", {}).get("Files", []):
    if f.get("Type") == "tree":
        continue
    print(f["Path"])
'
  )
  echo "  ${#files[@]} file(s) to fetch"

  for path in "${files[@]}"; do
    dest="${target}/${path}"
    # Skip anything already complete on disk (resume-safe re-runs).
    remote_size=$(
      curl -sS --noproxy '*' -I --max-time 60 --retry 5 \
        "${API}/${model}/repo?Revision=${REVISION}&FilePath=${path}" |
        awk 'BEGIN{IGNORECASE=1} /^content-length:/{print $2}' | tr -d '\r' | tail -1
    )
    if [[ -n "$remote_size" && -f "$dest" ]]; then
      local_size=$(stat -c %s "$dest" 2>/dev/null || echo 0)
      if [[ "$local_size" == "$remote_size" ]]; then
        echo "  [skip] ${path}"
        continue
      fi
    fi
    mkdir -p "$(dirname "$dest")"
    echo "  [get ] ${path}"
    curl -sS -L --noproxy '*' --retry 5 --retry-delay 3 -C - \
      -o "$dest" \
      "${API}/${model}/repo?Revision=${REVISION}&FilePath=${path}"
  done
  echo "  done: $(du -sh "$target" | cut -f1)"
done

echo "All requested models are in ${MODELS_DIR}"
