#!/bin/bash
# Install flash-attn from the official prebuilt wheel.
#
# WHY the prebuilt wheel and not a source build: building flash-attn needs a
# full CUDA toolkit (nvcc) plus ~1-2 hours and tens of GB of RAM. The project
# publishes matched wheels on its GitHub releases, and one exists for exactly
# the combination used here — cu12, torch 2.6, cp312, cxx11 ABI FALSE (check
# yours via torch._C._GLIBCXX_USE_CXX11_ABI). Installing it is a download, not
# a compile.
#
# --no-deps is essential: flash-attn's metadata pins torch/torchvision, and
# letting pip resolve would drag in a different torch build.
#
# Edit WHEEL_NAME, or override it in the environment, if your torch/CUDA/Python
# combination differs.
set -uo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${WORKDIR:-${TMPDIR:-/tmp}/flash-attn-wheel}"
WHEEL_NAME="${WHEEL_NAME:-flash_attn-2.8.3+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl}"
URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/${WHEEL_NAME}"
# Set PROXY to route the download through an HTTP proxy; unset means direct.
PROXY="${PROXY:-}"

mkdir -p "$WORKDIR"

if [[ ! -s "${WORKDIR}/${WHEEL_NAME}" ]]; then
  echo "=== downloading ${WHEEL_NAME} ==="
  # aria2c rather than pip's own fetcher: GitHub release downloads are slow and
  # stall, and pip offers no resume. -x16 measured fastest.
  proxy_args=()
  [[ -n "$PROXY" ]] && proxy_args=(--all-proxy="$PROXY")
  aria2c -x 16 -s 16 -k 1M --file-allocation=none --continue=true \
    --max-tries=30 --retry-wait=5 --summary-interval=0 --no-conf \
    "${proxy_args[@]}" -d "$WORKDIR" -o "$WHEEL_NAME" "$URL" 2>&1 | tail -3
else
  echo "=== wheel already present ==="
fi
ls -la "${WORKDIR}/${WHEEL_NAME}" 2>/dev/null | awk '{printf "  %.1f MB\n", $5/1048576}'

echo "=== installing (--no-deps) ==="
python -m pip install --no-deps "${WORKDIR}/${WHEEL_NAME}" 2>&1 | tail -3

echo "=== verifying ==="
python - <<'PY'
import torch
print("  torch still:", torch.__version__)
try:
    import flash_attn
    from flash_attn import flash_attn_func  # noqa: F401
    print("  flash_attn:", flash_attn.__version__, "-> import OK")
except Exception as exc:
    print("  flash_attn FAILED:", type(exc).__name__, str(exc)[:160])
PY
