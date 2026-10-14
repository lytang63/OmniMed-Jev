# shellcheck shell=bash
# Sourced by the run_*.sh launchers. Puts the shell in the project root and
# gives every large cache directory a home that can be overridden.
#
# Every value is written as ${VAR:-default}, so an existing environment value
# always wins — export what you need before sourcing this file (or edit the
# defaults) to match your machine.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"

# --- Conda (optional) --------------------------------------------------------
# Set CONDA_ENV to have this script activate an environment, e.g.
#   export CONDA_ENV=omnimedjev
#   export CONDA_ROOT=$HOME/miniconda3
# Left unset, the interpreter already on PATH is used as-is.
if [[ -n "${CONDA_ENV:-}" ]]; then
  if [[ -n "${CONDA_ROOT:-}" ]]; then
    # shellcheck source=/dev/null
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  fi
  conda activate "$CONDA_ENV"
fi

# --- Optional HTTP proxy -----------------------------------------------------
# The download scripts read PROXY directly; exporting it here also covers any
# library that reads the standard variables.
if [[ -n "${PROXY:-}" ]]; then
  export http_proxy="$PROXY"
  export https_proxy="$PROXY"
  export no_proxy="${no_proxy:-localhost,127.0.0.1,::1}"
  export NO_PROXY="$no_proxy"
fi

# --- Keep large caches off the system disk -----------------------------------
export TMPDIR="${TMPDIR:-${HOME}/.cache/omnimedjev/tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${HOME}/.cache/pip}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${PROJECT_ROOT}/models/.modelscope}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HOME}/.cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${XDG_CACHE_HOME}/triton}"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$HF_HOME" "$HF_DATASETS_CACHE" \
         "$MODELSCOPE_CACHE" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR"

# Hugging Face: hf_transfer is gone in huggingface_hub 1.x and the Xet
# high-throughput path is unreliable behind a proxy, so keep it off and let the
# plain HTTP downloader run.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

# Uncomment to use a domestic PyPI mirror:
# export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

# Running from the project root makes the relative `models/...` paths inside
# configs/*.yaml resolve correctly.
cd "$PROJECT_ROOT" || return 1
