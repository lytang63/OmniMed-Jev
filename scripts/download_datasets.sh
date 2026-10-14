#!/bin/bash
# Download the source datasets used by OmniMed-Jev.
#
# Everything lands under datas/raw/<dataset>/ and is never modified in place.
# Each file is staged on tmpfs, checked against Zenodo's published size AND md5,
# and only then moved into datas/raw/. See ARIA_COMMON below for why the
# connection count is 8 and not 16.
#
# ChestX-Det images come from the ModelScope mirror OmniData/ChestX-Det, whose
# raw/*.zip sizes match the Deepwise originals byte-for-byte. MedMNIST, CellSeg
# and IUGC come from Zenodo, which is the canonical host for all three.
#
# Usage:
#   ./scripts/download_datasets.sh                 # everything
#   ./scripts/download_datasets.sh medmnist chestx # only named groups
#   GROUPS: medmnist chestx txlpbc camus iugc cellseg iuxray
#
# The counting and regression families (txlpbc, camus) come from Hugging Face
# and need `huggingface-cli`; every other group uses aria2c. See the
# prerequisites section in docs/REPRODUCE.md.
#
# Set PROXY to route the fetches through an HTTP proxy, e.g.
#   PROXY=http://127.0.0.1:7890 ./scripts/download_datasets.sh
# With PROXY unset every request goes out directly.
set -uo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="${PROJECT_ROOT}/datas/raw"
LOGS="${PROJECT_ROOT}/datas/logs"
PIDFILE="${LOGS}/download.pid"
# tmpfs staging area; see the comment in fetch() for why downloads do not land
# on datas/raw/ directly.
STAGE_ROOT="${STAGE_ROOT:-/dev/shm}"
PROXY="${PROXY:-}"
# Zenodo asks for an identifiable User-Agent (see ARIA_COMMON below); set
# CONTACT to a real address so the request carries one.
CONTACT="${CONTACT:-anonymous@example.com}"
USER_AGENT="OmniMed-Jev-downloader/1.0 (+https://github.com/lytang63/OmniMed-Jev; ${CONTACT})"
# curl equivalent for the metadata API. Must be defined before ZENODO_PROXY_ARGS.
CURL_COMMON=(-A "$USER_AGENT")
# The metadata API lives on the same host, so it takes the same route as the
# files: through PROXY when one is configured, direct otherwise.
ZENODO_PROXY_ARGS=("${CURL_COMMON[@]}")
[[ -n "$PROXY" ]] && ZENODO_PROXY_ARGS=(--proxy "$PROXY" "${CURL_COMMON[@]}")
# CONNECTION COUNT IS A CORRECTNESS SETTING HERE, NOT A SPEED KNOB.
#
# Measured against Zenodo on breastmnist_224.npz (30 MB) with md5 verification:
#     -x1 / -x2 / -x4 / -x8  -> md5 correct
#     -x16                   -> file lands at exactly the right size with the
#                               wrong md5, or wedges part-way
# Every one of 11 consecutive -x16 fetches failed verification. Do not raise
# this without re-running the md5 check above.
#
# --continue is also deliberately absent: resuming trusts bytes already on
# disk, and a torn partial (right size, bad tail) yields a corrupt result that
# still reports success. See the guard in fetch().
ARIA_COMMON=(-x8 -s8 -k1M --file-allocation=none
             --max-tries=10 --retry-wait=5 --console-log-level=warn
             --summary-interval=0 --no-conf
             # Drop a connection that stalls, so one wedged range cannot block
             # the whole sequential queue.
             --lowest-speed-limit=50K
             # Zenodo rate-limits or blocks high-volume requests that carry a
             # generic User-Agent ("the default curl or python-requests") and
             # asks for an identifiable one. aria2c's default UA is exactly
             # that kind of generic string, so set an explicit one. MedMNIST,
             # CellSeg and IUGC all start returning 0 bytes after a few hundred
             # requests without it.
             --user-agent="$USER_AGENT")
mkdir -p "$RAW" "$LOGS"
if ! touch "${STAGE_ROOT}/.write_test" 2>/dev/null; then
  echo "STAGE_ROOT ${STAGE_ROOT} is not writable; set STAGE_ROOT to a local dir." >&2
  exit 1
fi
rm -f "${STAGE_ROOT}/.write_test"
# PID file rather than pgrep: an interactive caller's own command line often
# contains this script's name (e.g. `bash -n scripts/download_datasets.sh`),
# so pattern-matching kills the caller. A pidfile cannot self-match.
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

# fetch <outdir> <proxy|direct> <url> [label] [expected-md5] [expected-size]
#
# The md5 check is not optional in spirit: a dermamnist_224.npz was observed to
# download to the *exactly correct byte count* while its tail was corrupt, and
# aria2c then span for 30+ minutes re-fetching bad pieces without ever
# completing. Size alone is not evidence of a good file.
#
# The size is also checked, separately from the md5, purely so the log says
# "incomplete" when the transfer stopped early and "md5 MISMATCH" when a
# complete-sized file has the wrong contents. Those two failures have different
# causes and conflating them (which this script did at first) sends you hunting
# the wrong bug.
fetch() {
  local outdir="$1" mode="$2" url="$3" label="${4:-$(basename "${url%%\?*}")}" want_md5="${5:-}" want_size="${6:-}"
  mkdir -p "$outdir"
  local dest="$outdir/$label"
  local proxy_args=()
  if [[ "$mode" == "proxy" ]]; then
    proxy_args=(--all-proxy="$PROXY")
  else
    # aria2c honours http_proxy/https_proxy from the environment, so an empty
    # --all-proxy alone is not enough; strip the vars via env -u.
    proxy_args=(--all-proxy="")
  fi

  if [[ -f "$dest" ]]; then
    if [[ -n "$want_md5" && "$(md5sum "$dest" | cut -d' ' -f1)" == "$want_md5" ]]; then
      echo ">>> [skip] $label (md5 ok)"
      return 0
    fi
    # An existing file that does NOT match is either a torn partial or a
    # corrupt result. Resuming onto it would verify the *new* bytes against a
    # bad prefix and produce another corrupt file of exactly the right size —
    # which is precisely how dermamnist_224.npz and four CellSeg/IUGC archives
    # were silently corrupted. Throw it away and start clean.
    echo "    discarding untrusted existing $label before a clean fetch"
    rm -f "$dest"
  fi

  # Stage on tmpfs, then move into place.
  #
  # Downloads written straight to a network filesystem produced files of exactly
  # the right size with the wrong md5 roughly 3 times out of 4, while the same
  # bytes fetched to local memory did not. /dev/shm is a tmpfs, so staging there
  # removes the network filesystem from the write path for the part that
  # matters. The final move is a plain sequential copy.
  local stage
  stage="$(mktemp -d "${STAGE_ROOT}/omnimedtmp.XXXXXX")"

  local attempt rc=1 got
  for attempt in $(seq 1 "${FETCH_ATTEMPTS:-3}"); do
    echo ">>> [$mode] $label  (attempt ${attempt}/${FETCH_ATTEMPTS:-3}, staging on ${STAGE_ROOT})"
    env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
      aria2c "${ARIA_COMMON[@]}" "${proxy_args[@]}" -d "$stage" -o "$label" "$url" \
      2>&1 | tail -2

    local staged="$stage/$label"
    local have_size=0
    [[ -f "$staged" ]] && have_size="$(stat -c %s "$staged")"
    rc=0
    if [[ "$have_size" == "0" ]]; then
      echo "    no data received" >&2
      rc=1
    elif [[ -n "$want_size" && "$have_size" != "$want_size" ]]; then
      echo "    INCOMPLETE: ${have_size}/${want_size} bytes — transfer stopped early" >&2
      rc=1
    elif [[ -n "$want_md5" ]]; then
      got="$(md5sum "$staged" | cut -d' ' -f1)"
      if [[ "$got" != "$want_md5" ]]; then
        echo "    CORRUPT: full ${have_size} bytes but md5 $got != $want_md5" >&2
        rc=1
      fi
    fi

    if [[ "$rc" == "0" ]]; then
      mv -f "$staged" "$dest"
      [[ -n "$want_md5" ]] && echo "    md5 ok -> $dest"
      break
    fi
    rm -f "$staged" "$staged".aria2
  done

  rm -rf "$stage"
  return "$rc"
}

# zenodo_meta <record-id> <filename> -> "<md5> <size>", or empty if unavailable
zenodo_meta() {
  curl -sS --max-time 60 ${ZENODO_PROXY_ARGS[@]+"${ZENODO_PROXY_ARGS[@]}"} \
    "https://zenodo.org/api/records/$1" 2>/dev/null |
    python3 -c "
import json, sys
want = sys.argv[1]
try:
    for f in json.load(sys.stdin).get('files', []):
        if f.get('key') == want:
            print(f['checksum'].split(':', 1)[1], f['size']); break
except Exception:
    pass
" "$2"
}

MEDMNIST_RECORD=10519652
IUGC_RECORD=17355570
CELLSEG_RECORD=10719375

# MedMNIST+ 224: all twelve 2D subsets (~28.2 GB total).
# dermamnist is LAST on purpose: Zenodo throttles that particular file hard
# enough that it repeatedly wedges at the same offset, and the queue is
# sequential, so putting it first starves everything behind it.
MEDMNIST_224=(breastmnist pathmnist chestmnist retinamnist bloodmnist
              octmnist tissuemnist organamnist organcmnist organsmnist pneumoniamnist
              dermamnist)

dl_medmnist() {
  echo "=== MedMNIST+ 224 (Zenodo ${MEDMNIST_RECORD}) ==="
  for s in "${MEDMNIST_224[@]}"; do
    local meta md5 size
    meta="$(zenodo_meta "$MEDMNIST_RECORD" "${s}_224.npz")"
    md5="${meta%% *}"; size="${meta##* }"
    fetch "$RAW/medmnistplus" proxy \
      "https://zenodo.org/records/${MEDMNIST_RECORD}/files/${s}_224.npz?download=1" \
      "${s}_224.npz" "$md5" "$size" || echo "    !! ${s}_224.npz failed verification, re-run to retry" >&2
  done
}

dl_chestx() {
  echo "=== ChestX-Det (images: ModelScope mirror; annotations: GitHub) ==="
  local ms="https://modelscope.cn/api/v1/datasets/OmniData/ChestX-Det/repo?Revision=master&FilePath=raw"
  fetch "$RAW/chestx-det" direct "${ms}/train_data.zip" train_data.zip
  fetch "$RAW/chestx-det" direct "${ms}/test_data.zip"  test_data.zip
  local gh="https://raw.githubusercontent.com/Deepwise-AILab/ChestX-Det-Dataset/main"
  local out="$RAW/chestx-det"
  mkdir -p "$out"
  for f in ChestX_Det_train.json ChestX_Det_test.json; do
    echo ">>> [direct] $f"
    env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
      curl -sS -L --retry 5 -o "$out/$f" "$gh/$f" && echo "    $(stat -c %s "$out/$f") bytes"
  done
}

dl_iugc() {
  echo "=== IUGC 2025 (Zenodo ${IUGC_RECORD}) ==="
  local meta md5 size
  meta="$(zenodo_meta "$IUGC_RECORD" "Dataset.zip")"
  md5="${meta%% *}"; size="${meta##* }"
  fetch "$RAW/iugc2025" proxy \
    "https://zenodo.org/records/${IUGC_RECORD}/files/Dataset.zip?download=1" "Dataset.zip" "$md5" "$size"
}

dl_cellseg() {
  echo "=== NeurIPS22 CellSeg (Zenodo ${CELLSEG_RECORD}) — supervised parts only ==="
  # train-unlabeled-part1/2 (~14.7 GB) are deliberately skipped: they are only
  # useful for semi-supervised pretraining, not for the supervised benchmark.
  for f in Training-labeled.zip Tuning.zip Testing.zip; do
    local meta md5 size
    meta="$(zenodo_meta "$CELLSEG_RECORD" "$f")"
    md5="${meta%% *}"; size="${meta##* }"
    fetch "$RAW/neurips22-cellseg" proxy \
      "https://zenodo.org/records/${CELLSEG_RECORD}/files/${f}?download=1" "$f" "$md5" "$size"
  done
}

dl_iuxray() {
  echo "=== IU X-Ray (NLM Open-I) ==="
  # NLMCXR_dcm.tgz is 75 GB of DICOM; the PNG collection is all the report task needs.
  fetch "$RAW/iu-xray" proxy "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz"     NLMCXR_png.tgz
  fetch "$RAW/iu-xray" proxy "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz" NLMCXR_reports.tgz
}

# The counting and regression families. These live on Hugging Face rather than
# Zenodo, so they go through huggingface-cli instead of fetch().
hf_dataset() {
  local repo="$1" dest="$2"
  if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "  !! huggingface-cli not found; pip install huggingface_hub" >&2
    return 1
  fi
  echo "=== ${repo} -> ${dest} ==="
  if [[ -n "$PROXY" ]]; then
    HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
      HTTPS_PROXY="$PROXY" HTTP_PROXY="$PROXY" \
      huggingface-cli download "$repo" --repo-type dataset --local-dir "$dest"
  else
    huggingface-cli download "$repo" --repo-type dataset --local-dir "$dest"
  fi
}

dl_txlpbc() {
  hf_dataset KeenForgeAI/TXL-PBC-corrected "$RAW/txl-pbc"
}

dl_camus() {
  hf_dataset MedOtter/CAMUS "$RAW/camus"
}

# NOTE: do not name this array GROUPS — that is a bash special variable holding
# the caller's supplementary group IDs, and assignment to it silently fails.
if [[ $# -eq 0 ]]; then
  TARGETS=(medmnist chestx txlpbc camus iugc cellseg iuxray)
else
  TARGETS=("$@")
fi

START=$(date +%s)
for g in "${TARGETS[@]}"; do
  case "$g" in
    medmnist) dl_medmnist ;;
    chestx)   dl_chestx ;;
    txlpbc)   dl_txlpbc ;;
    camus)    dl_camus ;;
    iugc)     dl_iugc ;;
    cellseg)  dl_cellseg ;;
    iuxray)   dl_iuxray ;;
    *) echo "Unknown group: $g" >&2; exit 1 ;;
  esac
done
END=$(date +%s)

echo
echo "=== download summary (elapsed $(( (END-START)/60 ))m $(( (END-START)%60 ))s) ==="
for d in "$RAW"/*/; do
  n=$(find "$d" -maxdepth 1 -type f \( -name '*.zip' -o -name '*.npz' -o -name '*.tgz' \) | wc -l)
  sz=$(du -sh "$d" 2>/dev/null | cut -f1)
  printf "  %-22s %2d archive(s)  %s\n" "$(basename "$d")" "$n" "$sz"
done
