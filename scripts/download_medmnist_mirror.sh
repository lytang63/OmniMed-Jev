#!/bin/bash
# Download MedMNIST+ 224 from the ModelScope mirror into datas/raw/medmnistplus_ms/.
#
# WHY a mirror at all: Zenodo (the canonical host) rate-limits automated clients
# and can return 0 bytes for every file. ModelScope user `danjacobellis` mirrors
# all twelve 2D subsets as HuggingFace parquet, which is the same data in a
# different container — verified against RetinaMNIST, whose test split came back
# with 400 images and the exact official label histogram {0:174,1:46,2:92,3:68,
# 4:20}. It is also smaller (16.0 GB vs 28.2 GB).
#
# Layout per subset:  {subset}_224/data/{train,validation,test}-NNNNN-of-NNNNN.parquet
#                     {subset}_224/dataset_infos.json
#
# Zenodo publishes md5 for its .npz files; ModelScope does not, so verification
# here is SIZE ONLY. That is weaker, but a size mismatch still catches a
# truncated transfer.
#
# Usage:
#   ./scripts/download_medmnist_mirror.sh            # all twelve subsets
#   ./scripts/download_medmnist_mirror.sh dermamnist # one or more, by stem
set -uo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${PROJECT_ROOT}/datas/raw/medmnistplus_ms"
LOGS="${PROJECT_ROOT}/datas/logs"
STAGE_ROOT="${STAGE_ROOT:-/dev/shm}"
PIDFILE="${LOGS}/medmnist_ms.pid"
API="https://modelscope.cn/api/v1/datasets/danjacobellis"
# ModelScope is fetched direct (--noproxy), so any ambient http_proxy is ignored.
CONTACT="${CONTACT:-anonymous@example.com}"
USER_AGENT="OmniMed-Jev-downloader/1.0 (+https://github.com/lytang63/OmniMed-Jev; ${CONTACT})"

SUBSETS=(pathmnist chestmnist tissuemnist octmnist organamnist bloodmnist organsmnist
         organcmnist breastmnist dermamnist retinamnist pneumoniamnist)

mkdir -p "$DEST" "$LOGS"
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

if [[ $# -gt 0 ]]; then
  SUBSETS=("$@")
fi

# list_files <subset> -> "<path> <size>" per line
list_files() {
  curl -sS -A "$USER_AGENT" --noproxy '*' --max-time 60 --retry 3 \
    "${API}/$1/repo/tree?Revision=master&Recursive=true" 2>/dev/null |
    python3 -c "
import json, sys
d = json.load(sys.stdin)
if d.get('Code') != 200:
    sys.exit(0)
for f in d['Data']['Files']:
    if f.get('Type') == 'tree':
        continue
    if f['Path'].startswith('data/') or f['Path'] == 'dataset_infos.json':
        print(f['Path'], f['Size'])
"
}

total_ok=0
total_fail=0
for subset in "${SUBSETS[@]}"; do
  echo "=== ${subset}_224 ==="
  mapfile -t ENTRIES < <(list_files "${subset}_224")
  if [[ ${#ENTRIES[@]} -eq 0 ]]; then
    echo "  !! could not list files" >&2
    total_fail=$((total_fail + 1))
    continue
  fi

  for entry in "${ENTRIES[@]}"; do
    path="${entry% *}"
    want_size="${entry##* }"
    dest="${DEST}/${subset}_224/${path}"
    mkdir -p "$(dirname "$dest")"

    if [[ -f "$dest" && "$(stat -c %s "$dest")" == "$want_size" ]]; then
      echo "  [skip] $path"
      continue
    fi
    rm -f "$dest"

    ok=0
    for attempt in 1 2 3; do
      stage="$(mktemp -d "${STAGE_ROOT}/msdl.XXXXXX")"
      # The API returns the raw bytes for any path with FilePath=; URL-encode
      # the slashes so `data/...` survives as a single query value.
      encoded="$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1], safe=''))" "$path")"
      aria2c -x8 -s8 --file-allocation=none --summary-interval=0 --no-conf \
        --user-agent="$USER_AGENT" --all-proxy="" \
        -d "$stage" -o payload \
        "${API}/${subset}_224/repo?Revision=master&FilePath=${encoded}" >/dev/null 2>&1
      have="$(stat -c %s "$stage/payload" 2>/dev/null || echo 0)"
      if [[ "$have" == "$want_size" ]]; then
        mv -f "$stage/payload" "$dest"
        echo "  [ok  ] $path  ($((have / 1048576)) MB)"
        ok=1
      else
        echo "  [retry ${attempt}/3] $path  ${have}/${want_size} bytes" >&2
      fi
      rm -rf "$stage"
      [[ "$ok" == "1" ]] && break
      sleep 3
    done
    if [[ "$ok" == "1" ]]; then total_ok=$((total_ok + 1)); else
      echo "  !! FAILED $path" >&2
      total_fail=$((total_fail + 1))
    fi
  done
done

echo
echo "=== done: ${total_ok} file(s) ok, ${total_fail} failed ==="
du -sh "$DEST" 2>/dev/null | tail -1
