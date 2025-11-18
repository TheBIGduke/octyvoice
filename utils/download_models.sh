#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MODELS_FILE="$SCRIPT_DIR/../config/models.yml"
CACHE_DIR="$HOME/.cache/Local-LLM-for-Robots"

have_cmd(){ command -v "$1" >/dev/null 2>&1; }
die(){ echo "ERROR: $*" >&2; exit 1; }
fetch(){
  local url="$1" out="$2"
  if have_cmd curl; then curl -L --fail --retry 3 -o "$out" "$url"
  else have_cmd wget || die "A curl or wget is needed"; wget -O "$out" "$url"; fi
}

download_file_or_zip(){
  local url="$1"; local out_dir="$2"; local name_hint="${3:-}"
  local fname ext tmp out dname
  [[ -n "$url" ]] || return 1
  mkdir -p "$out_dir"
  fname="$(basename "${url%%\?*}")"
  out="$out_dir/$fname"

  # Simple filename override logic
  if [[ -n "$name_hint" ]]; then
      out="$out_dir/$name_hint"
  fi

  if [[ -f "$out" ]]; then
    echo "  - exists: $out"
  else
    echo "  - downloading: $url -> $out"
    fetch "$url" "$out"
  fi
}

[[ -f "$MODELS_FILE" ]] || die "Missing config/models.yml"
have_cmd yq || die "Missing 'yq' (sudo snap install yq)"

echo "[*] Config: $MODELS_FILE"
echo "[*] Cache: $CACHE_DIR"

# ====== STT ======
echo "[STT] Checking STT models..."
STT_LEN="$(yq -r '(.stt // []) | length' "$MODELS_FILE")"
for i in $(seq 0 $((STT_LEN-1))); do
  NAME="$(yq -r ".stt[$i].name" "$MODELS_FILE")"
  URL="$(yq -r ".stt[$i].url" "$MODELS_FILE")"
  download_file_or_zip "$URL" "$CACHE_DIR/stt" "$NAME"
done

# ====== TTS ======
echo "[TTS] Checking TTS models..."
TTS_LEN="$(yq -r '(.tts // []) | length' "$MODELS_FILE")"
for i in $(seq 0 $((TTS_LEN-1))); do
  NAME="$(yq -r ".tts[$i].name" "$MODELS_FILE")"
  URL="$(yq -r ".tts[$i].url" "$MODELS_FILE")"
  download_file_or_zip "$URL" "$CACHE_DIR/tts" "$NAME"
done

echo "Models ready."