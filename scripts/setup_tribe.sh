#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRIBE_VENV="${TRIBE_VENV:-${ROOT_DIR}/.tribe_venv}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
cd "${ROOT_DIR}"

command -v uv >/dev/null || {
  printf '%s\n' 'uv is required; install it from https://docs.astral.sh/uv/' >&2
  exit 1
}

if [[ ! -x "${TRIBE_VENV}/bin/python" ]]; then
  uv venv --python "${PYTHON_BIN}" --seed "${TRIBE_VENV}"
fi

PY="${TRIBE_VENV}/bin/python"
uv pip install --python "${PY}" \
  pydantic==2.13.4 polars==1.43.2 matplotlib==3.11.1 \
  mne==1.12.1 mne-bids==0.19.0 pyprep==0.8.0 \
  neuralset==0.0.2 neuraltrain==0.0.2 exca==0.5.20 \
  x-transformers==1.27.20 gtts langdetect spacy Levenshtein \
  huggingface_hub==0.26.2 transformers==4.46.2 moviepy==2.2.1 \
  soundfile==0.13.1 julius
uv pip install --python "${PY}" --no-deps -e "${ROOT_DIR}/third_party/tribev2"

"${PY}" "${ROOT_DIR}/scripts/download_tribe_assets.py" "$@"

printf '\n%s\n' 'TRIBE setup complete. No model is loaded by this script.'
printf 'For gated Llama access, rerun with: %s --include-llama\n' "$0"
