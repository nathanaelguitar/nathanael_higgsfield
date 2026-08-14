#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
MODELS_ROOT="${MODELS_ROOT:-${ROOT_DIR}/models}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
DOWNLOAD_MODELS=0
INSTALL_HEAVY=0
INSTALL_COMFYUI=0

usage() {
  printf '%s\n' "Usage: $0 [--install-heavy] [--install-comfyui] [--download-models]"
  printf '%s\n' "  --install-heavy   install CUDA PyTorch and model-runtime dependencies"
  printf '%s\n' "  --install-comfyui clone/install ComfyUI wrapper dependencies"
  printf '%s\n' "  --download-models download default checkpoints (several GB)"
}

for arg in "$@"; do
  case "$arg" in
    --install-heavy) INSTALL_HEAVY=1 ;;
    --install-comfyui) INSTALL_COMFYUI=1 ;;
    --download-models) DOWNLOAD_MODELS=1 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$arg" >&2; usage >&2; exit 2 ;;
  esac
done

command -v uv >/dev/null || { printf '%s\n' 'uv is required; install it from https://docs.astral.sh/uv/'; exit 1; }
command -v ffmpeg >/dev/null || { printf '%s\n' 'ffmpeg is required.'; exit 1; }
command -v ffprobe >/dev/null || { printf '%s\n' 'ffprobe is required.'; exit 1; }

ARCH="$(uname -m)"
ensure_repo() {
  local name="$1"
  local url="$2"
  local destination="${ROOT_DIR}/third_party/${name}"
  if [[ ! -d "${destination}/.git" ]]; then
    git clone --depth 1 "${url}" "${destination}"
  fi
}

mkdir -p "${ROOT_DIR}/third_party"
ensure_repo "Wan2.1" "https://github.com/Wan-Video/Wan2.1.git"
ensure_repo "Hallo" "https://github.com/fudan-generative-vision/hallo.git"
ensure_repo "LivePortrait" "https://github.com/KwaiVGI/LivePortrait.git"
ensure_repo "EchoMimicV3" "https://github.com/antgroup/echomimic_v3.git"
ensure_repo "MimicMotion" "https://github.com/tencent/MimicMotion.git"
ensure_repo "CodeFormer" "https://github.com/sczhou/CodeFormer.git"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  uv venv --python "${PYTHON_BIN}" "${VENV_DIR}"
fi
PY="${VENV_DIR}/bin/python"
uv pip install --python "${PY}" -r "${ROOT_DIR}/requirements-base.txt"

if (( INSTALL_HEAVY )); then
  # CUDA 12.8 wheels are forward-compatible with the installed CUDA 13 driver.
  # Override TORCH_INDEX_URL for a vendor-specific DGX ARM64 wheel if needed.
  TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
  printf 'Installing CUDA PyTorch from %s for %s...\n' "${TORCH_INDEX_URL}" "${ARCH}"
  uv pip install --python "${PY}" torch torchvision torchaudio --index-url "${TORCH_INDEX_URL}"
  uv pip install --python "${PY}" diffusers transformers accelerate safetensors \
    omegaconf librosa moviepy opencv-python-headless einops sentencepiece
  # FlashAttention is optional and may not publish a GB10/aarch64 wheel.
  if ! uv pip install --python "${PY}" flash-attn --no-build-isolation; then
    printf '%s\n' 'WARNING: flash-attn unavailable; attention falls back to PyTorch.' >&2
  fi
fi

if (( INSTALL_COMFYUI )); then
  ensure_repo "ComfyUI" "https://github.com/comfyanonymous/ComfyUI.git"
  ensure_repo "ComfyUI-WanVideoWrapper" "https://github.com/kijai/ComfyUI-WanVideoWrapper.git"
  uv pip install --python "${PY}" -r "${ROOT_DIR}/third_party/ComfyUI/requirements.txt"
  if [[ -f "${ROOT_DIR}/third_party/ComfyUI-WanVideoWrapper/requirements.txt" ]]; then
    uv pip install --python "${PY}" -r "${ROOT_DIR}/third_party/ComfyUI-WanVideoWrapper/requirements.txt"
  fi
fi

if (( DOWNLOAD_MODELS )); then
  uv pip install --python "${PY}" "huggingface_hub[cli]"
  HF_BIN="${VENV_DIR}/bin/hf"
  mkdir -p "${MODELS_ROOT}"
  "${HF_BIN}" download Wan-AI/Wan2.1-T2V-1.3B --local-dir "${MODELS_ROOT}/Wan2.1-T2V-1.3B"
  "${HF_BIN}" download BadToBest/EchoMimicV3 --local-dir "${MODELS_ROOT}/EchoMimicV3"
  "${HF_BIN}" download facebook/wav2vec2-base-960h --local-dir "${MODELS_ROOT}/EchoMimicV3/wav2vec2-base-960h"
  "${HF_BIN}" download fudan-generative-ai/hallo --local-dir "${MODELS_ROOT}/Hallo"
  "${HF_BIN}" download KlingTeam/LivePortrait --local-dir "${MODELS_ROOT}/LivePortrait" --exclude '*.git*' README.md docs
  "${HF_BIN}" download tencent/MimicMotion --include MimicMotion_1-1.pth --local-dir "${MODELS_ROOT}/MimicMotion"
  if [[ -f "${ROOT_DIR}/third_party/CodeFormer/scripts/download_pretrained_models.py" ]]; then
    (cd "${ROOT_DIR}/third_party/CodeFormer" && "${PY}" scripts/download_pretrained_models.py facelib)
    (cd "${ROOT_DIR}/third_party/CodeFormer" && "${PY}" scripts/download_pretrained_models.py CodeFormer)
  fi
  printf '%s\n' 'Model downloads complete, including CodeFormer release weights when its downloader succeeds.'
fi

printf '\n%s\n' 'Setup complete.'
printf 'Run: %s --demo --output %s/outputs/demo.mp4\n' "${PY}" "${ROOT_DIR}"
printf 'Production: %s --reference portrait.jpg --audio voice.wav --animation-backend echomimic\n' "${PY}"
