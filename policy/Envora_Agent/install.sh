#!/bin/bash
# The Envora_Agent policy environment: a Python 3.11 uv venv at ./.venv with
# torch (CUDA 12.8), SAM 3, Molmo2-4B's transformers and XPolicyLab, then the
# model weights (download_weights.sh; ENVORA_SKIP_WEIGHTS=1 skips them).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
TORCH_INDEX="${ENVORA_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "[Envora_Agent] Creating ${VENV_DIR} with Python 3.11"
    uv venv --python 3.11 "${VENV_DIR}"
else
    echo "[Envora_Agent] Reusing ${VENV_DIR}"
fi
PYTHON="${VENV_DIR}/bin/python"

UV_LINK_MODE=copy uv pip install --python "${PYTHON}" \
    --index-url "${TORCH_INDEX}" \
    "torch==2.7.0" "torchvision==0.22.0"

UV_LINK_MODE=copy uv pip install --python "${PYTHON}" \
    "numpy==1.26.4" \
    "scipy==1.15.3" \
    "opencv-python-headless==4.11.0.86" \
    "pillow==11.3.0" \
    "shapely==2.1.2" \
    "numba==0.59.1" \
    "transformers==4.57.6" \
    "accelerate==1.15.0" \
    "huggingface-hub==0.35.3" \
    "timm==1.0.17" \
    "einops==0.8.2" \
    "pyyaml==6.0.2" \
    "pycocotools==2.0.10" \
    "sam3 @ git+https://github.com/facebookresearch/sam3.git@660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7"

UV_LINK_MODE=copy uv pip install --python "${PYTHON}" -e "${XPL_ROOT}"

PYTHONPATH="${SCRIPT_DIR}/vendor" "${PYTHON}" - <<'PY'
import cv2, numba, scipy, shapely, torch, transformers
import XPolicyLab
from envora_sim.environment.robodojo_physx.solutions.unified.agent import ROUTES
from envora_sim.tools import molmo2_4b, sam3

print(f"[Envora_Agent] torch={torch.__version__} cuda={torch.cuda.is_available()}")
print(f"[Envora_Agent] transformers={transformers.__version__} {len(ROUTES)} task routes")
PY

if [[ "${ENVORA_SKIP_WEIGHTS:-0}" != "1" ]]; then
    bash "${SCRIPT_DIR}/download_weights.sh" "${PYTHON}"
fi
echo "[Envora_Agent] Installation finished"
