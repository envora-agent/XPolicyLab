#!/bin/bash
# Fetch the agent's two perception models; no weights live in this repository.
#   SAM 3:      squarefk/sam3-mirror sam3.pt at a pinned revision, SHA-256 checked,
#               saved to $SAM3_CHECKPOINT or vendor/envora_sim/tools/sam3/checkpoints/sam3.pt
#               (set deploy.yml sam3_checkpoint to a non-default path).
#   Molmo2-4B:  allenai/Molmo2-4B at a pinned commit, into the Hugging Face cache
#               ($HF_HOME), where the server loads it from.
# Usage: bash download_weights.sh [python]   (default: ./.venv/bin/python)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${1:-${SCRIPT_DIR}/.venv/bin/python}"
SAM3_CHECKPOINT="${SAM3_CHECKPOINT:-${SCRIPT_DIR}/vendor/envora_sim/tools/sam3/checkpoints/sam3.pt}"

"${PYTHON}" - "${SAM3_CHECKPOINT}" <<'PY'
import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

SAM3_REVISION = "b675df84f9418772b352b36357d23c1dcba3e767"
SAM3_SHA256 = "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
MOLMO_REVISION = "042abfa7a38879a376cec03d949eff0aefaa0600"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while chunk := source.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


checkpoint = Path(sys.argv[1]).expanduser().absolute()
if checkpoint.is_file() and sha256(checkpoint) == SAM3_SHA256:
    print(f"[Envora_Agent] SAM 3 verified: {checkpoint}")
else:
    downloaded = hf_hub_download(
        "squarefk/sam3-mirror", "sam3.pt", revision=SAM3_REVISION, token=False
    )
    if sha256(downloaded) != SAM3_SHA256:
        raise SystemExit("[Envora_Agent] SAM 3 checkpoint SHA-256 mismatch")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=checkpoint.parent, delete=False) as partial:
        pass
    shutil.copyfile(downloaded, partial.name)
    Path(partial.name).replace(checkpoint)
    print(f"[Envora_Agent] SAM 3 saved: {checkpoint}")

folder = snapshot_download("allenai/Molmo2-4B", revision=MOLMO_REVISION, token=False)
print(f"[Envora_Agent] Molmo2-4B at {MOLMO_REVISION}: {folder}")
PY
