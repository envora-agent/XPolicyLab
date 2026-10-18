#!/bin/bash
set -euo pipefail

bench_name=${1}
task_name=${2}
ckpt_name=${3}
env_cfg_type=${4}
action_type=${5}
seed=${6}
policy_gpu_id=${7}
policy_conda_env=${8}
policy_server_port=${9}
policy_server_host=${10:-"localhost"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${XPL_ROOT}/policy/${policy_name}/deploy.yml"

echo "[SERVER] policy=${policy_name}, task=${task_name}, policy_server_port=${policy_server_port}"

# <policy_env>: `uv` (policy_uv_env_path in deploy.yml, default this directory),
# a directory holding .venv, or a conda environment name.
if [[ "${policy_conda_env}" == "uv" || "${policy_conda_env}" == */* ]]; then
    if [[ "${policy_conda_env}" == "uv" ]]; then
        policy_uv_env_path="$(sed -n 's/^policy_uv_env_path:[[:space:]]*//p' "${yaml_file}")"
        policy_uv_env_path="${policy_uv_env_path:-.}"
    else
        policy_uv_env_path="${policy_conda_env}"
    fi
    if [[ "${policy_uv_env_path}" != /* ]]; then
        policy_uv_env_path="${SCRIPT_DIR}/${policy_uv_env_path}"
    fi
    if [[ ! -f "${policy_uv_env_path}/.venv/bin/activate" ]]; then
        echo "[SERVER][ERROR] uv venv not found: ${policy_uv_env_path}/.venv" >&2
        echo "[SERVER][ERROR] Run: bash ${SCRIPT_DIR}/install.sh" >&2
        exit 1
    fi
    echo "[SERVER] Activating uv environment: ${policy_uv_env_path}/.venv"
    source "${policy_uv_env_path}/.venv/bin/activate"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
    echo "[SERVER] Activating Conda environment: ${policy_conda_env}"
    conda activate "${policy_conda_env}"
fi
echo "[SERVER] Using python: $(command -v python)"

# The vendored envora_sim comes first: the SAM 3 and Molmo2-4B workers the model
# starts import it too, on the same GPU.
exec env \
    PYTHONUNBUFFERED=1 \
    PYTHONWARNINGS=ignore::UserWarning \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    PYTHONPATH="${SCRIPT_DIR}/vendor:${BENCH_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    python "${XPL_ROOT}/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides \
            port="${policy_server_port}" \
            host="${policy_server_host}" \
            bench_name="${bench_name}" \
            task_name="${task_name}" \
            ckpt_name="${ckpt_name}" \
            env_cfg_type="${env_cfg_type}" \
            seed="${seed}" \
            policy_name="${policy_name}" \
            action_type="${action_type}"
