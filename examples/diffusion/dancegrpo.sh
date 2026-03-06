#!/usr/bin/env bash

set -xeuo pipefail

export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-disabled}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MODEL_PATH="${MODEL_PATH:-/share/models/Wan2.1-T2V-1.3B-Diffusers}"
VIDEOALIGN_MODEL_PATH="${VIDEOALIGN_MODEL_PATH:-}"

if [[ -z "${VIDEOALIGN_MODEL_PATH}" ]]; then
  echo "VIDEOALIGN_MODEL_PATH is required for dancegrpo dual rewards." >&2
  exit 1
fi

python3 -m verl.trainer.main_ppo \
  --config-path="${REPO_ROOT}/examples/diffusion" \
  --config-name=dancegrpo \
  hydra.job.chdir=false \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.trust_remote_code=true \
  actor_rollout_ref.ref.scheduler="${MODEL_PATH}/scheduler" \
  reward_model.reward_kwargs.videoalign.load_from_pretrained="${VIDEOALIGN_MODEL_PATH}" \
  trainer.experiment_name="${EXPERIMENT_NAME:-wan2_1_t2v_1.3b_dancegrpo_case4}" \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE:-8}"
