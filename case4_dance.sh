#!/usr/bin/env bash

set -euo pipefail
set -x

export PYTHONUNBUFFERED=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${ROOT_DIR}/examples/diffusion}"
CONFIG_NAME="${CONFIG_NAME:-config_video_diffusion_case4_dance}"
CONFIG_FILE="${CONFIG_PATH}/${CONFIG_NAME}.yaml"

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Missing config file: ${CONFIG_FILE}" >&2
  echo "Please create it first (see case4todolist.md, Stage E)." >&2
  exit 1
fi

PROJECT_NAME="${PROJECT_NAME:-dance_case4}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-case4_dance_$(date +%Y%m%d_%H%M%S)}"
NNODES="${NNODES:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1}"

declare -a OVERRIDES
OVERRIDES+=("hydra.job.chdir=false")
OVERRIDES+=("trainer.project_name=${PROJECT_NAME}")
OVERRIDES+=("trainer.experiment_name=${EXPERIMENT_NAME}")
OVERRIDES+=("trainer.nnodes=${NNODES}")
OVERRIDES+=("trainer.n_gpus_per_node=${N_GPUS_PER_NODE}")
OVERRIDES+=("trainer.max_train_steps=${MAX_TRAIN_STEPS}")

# Optional path overrides for quick local bring-up.
if [[ -n "${MODEL_PATH:-}" ]]; then
  OVERRIDES+=("actor_rollout_ref.actor.extra.dance.pretrained_model_name_or_path=${MODEL_PATH}")
fi
if [[ -n "${VAE_MODEL_PATH:-}" ]]; then
  OVERRIDES+=("actor_rollout_ref.actor.extra.dance.vae_model_path=${VAE_MODEL_PATH}")
fi
if [[ -n "${VIDEOALIGN_CKPT_PATH:-}" ]]; then
  OVERRIDES+=("actor_rollout_ref.actor.extra.dance.videoalign_ckpt_path=${VIDEOALIGN_CKPT_PATH}")
fi
if [[ -n "${DATA_JSON_PATH:-}" ]]; then
  OVERRIDES+=("data.data_json_path=${DATA_JSON_PATH}")
fi

python3 -m verl.trainer.main_ppo \
  --config-path="${CONFIG_PATH}" \
  --config-name="${CONFIG_NAME}" \
  "${OVERRIDES[@]}" \
  "$@"

