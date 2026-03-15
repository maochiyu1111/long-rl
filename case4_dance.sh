#!/usr/bin/env bash

set -euo pipefail
set -x

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

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
WORLD_SIZE="${WORLD_SIZE:-$((NNODES * N_GPUS_PER_NODE))}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-$(( WORLD_SIZE > 16 ? WORLD_SIZE : 16 ))}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$(( ((PPO_MINI_BATCH_SIZE + WORLD_SIZE - 1) / WORLD_SIZE) * WORLD_SIZE ))}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}"

# Keep the default bring-up config aligned with trainer/worker validation.
NORMALIZED_PPO_MINI_BATCH_SIZE=$((PPO_MINI_BATCH_SIZE / WORLD_SIZE))
if (( NORMALIZED_PPO_MINI_BATCH_SIZE <= 0 )); then
  echo "PPO_MINI_BATCH_SIZE (${PPO_MINI_BATCH_SIZE}) must be >= WORLD_SIZE (${WORLD_SIZE})." >&2
  exit 1
fi
if (( NORMALIZED_PPO_MINI_BATCH_SIZE % PPO_MICRO_BATCH_SIZE_PER_GPU != 0 )); then
  echo "Normalized ppo_mini_batch_size (${NORMALIZED_PPO_MINI_BATCH_SIZE}) must be divisible by PPO_MICRO_BATCH_SIZE_PER_GPU (${PPO_MICRO_BATCH_SIZE_PER_GPU})." >&2
  exit 1
fi
if (( TRAIN_BATCH_SIZE < PPO_MINI_BATCH_SIZE )); then
  echo "TRAIN_BATCH_SIZE (${TRAIN_BATCH_SIZE}) must be >= PPO_MINI_BATCH_SIZE (${PPO_MINI_BATCH_SIZE})." >&2
  exit 1
fi
if (( TRAIN_BATCH_SIZE % WORLD_SIZE != 0 )); then
  echo "TRAIN_BATCH_SIZE (${TRAIN_BATCH_SIZE}) must be divisible by WORLD_SIZE (${WORLD_SIZE})." >&2
  exit 1
fi

declare -a OVERRIDES
OVERRIDES+=("hydra.job.chdir=false")
OVERRIDES+=("trainer.project_name=${PROJECT_NAME}")
OVERRIDES+=("trainer.experiment_name=${EXPERIMENT_NAME}")
OVERRIDES+=("trainer.nnodes=${NNODES}")
OVERRIDES+=("trainer.n_gpus_per_node=${N_GPUS_PER_NODE}")
OVERRIDES+=("trainer.max_train_steps=${MAX_TRAIN_STEPS}")
OVERRIDES+=("data.train_batch_size=${TRAIN_BATCH_SIZE}")
OVERRIDES+=("data.gen_batch_size=${GEN_BATCH_SIZE}")
OVERRIDES+=("actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}")
OVERRIDES+=("actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}")

VIDEOALIGN_CKPT_PATH=/share/models/dancegrpo/videoalign_ckpt
VIDEOALIGN_BASE_MODEL_PATH=/workspace/models/Qwen2-VL-2B-Instruct

# Optional path overrides for quick local bring-up.
if [[ -n "${MODEL_PATH:-}" ]]; then
  OVERRIDES+=("actor_rollout_ref.model.path=${MODEL_PATH}")
  OVERRIDES+=("actor_rollout_ref.actor.extra.dance.pretrained_model_name_or_path=${MODEL_PATH}")
  if [[ -z "${VAE_MODEL_PATH:-}" ]]; then
    OVERRIDES+=("actor_rollout_ref.actor.extra.dance.vae_model_path=${MODEL_PATH}")
  fi
fi
if [[ -n "${VAE_MODEL_PATH:-}" ]]; then
  OVERRIDES+=("actor_rollout_ref.actor.extra.dance.vae_model_path=${VAE_MODEL_PATH}")
fi
if [[ -n "${VIDEOALIGN_CKPT_PATH:-}" ]]; then
  OVERRIDES+=("actor_rollout_ref.actor.extra.dance.videoalign_ckpt_path=${VIDEOALIGN_CKPT_PATH}")
fi
if [[ -n "${VIDEOALIGN_BASE_MODEL_PATH:-}" ]]; then
  OVERRIDES+=("actor_rollout_ref.actor.extra.dance.videoalign_base_model_name_or_path=${VIDEOALIGN_BASE_MODEL_PATH}")
fi
if [[ -n "${DATA_JSON_PATH:-}" ]]; then
  OVERRIDES+=("data.data_json_path=${DATA_JSON_PATH}")
fi

python3 -m verl.trainer.main_ppo \
  --config-path="${CONFIG_PATH}" \
  --config-name="${CONFIG_NAME}" \
  "${OVERRIDES[@]}" \
  "$@"
