#!/bin/bash

set -euo pipefail
set -x

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

WORKING_DIR=${WORKING_DIR:-/workspace/projects/long-rl}
MODEL_PATH=${MODEL_PATH:-/share/models/dancegrpo/HunyuanVideo}
VIDEOALIGN_CKPT_PATH=${VIDEOALIGN_CKPT_PATH:-/share/models/dancegrpo/videoalign_ckpt}
VIDEOALIGN_BASE_MODEL_PATH=${VIDEOALIGN_BASE_MODEL_PATH:-/workspace/models/Qwen2-VL-2B-Instruct}
DATA_JSON_PATH=${DATA_JSON_PATH:-/share/models/dancegrpo/rl_embeddings/videos2caption.json}

DIST_MASTER_ADDR=${DIST_MASTER_ADDR:-null}
DIST_MASTER_PORT=${DIST_MASTER_PORT:-null}

PROJECT_NAME=${PROJECT_NAME:-dance_case2}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-case2_dance}

NNODES=${NNODES:-1}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
DISAGGREGATE_ACTOR_N_GPUS_PER_NODE=${DISAGGREGATE_ACTOR_N_GPUS_PER_NODE:-4}
DISAGGREGATE_ROLLOUT_REF_N_GPUS_PER_NODE=${DISAGGREGATE_ROLLOUT_REF_N_GPUS_PER_NODE:-4}

MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-4}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-16}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}

NUM_GENERATIONS=${NUM_GENERATIONS:-4}
BESTOFN=${BESTOFN:-4}
SAMPLING_STEPS=${SAMPLING_STEPS:-28}
NUM_STEPS=${NUM_STEPS:-28}
NUM_FRAMES=${NUM_FRAMES:-33}
HEIGHT=${HEIGHT:-256}
WIDTH=${WIDTH:-256}
TIMESTEP_FRACTION=${TIMESTEP_FRACTION:-0.5}
USE_VIDEOALIGN=${USE_VIDEOALIGN:-true}
TOTAL_GPUS=$(( NNODES * N_GPUS_PER_NODE ))

if (( N_GPUS_PER_NODE <= 0 )); then
  echo "N_GPUS_PER_NODE must be > 0, got ${N_GPUS_PER_NODE}" >&2
  exit 1
fi

if (( NNODES <= 0 || TOTAL_GPUS <= 0 )); then
  echo "NNODES and total GPUs must be > 0, got nnodes=${NNODES}, total_gpus=${TOTAL_GPUS}" >&2
  exit 1
fi

if (( DISAGGREGATE_ACTOR_N_GPUS_PER_NODE <= 0 || DISAGGREGATE_ROLLOUT_REF_N_GPUS_PER_NODE <= 0 )); then
  echo "Disaggregate split must be > 0, got actor=${DISAGGREGATE_ACTOR_N_GPUS_PER_NODE}, rollout_ref=${DISAGGREGATE_ROLLOUT_REF_N_GPUS_PER_NODE}" >&2
  exit 1
fi

if (( DISAGGREGATE_ACTOR_N_GPUS_PER_NODE + DISAGGREGATE_ROLLOUT_REF_N_GPUS_PER_NODE > N_GPUS_PER_NODE )); then
  echo "Per-node disaggregate split exceeds available GPUs: actor=${DISAGGREGATE_ACTOR_N_GPUS_PER_NODE}, rollout_ref=${DISAGGREGATE_ROLLOUT_REF_N_GPUS_PER_NODE}, total=${N_GPUS_PER_NODE}" >&2
  exit 1
fi

if (( TRAIN_BATCH_SIZE % TOTAL_GPUS != 0 )); then
  echo "TRAIN_BATCH_SIZE must be divisible by total GPUs for the current FSDP checks: train_batch_size=${TRAIN_BATCH_SIZE}, total_gpus=${TOTAL_GPUS}" >&2
  exit 1
fi

if (( PPO_MINI_BATCH_SIZE > TRAIN_BATCH_SIZE )); then
  echo "PPO_MINI_BATCH_SIZE must be <= TRAIN_BATCH_SIZE, got mini=${PPO_MINI_BATCH_SIZE}, train=${TRAIN_BATCH_SIZE}" >&2
  exit 1
fi

if (( BESTOFN <= 0 || NUM_GENERATIONS <= 0 || BESTOFN > NUM_GENERATIONS )); then
  echo "Require 0 < BESTOFN <= NUM_GENERATIONS, got bestofn=${BESTOFN}, num_generations=${NUM_GENERATIONS}" >&2
  exit 1
fi

python3 -m verl.trainer.main_ppo \
  --config-path="${WORKING_DIR}/examples/diffusion" \
  --config-name=config_video_diffusion_case2_dance \
  hydra.job.chdir=false \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.nnodes="${NNODES}" \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
  trainer.disaggregate_actor_n_gpus_per_node="${DISAGGREGATE_ACTOR_N_GPUS_PER_NODE}" \
  trainer.disaggregate_rollout_ref_n_gpus_per_node="${DISAGGREGATE_ROLLOUT_REF_N_GPUS_PER_NODE}" \
  trainer.dist_master_addr="${DIST_MASTER_ADDR}" \
  trainer.dist_master_port="${DIST_MASTER_PORT}" \
  trainer.max_train_steps="${MAX_TRAIN_STEPS}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.gen_batch_size="${GEN_BATCH_SIZE}" \
  data.data_json_path="${DATA_JSON_PATH}" \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.actor.gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}" \
  actor_rollout_ref.rollout.sampling_steps="${SAMPLING_STEPS}" \
  actor_rollout_ref.rollout.num_steps="${NUM_STEPS}" \
  actor_rollout_ref.rollout.num_frames="${NUM_FRAMES}" \
  actor_rollout_ref.rollout.height="${HEIGHT}" \
  actor_rollout_ref.rollout.width="${WIDTH}" \
  actor_rollout_ref.rollout.num_generations="${NUM_GENERATIONS}" \
  actor_rollout_ref.rollout.bestofn="${BESTOFN}" \
  actor_rollout_ref.actor.extra.dance.pretrained_model_name_or_path="${MODEL_PATH}" \
  actor_rollout_ref.actor.extra.dance.vae_model_path="${MODEL_PATH}" \
  actor_rollout_ref.actor.extra.dance.videoalign_ckpt_path="${VIDEOALIGN_CKPT_PATH}" \
  actor_rollout_ref.actor.extra.dance.use_videoalign="${USE_VIDEOALIGN}" \
  actor_rollout_ref.actor.extra.dance.timestep_fraction="${TIMESTEP_FRACTION}" \
  +actor_rollout_ref.actor.extra.dance.videoalign_base_model_name_or_path="${VIDEOALIGN_BASE_MODEL_PATH}" \
  actor_rollout_ref.actor.extra.dance.master_weight_type=bf16
