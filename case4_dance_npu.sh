#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VERL_DANCE_ENABLE_TRACE=${VERL_DANCE_ENABLE_TRACE:-1}
export VERL_DANCE_TRACE_LOG_ALL_STEPS=${VERL_DANCE_TRACE_LOG_ALL_STEPS:-1}
export VERL_DANCE_TRACE_DIR=${VERL_DANCE_TRACE_DIR:-./dance_case4_traces}
export VERL_DANCE_SANITIZE_EXPORT_VIDEO=${VERL_DANCE_SANITIZE_EXPORT_VIDEO:-1}
export VERL_DANCE_SANITIZE_REWARD_NAN=${VERL_DANCE_SANITIZE_REWARD_NAN:-1}
export VERL_DANCE_STABILIZE_RATIO=${VERL_DANCE_STABILIZE_RATIO:-1}

MODEL_PATH=/home/qzy/models/HunyuanVideo
VIDEOALIGN_CKPT_PATH=/home/qzy/models/videoalign_ckpt
VIDEOALIGN_BASE_MODEL_PATH=/home/qzy/models/Qwen2-VL-2B-Instruct
DATA_JSON_PATH=/home/qzy/models/rl_embeddings_128/videos2caption.json

TRACE_BATCH_SIZE=${TRACE_BATCH_SIZE:-8}
TRACE_PPO_MINI_BATCH_SIZE=${TRACE_PPO_MINI_BATCH_SIZE:-${TRACE_BATCH_SIZE}}
TRACE_SAMPLING_STEPS=${TRACE_SAMPLING_STEPS:-12}

python3 -m verl.trainer.main_ppo \
  --config-path=/home/qzy/project/long-rl/examples/diffusion \
  --config-name=config_video_diffusion_case4_dance_npu \
  hydra.job.chdir=false \
  trainer.project_name=dance_case4 \
  trainer.experiment_name=case4_dance_npu \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=8 \
  trainer.max_train_steps=4 \
  trainer.device=npu \
  data.train_batch_size=${TRACE_BATCH_SIZE} \
  data.gen_batch_size=${TRACE_BATCH_SIZE} \
  data.data_json_path=${DATA_JSON_PATH} \
  actor_rollout_ref.model.path=${MODEL_PATH} \
  actor_rollout_ref.actor.ppo_mini_batch_size=${TRACE_PPO_MINI_BATCH_SIZE} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.num_steps=${TRACE_SAMPLING_STEPS} \
  actor_rollout_ref.rollout.sampling_steps=${TRACE_SAMPLING_STEPS} \
  actor_rollout_ref.actor.extra.dance.pretrained_model_name_or_path=${MODEL_PATH} \
  actor_rollout_ref.actor.extra.dance.vae_model_path=${MODEL_PATH} \
  actor_rollout_ref.actor.extra.dance.videoalign_ckpt_path=${VIDEOALIGN_CKPT_PATH} \
  actor_rollout_ref.actor.extra.dance.videoalign_base_model_name_or_path=${VIDEOALIGN_BASE_MODEL_PATH} \
  +actor_rollout_ref.actor.extra.dance.enable_trace=true \
  +actor_rollout_ref.actor.extra.dance.trace_log_all_steps=true \
  +actor_rollout_ref.actor.extra.dance.sanitize_export_video=true \
  +actor_rollout_ref.actor.extra.dance.sanitize_reward_nan=true \
  +actor_rollout_ref.actor.extra.dance.stabilize_ratio=true
