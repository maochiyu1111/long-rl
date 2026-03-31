#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MODEL_PATH=/home/qzy/models/HunyuanVideo
VIDEOALIGN_CKPT_PATH=/home/qzy/models/videoalign_ckpt
VIDEOALIGN_BASE_MODEL_PATH=/home/qzy/models/Qwen2-VL-2B-Instruct
DATA_JSON_PATH=/home/qzy/models/rl_embeddings_128/videos2caption.json
REPORT_DIR=${REPORT_DIR:-/home/qzy/project/long-rl/outputs/single/dance_case4_step_timing_12}

python3 -m verl.trainer.main_ppo \
  --config-path=/home/qzy/project/long-rl/examples/diffusion \
  --config-name=config_video_diffusion_case4_dance_npu \
  hydra.job.chdir=false \
  trainer.project_name=dance_case4 \
  trainer.experiment_name=case4_dance_npu \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=8 \
  trainer.max_train_steps=3 \
  trainer.device=npu \
  data.train_batch_size=16 \
  data.gen_batch_size=16 \
  data.data_json_path=${DATA_JSON_PATH} \
  actor_rollout_ref.model.path=${MODEL_PATH} \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.extra.dance.pretrained_model_name_or_path=${MODEL_PATH} \
  actor_rollout_ref.actor.extra.dance.vae_model_path=${MODEL_PATH} \
  actor_rollout_ref.actor.extra.dance.videoalign_ckpt_path=${VIDEOALIGN_CKPT_PATH} \
  actor_rollout_ref.actor.extra.dance.videoalign_base_model_name_or_path=${VIDEOALIGN_BASE_MODEL_PATH} \
  +trainer.step_timing_report_dir=${REPORT_DIR}
