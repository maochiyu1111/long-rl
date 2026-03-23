#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VERL_SOCKET_IFACE_PREFIX=192.158.0.

RAY_ADDRESS=http://192.158.0.14:8265
WORKING_DIR=/workspace/projects/long-rl
RUNTIME_ENV=/workspace/projects/long-rl/verl/trainer/runtime_env.yaml

MODEL_PATH=/share/models/dancegrpo/HunyuanVideo
VIDEOALIGN_CKPT_PATH=/share/models/dancegrpo/videoalign_ckpt
VIDEOALIGN_BASE_MODEL_PATH=/workspace/models/Qwen2-VL-2B-Instruct
DATA_JSON_PATH=/share/models/dancegrpo/rl_embeddings/videos2caption.json

ray job submit --address="${RAY_ADDRESS}" \
  --working-dir "${WORKING_DIR}" \
  --runtime-env "${RUNTIME_ENV}" \
  -- \
  python3 -m verl.trainer.main_ppo \
    --config-path=/workspace/projects/long-rl/examples/diffusion \
    --config-name=config_video_diffusion_case4_dance_nodes \
    hydra.job.chdir=false \
    trainer.project_name=dance_case4_nodes \
    trainer.experiment_name=case4_dance_nodes \
    trainer.nnodes=2 \
    trainer.n_gpus_per_node=8 \
    trainer.max_train_steps=4 \
    data.train_batch_size=32 \
    data.gen_batch_size=32 \
    data.data_json_path=${DATA_JSON_PATH} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.extra.dance.pretrained_model_name_or_path=${MODEL_PATH} \
    actor_rollout_ref.actor.extra.dance.vae_model_path=${MODEL_PATH} \
    actor_rollout_ref.actor.extra.dance.videoalign_ckpt_path=${VIDEOALIGN_CKPT_PATH} \
    +actor_rollout_ref.actor.extra.dance.videoalign_base_model_name_or_path=${VIDEOALIGN_BASE_MODEL_PATH}
