#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VERL_SOCKET_IFACE_PREFIX=192.168.0.

RAY_ADDRESS=http://192.168.0.85:8265
RAY_HEAD_HOST=192.168.0.85
WORKING_DIR=/home/qzy/project/long-rl
RUNTIME_ENV=/home/qzy/project/long-rl/verl/trainer/runtime_env.yaml

MODEL_PATH=/home/qzy/models/HunyuanVideo
VIDEOALIGN_CKPT_PATH=/home/qzy/models/videoalign_ckpt
VIDEOALIGN_BASE_MODEL_PATH=/home/qzy/models/Qwen2-VL-2B-Instruct
DATA_JSON_PATH=/home/qzy/models/rl_embeddings_128/videos2caption.json
REPORT_DIR=${REPORT_DIR:-/home/qzy/project/long-rl/outputs/nodes/dance_case4_step_timing}

export NO_PROXY="${RAY_HEAD_HOST},127.0.0.1,localhost${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="${RAY_HEAD_HOST},127.0.0.1,localhost${no_proxy:+,${no_proxy}}"

env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
ray job submit --address="${RAY_ADDRESS}" \
  --working-dir "${WORKING_DIR}" \
  --runtime-env "${RUNTIME_ENV}" \
  -- \
  python3 -m verl.trainer.main_ppo \
    --config-path=/home/qzy/project/long-rl/examples/diffusion \
    --config-name=config_video_diffusion_case4_dance_nodes_npu \
    hydra.job.chdir=false \
    trainer.project_name=dance_case4 \
    trainer.experiment_name=case4_dance_npu \
    trainer.nnodes=2 \
    trainer.n_gpus_per_node=8 \
    trainer.max_train_steps=1 \
    trainer.device=npu \
    data.train_batch_size=32 \
    data.gen_batch_size=32 \
    data.data_json_path=${DATA_JSON_PATH} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.extra.dance.pretrained_model_name_or_path=${MODEL_PATH} \
    actor_rollout_ref.actor.extra.dance.vae_model_path=${MODEL_PATH} \
    actor_rollout_ref.actor.extra.dance.videoalign_ckpt_path=${VIDEOALIGN_CKPT_PATH} \
    actor_rollout_ref.actor.extra.dance.videoalign_base_model_name_or_path=${VIDEOALIGN_BASE_MODEL_PATH} \
    +trainer.step_timing_report_dir=${REPORT_DIR}
