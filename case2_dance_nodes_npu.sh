#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VERL_SOCKET_IFACE_PREFIX=${VERL_SOCKET_IFACE_PREFIX:-192.168.0.}

RAY_ADDRESS=http://192.168.0.63:8265
RAY_HEAD_HOST=192.168.0.63
DIST_MASTER_ADDR=null
DIST_MASTER_PORT=null
WORKING_DIR=/home/qzy/project/long-rl
RUNTIME_ENV=/home/qzy/project/long-rl/verl/trainer/runtime_env_npu.yaml
REPORT_DIR=${REPORT_DIR:-/home/qzy/project/long-rl/outputs/nodes/dance_case2_step_timing}

MODEL_PATH=/home/qzy/models/HunyuanVideo
VIDEOALIGN_CKPT_PATH=/home/qzy/models/videoalign_ckpt
VIDEOALIGN_BASE_MODEL_PATH=/home/qzy/models/Qwen2-VL-2B-Instruct
DATA_JSON_PATH=/home/qzy/models/rl_embeddings_128/videos2caption.json

export NO_PROXY="${RAY_HEAD_HOST},127.0.0.1,localhost${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="${RAY_HEAD_HOST},127.0.0.1,localhost${no_proxy:+,${no_proxy}}"

env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
ray job submit --address="${RAY_ADDRESS}" \
  --working-dir "${WORKING_DIR}" \
  --runtime-env "${RUNTIME_ENV}" \
  -- \
  python3 -m verl.trainer.main_ppo \
    --config-path=/home/qzy/project/long-rl/examples/diffusion \
    --config-name=config_video_diffusion_case2_dance \
    hydra.job.chdir=false \
    trainer.project_name=dance_case2_nodes_npu \
    trainer.experiment_name=case2_dance_nodes_npu \
    trainer.device=npu \
    trainer.nnodes=2 \
    trainer.n_gpus_per_node=8 \
    trainer.disaggregate_actor_n_gpus_per_node=8 \
    trainer.disaggregate_rollout_ref_n_gpus_per_node=8 \
    trainer.disaggregate_actor_nnodes=1 \
    trainer.disaggregate_rollout_ref_nnodes=1 \
    trainer.dist_master_addr=${DIST_MASTER_ADDR} \
    trainer.dist_master_port=${DIST_MASTER_PORT} \
    trainer.max_train_steps=1 \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=32 \
    data.cfg=0.0 \
    data.dataloader_num_workers=8 \
    data.gen_batch_size=64 \
    data.data_json_path=${DATA_JSON_PATH} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.gradient_accumulation_steps=4 \
    actor_rollout_ref.rollout.sampling_steps=20 \
    actor_rollout_ref.rollout.num_steps=28 \
    actor_rollout_ref.rollout.num_frames=53 \
    actor_rollout_ref.rollout.height=352 \
    actor_rollout_ref.rollout.width=352 \
    actor_rollout_ref.rollout.num_generations=8 \
    actor_rollout_ref.rollout.bestofn=4 \
    actor_rollout_ref.rollout.vq_coef=1.0 \
    actor_rollout_ref.rollout.mq_coef=0.0 \
    actor_rollout_ref.rollout.shift=5 \
    actor_rollout_ref.rollout.eta=0.25 \
    actor_rollout_ref.rollout.fps=8 \
    actor_rollout_ref.rollout.use_same_noise=true \
    actor_rollout_ref.rollout.use_group=true \
    actor_rollout_ref.actor.extra.dance.pretrained_model_name_or_path=${MODEL_PATH} \
    actor_rollout_ref.actor.extra.dance.vae_model_path=${MODEL_PATH} \
    actor_rollout_ref.actor.extra.dance.videoalign_ckpt_path=${VIDEOALIGN_CKPT_PATH} \
    actor_rollout_ref.actor.extra.dance.use_videoalign=true \
    actor_rollout_ref.actor.extra.dance.timestep_fraction=0.6 \
    +actor_rollout_ref.actor.extra.dance.videoalign_base_model_name_or_path=${VIDEOALIGN_BASE_MODEL_PATH} \
    actor_rollout_ref.actor.extra.dance.master_weight_type=bf16 \
    +trainer.step_timing_report_dir=${REPORT_DIR}
