#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MODEL_ROOT=${MODEL_ROOT:-/home/qzy/models}
FLUX_MODEL_PATH=${FLUX_MODEL_PATH:-${MODEL_ROOT}/flux}
DATA_JSON_PATH=${DATA_JSON_PATH:-${MODEL_ROOT}/flux/rl_embeddings/videos2caption.json}
OPEN_CLIP_CKPT_PATH=${OPEN_CLIP_CKPT_PATH:-${MODEL_ROOT}/open_clip_pytorch_model.bin}
HPSV2_CKPT_PATH=${HPSV2_CKPT_PATH:-${MODEL_ROOT}/HPS_v2.1_compressed.pt}
REF_SCHEDULER_PATH=${REF_SCHEDULER_PATH:-${MODEL_ROOT}/Wan2.1-T2V-1.3B-Diffusers/scheduler}
REPORT_DIR=${REPORT_DIR:-/workspace/projects/long-rl/outputs/single/flux_case4_step_timing}

export OPEN_CLIP_CKPT_PATH
export HPSV2_CKPT_PATH

python3 -m verl.trainer.main_flux \
  --config-path=/workspace/projects/long-rl/examples/diffusion \
  --config-name=config_video_diffusion_case4_flux \
  hydra.job.chdir=false \
  trainer.project_name=flux_case4 \
  trainer.experiment_name=case4_flux \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=8 \
  trainer.max_train_steps=3 \
  algorithm.adv_estimator=grpo \
  data.train_batch_size=32 \
  data.cfg=0.0 \
  data.dataloader_num_workers=8 \
  data.gen_batch_size=16 \
  data.data_json_path=${DATA_JSON_PATH} \
  actor_rollout_ref.model.path=${FLUX_MODEL_PATH} \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.gradient_accumulation_steps=8 \
  actor_rollout_ref.actor.grad_clip=0.01 \
  actor_rollout_ref.rollout.sampling_steps=16 \
  actor_rollout_ref.rollout.num_frames=1 \
  actor_rollout_ref.rollout.height=720 \
  actor_rollout_ref.rollout.width=720 \
  actor_rollout_ref.rollout.num_generations=2 \
  actor_rollout_ref.rollout.bestofn=2 \
  actor_rollout_ref.rollout.guidance_scale=3.5 \
  actor_rollout_ref.rollout.shift=3.0 \
  actor_rollout_ref.rollout.eta=0.3 \
  actor_rollout_ref.rollout.fps=8 \
  actor_rollout_ref.rollout.use_same_noise=true \
  actor_rollout_ref.rollout.use_group=true \
  actor_rollout_ref.actor.extra.dance.pretrained_model_name_or_path=${FLUX_MODEL_PATH} \
  actor_rollout_ref.actor.extra.dance.vae_model_path=${FLUX_MODEL_PATH} \
  actor_rollout_ref.actor.extra.dance.timestep_fraction=1.0 \
  actor_rollout_ref.actor.extra.dance.master_weight_type=fp32 \
  actor_rollout_ref.actor.extra.dance.use_hpsv2=true \
  actor_rollout_ref.actor.extra.dance.use_pickscore=false \
  actor_rollout_ref.ref.scheduler=${REF_SCHEDULER_PATH} \
  +trainer.step_timing_report_dir=${REPORT_DIR} \
  "$@"
