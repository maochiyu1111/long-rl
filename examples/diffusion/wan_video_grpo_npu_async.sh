#!/bin/bash
set -x
wandb disabled
export PYTHONUNBUFFERED=1

MODEL_PATH=/home/qzy/models/Wan2.1-T2V-1.3B-Diffusers

python3 -m verl.trainer.main_ppo \
    --config-path=/home/qzy/project/verl-disaggregate/examples/diffusion \
    --config-name=config_video_diffusion_npu_async \
    hydra.job.chdir=false \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.trust_remote_code=true \
    trainer.experiment_name=video_generation_grpo_async \
    trainer.n_gpus_per_node=8 \
    trainer.device=npu $@
