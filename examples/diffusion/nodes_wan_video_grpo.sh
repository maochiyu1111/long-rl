#!/bin/bash
set -x
wandb disabled
export PYTHONUNBUFFERED=1

MODEL_PATH=/share/models/Wan2.1-T2V-1.3B-Diffusers



ray job submit --address="192.158.0.11:6379" \
  --working-dir /workspace/projects/verl-disaggregate \
  -- \
  python3 -m verl.trainer.main_ppo \
    --config-path=/workspace/projects/verl-disaggregate/examples/diffusion \
    --config-name=config_video_diffusion \
    hydra.job.chdir=false \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.trust_remote_code=true \
    trainer.experiment_name=video_generation_grpo \
    trainer.nnodes=2 \
    trainer.n_gpus_per_node=8 \
    2>&1 | tee video_grpo.log
