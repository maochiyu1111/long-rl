#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export RAY_ADDRESS=${RAY_ADDRESS:-10.244.2.183:6379}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
export VERL_SOCKET_IFACE_PREFIX=${VERL_SOCKET_IFACE_PREFIX:-10.244.2.}

LOG_DIR=${LOG_DIR:-/workspace/projects/long-rl/logs}
mkdir -p "${LOG_DIR}"
RUN_TAG=${RUN_TAG:-$(date -u +%Y%m%d_%H%M%S)}
LOG_PATH="${LOG_DIR}/test_dance_gpu_${RUN_TAG}.log"
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "Logging to ${LOG_PATH}"
echo "Using RAY_ADDRESS=${RAY_ADDRESS}"
echo "Using NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
echo "Using GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME}"

MODEL_PATH=${MODEL_PATH:-/share/models/dancegrpo/HunyuanVideo}
VIDEOALIGN_CKPT_PATH=${VIDEOALIGN_CKPT_PATH:-/share/models/dancegrpo/videoalign_ckpt}
VIDEOALIGN_BASE_MODEL_PATH=${VIDEOALIGN_BASE_MODEL_PATH:-/workspace/models/Qwen2-VL-2B-Instruct}
DATA_JSON_PATH=${DATA_JSON_PATH:-/share/models/dancegrpo/rl_embeddings/videos2caption.json}

python3 -m verl.trainer.main_ppo \
  --config-path=/workspace/projects/long-rl/examples/diffusion \
  --config-name=config_video_diffusion_case4_dance \
  hydra.job.chdir=false \
  trainer.project_name=dance_case4 \
  trainer.experiment_name=test_dance_gpu_rollout_only_1card \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=1 \
  trainer.dist_master_addr=10.244.2.183 \
  trainer.max_train_steps=1 \
  +algorithm.seed=1234 \
  data.train_batch_size=1 \
  data.gen_batch_size=1 \
  data.data_json_path=${DATA_JSON_PATH} \
  actor_rollout_ref.model.path=${MODEL_PATH} \
  actor_rollout_ref.actor.ppo_mini_batch_size=1 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.extra.dance.pretrained_model_name_or_path=${MODEL_PATH} \
  actor_rollout_ref.actor.extra.dance.vae_model_path=${MODEL_PATH} \
  actor_rollout_ref.actor.extra.dance.master_weight_type=bf16 \
  actor_rollout_ref.actor.extra.dance.use_videoalign=false \
  +actor_rollout_ref.actor.extra.dance.rollout_only=true \
  actor_rollout_ref.actor.extra.dance.videoalign_ckpt_path=${VIDEOALIGN_CKPT_PATH} \
  +actor_rollout_ref.actor.extra.dance.videoalign_base_model_name_or_path=${VIDEOALIGN_BASE_MODEL_PATH} \
  +actor_rollout_ref.actor.extra.dance.enable_finite_checks=true \
  +actor_rollout_ref.actor.extra.dance.finite_check_sync=true \
  +actor_rollout_ref.actor.extra.dance.log_tensor_stats=true \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.num_generations=1 \
  actor_rollout_ref.rollout.bestofn=1

RUN_EXIT_CODE=$?

set +x

FIRST_FINITE_CHECK=$(
  grep -m1 'detected non-finite tensor at ' "${LOG_PATH}" \
    | sed -E 's/.*detected non-finite tensor at ([^|]+).*/\1/' \
    || true
)

if [[ -n "${FIRST_FINITE_CHECK}" ]]; then
  echo "TEST_RESULT: FAIL"
  echo "FIRST_NAN_STAGE: ${FIRST_FINITE_CHECK}"
  exit 1
fi

if [[ ${RUN_EXIT_CODE} -eq 0 ]]; then
  echo "TEST_RESULT: PASS"
  echo "FIRST_NAN_STAGE: NONE"
  echo "SUMMARY: no NaN detected in rollout-only test"
  exit 0
fi

echo "TEST_RESULT: FAIL"
echo "FIRST_NAN_STAGE: UNKNOWN"
echo "SUMMARY: run failed without finite-check hit, inspect ${LOG_PATH}"
exit "${RUN_EXIT_CODE}"
