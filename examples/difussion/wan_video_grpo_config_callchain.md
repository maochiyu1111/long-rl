# WAN 视频 GRPO 配置调用链说明

本文解释 `examples/new_supports/wan_video_grpo.sh` 中的 `config` 如何在代码中生效，并给出完整调用链条与关键配置项的映射关系，便于理解、排查与二次改动。

## 顶层入口
- Shell 脚本：`examples/new_supports/wan_video_grpo.sh` 使用
  ```bash
  python3 -m verl.trainer.main \
      config=examples/config_video_diffusion.yaml \
      worker.actor.model.trust_remote_code=true \
      worker.rollout.trust_remote_code=true \
      worker.actor.model.model_path=/workspace/models/Wan2.1-T2V-1.3B-Diffusers \
      trainer.experiment_name=video_generation_grpo \
      trainer.n_gpus_per_node=8
  ```
- 作用：通过 CLI 将 `config=...` 指向 YAML，并覆盖部分字段（模型路径、信任远程代码、实验名、GPU 数）。

## 配置加载与合并
- CLI 解析与合并：`verl/trainer/main.py:101-112`
  - `OmegaConf.from_cli()` 读取 CLI 键值，如 `worker.actor.model.model_path`。
  - 读取 `examples/config_video_diffusion.yaml` 并与默认 `PPOConfig` 合并。
  - 最终 `ppo_config` 为 Python 对象，随后执行 `deep_post_init()`。
- 关联填充：`verl/trainer/config.py:173-181`
  - 将 `data.max_prompt_length/max_response_length` 写入 `worker.rollout.prompt_length/response_length`。
  - 将 `algorithm` 的 KL 配置复制到 `worker.actor`（如 `kl_penalty`、`kl_coef`）。

## Ray 初始化与 Runner
- Ray 环境：`verl/trainer/main.py:114-125` 设置 `TOKENIZERS_PARALLELISM`、`NCCL_DEBUG` 等后 `ray.init(...)`。
- Runner 执行：`verl/trainer/main.py:31-99`
  - 打印配置 `print(json.dumps(config...))`。
  - 根据 `config.trainer.diffusion` 构造 `tokenizer` 与 `processor`：`main.py:40-55`。
  - 创建数据集与 DataLoader：`main.py:83-84` 调用 `create_dataloader(...)`。
  - 实例化 `RayPPOTrainer` 并启动训练：`main.py:85-99`。

## Diffusion 模式开关
- 开关来源：`examples/config_video_diffusion.yaml` 含 `trainer.diffusion: true`、`worker.actor.diffusion: true`、`worker.rollout.diffusion: true`。
- Trainer 感知：`verl/trainer/ray_trainer.py:205` 将 `self.diffusion = config.trainer.diffusion`。
- Worker 选择：`verl/workers/fsdp_workers.py:526-567` 根据 `self.diffusion` 走扩散模型构建与扩散式 rollout（Wan 或 SD3），否则走 vLLM 文本路径。

## 数据集与批次生成
- Dataloader 构造：`verl/trainer/data_loader.py:26-71`
  - 训练/验证数据集使用 `RLHFDataset`，批大小取 `data.rollout_batch_size` 或 `data.mini_rollout_batch_size`。
- Diffusion 下样本构造：`verl/utils/dataset.py:343-364`
  - 若处理器为 `WanProcessor`，对 `example[text]` 生成 `prompt_embeds` 与 `negative_prompt_embeds`，并写入样本字典。
- Wan 文本嵌入：`verl/utils/wan_processor.py:133-183`
  - 使用 UMT5 编码器与 Tokenizer 生成 `prompt_embeds`（bfloat16），供扩散管线与后续 logprob 计算使用。
- 生成批次：`verl/trainer/ray_trainer.py:499-590`
  - Diffusion 分支仅携带嵌入做生成：`ray_trainer.py:512-526`。

## Rollout 执行（Wan）
- Rollout 选择与初始化：`verl/workers/fsdp_workers.py:554-567`
  - `model_path` 包含 `wan` 时，创建 `WanRollout`。
- 生成函数：`verl/workers/rollout/diffusion_rollout.py:190-247`
  - 调用 `wan_pipeline_with_logprob(...)`，使用如下配置项：
    - `num_inference_steps = worker.rollout.num_steps`（`examples/config_video_diffusion.yaml:80`）
    - `guidance_scale = worker.rollout.guidance_scale`（`examples/config_video_diffusion.yaml:78`）
    - `height = worker.rollout.height`（`examples/config_video_diffusion.yaml:83`）
    - `width = worker.rollout.width`（`examples/config_video_diffusion.yaml:84`）
    - `num_frames = worker.rollout.num_frames`（`examples/config_video_diffusion.yaml:82`）
  - 返回 `videos/latents/timesteps/old_log_probs/kl` 封装为 `DataProto`。

## 奖励计算
- 批奖励管理：`verl/workers/reward/function.py:109-149`
  - Diffusion 路径直接对 `images` 或 `videos` 批量评分。
- 评分函数：`examples/reward_function/diffusion.py:51-95`
  - 对视频采样帧，用 BRISQUE 估计画质，归一化为 `accuracy` 并作为 `overall`（可加格式权重）。
- 配置位置：`examples/config_video_diffusion.yaml:99-103` 指向该 reward 函数。

## 参考策略 log_prob（Ref Policy）
- Ref Actor 构建：`verl/workers/fsdp_workers.py:589-634`
  - 以 `config.ref` 创建参考策略 Actor（不训练）。
- 调度器设置：`verl/workers/actor/dp_actor.py:229-234`
  - Diffusion 路径下，若 `config.scheduler` 为空，默认取 `os.path.join(config.model.model_path, "scheduler")`；
  - YAML 中可为 `ref` 显式指定：`examples/config_video_diffusion.yaml:86-98`。
- 参考 logprob 计算：`verl/workers/actor/dp_actor.py:535-568`
  - 调用 `compute_log_prob_flow_grpo(...)`：`verl/workers/diffusion_helper.py:547-646`，沿时序用调度器步进估计每步 logprob 与均值。

## 优势计算（Diffusion）
- 在 Trainer 中：`verl/trainer/ray_trainer.py:154-166`
  - Diffusion 目前支持 GRPO：将 `token_level_rewards` 视为标量，构造 `advantages/returns`。

## 策略更新（Actor）
- 更新入口：`verl/workers/actor/dp_actor.py:660-686+`
  - 构造 mini/micro-batches，读取 `old_log_probs`（来自 rollout）、`ref_log_probs`（来自上一步）、`advantages` 等，累计优化。
- PPO 相关损失与 KL：`verl/trainer/core_algos.py:274-354`（`compute_policy_loss`）、`verl/trainer/core_algos.py:466-499`（`compute_kl`，含 `flow_grpo`）。

## 关键配置项映射
- Data 侧：`examples/config_video_diffusion.yaml:1-21`
  - `rollout_batch_size` 控制训练批大小：`verl/trainer/data_loader.py:57-61`。
  - `max_prompt_length/max_response_length` 经 `post_init` 传入 `worker.rollout`：`verl/trainer/config.py:173-181`。
- Worker.Rollout：`verl/workers/rollout/config.py:19-65`
  - `diffusion/guidance_scale/num_steps/height/width/num_frames` 直连 `WanRollout.generate_sequences(...)`：`diffusion_rollout.py:199-210`。
- Worker.Actor：`examples/config_video_diffusion.yaml:34-64`
  - `diffusion: true` 触发 `dp_actor` 的扩散分支与调度器初始化：`dp_actor.py:225-234`。
  - `guidance_scale` 在 WAN logprob 与采样中使用：`verl/workers/diffusion_helper.py:782-865`、`618-646`。
  - `scheduler`（可在 `ref` 中指定）用于 Flow-Match Euler：`dp_actor.py:233-234`。
- Algorithm：`examples/config_video_diffusion.yaml:22-33`
  - `adv_estimator: grpo` 要求 `worker.rollout.n > 1`（多样本）检查：`verl/trainer/ray_trainer.py:246-250`。
  - `kl_penalty: flow_grpo` 走扩散自适配 KL：`verl/trainer/core_algos.py:466-499`。
  - `use_kl_loss` 决定 KL 加入 Reward 或作为单独损失：`verl/trainer/ray_trainer.py:669-676`。
- Trainer：`examples/config_video_diffusion.yaml:104-122`
  - `n_gpus_per_node` 与 `nnodes` 用于资源池：`verl/trainer/main.py:64-71`。

## CLI 覆盖的字段（来自 wan_video_grpo.sh）
- `worker.actor.model.model_path`：模型根路径，决定走 WAN 分支与子资源加载（含 `text_encoder`、`tokenizer`、`scheduler`）。
- `worker.actor.model.trust_remote_code` 与 `worker.rollout.trust_remote_code`：影响 HuggingFace 加载行为（WAN/SD3 均依赖 HF 组件）。
- `trainer.experiment_name`、`trainer.n_gpus_per_node`：用于日志与资源。

## 生成链条总览（顺序）
1. Shell 传入 YAML 与覆盖项 → `main()` 合并配置（`main.py:101-112`）
2. `Runner.run` 构造 tokenizer/processor（`main.py:40-55`）与 Dataloader（`data_loader.py:26-71`）
3. `RayPPOTrainer.init_workers` 建 WorkerGroup（`ray_trainer.py:263-306`），FSDPWorker 依据 `diffusion` 选择 WAN 路径（`fsdp_workers.py:526-567`）
4. 读样本（含 `prompt_embeds`）→ 生成视频与时序数据（`diffusion_rollout.py:190-247`）
5. 计算奖励（`reward/function.py:109-149` → `examples/reward_function/diffusion.py:51-95`）
6. 参考策略 logprob（`dp_actor.py:535-568` → `diffusion_helper.py:547-646`）
7. 优势（`ray_trainer.py:154-166`）与 KL/损失（`core_algos.py:274-354, 466-499`）
8. 更新 Actor/Critic（`dp_actor.py:660-686+`, `dp_critic.py:171-205`）并记录指标（`metrics.py:24-141`）

## 备注与排错要点
- `rollout.n` 必须大于 1 才能使用 GRPO：`ray_trainer.py:246-250`。
- 若未显式给 `scheduler`，Actor 默认使用 `model_path/scheduler`：`dp_actor.py:229-234`；Ref 可在 YAML 明确设置：`examples/config_video_diffusion.yaml:90`。
- `data.rollout_batch_size` 与各 micro/global batch 的整除约束会在 `RayPPOTrainer.__init__` 内严格校验：`ray_trainer.py:224-243`。

## 附：示例配置中的关键条目
- `examples/config_video_diffusion.yaml:10` 定义 `rollout_batch_size: 16`（注释：等价于 verl 的 `data.train_batch_size`）。

---
以上即 WAN 视频 GRPO 训练的配置调用链与关键项生效路径，覆盖从入口脚本到生成、奖励、优势与更新的全流程节点与代码位置。