# Case 4（long-rl）：基于当前代码的实际执行流程

本文按现在仓库里已经能跑通的 `case4_dance.sh` 来写，目标是把 Case4 在 `long-rl` 里的真实执行链路串清楚。

当前主链路不是通用 diffusion PPO 流，而是一个专门的 `dance_case4_mode` 分支。

## 1. 入口：`case4_dance.sh`

实际入口命令是：

```bash
bash case4_dance.sh
```

脚本会先做 4 件事：

1. 设置环境变量：
   - `PYTHONUNBUFFERED=1`
   - `HF_HUB_OFFLINE=1`
   - `TRANSFORMERS_OFFLINE=1`
   - `N_GPUS_VIDEO_INIT=${N_GPUS_VIDEO_INIT:-0}`
2. 定位配置文件：
   - `CONFIG_PATH=${ROOT_DIR}/examples/diffusion`
   - `CONFIG_NAME=config_video_diffusion_case4_dance`
3. 根据 `NNODES`、`N_GPUS_PER_NODE`、`PPO_MINI_BATCH_SIZE`、`PPO_MICRO_BATCH_SIZE_PER_GPU`、`TRAIN_BATCH_SIZE`、`GEN_BATCH_SIZE` 做一轮参数兼容性检查。
4. 把 override 拼好后执行：

```bash
python3 -m verl.trainer.main_ppo \
  --config-path="${CONFIG_PATH}" \
  --config-name="${CONFIG_NAME}" \
  "${OVERRIDES[@]}" \
  "$@"
```

脚本默认会覆盖这些关键项：

- `hydra.job.chdir=false`
- `trainer.project_name`
- `trainer.experiment_name`
- `trainer.nnodes`
- `trainer.n_gpus_per_node`
- `trainer.max_train_steps`
- `data.train_batch_size`
- `data.gen_batch_size`
- `actor_rollout_ref.actor.ppo_mini_batch_size`
- `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu`

可选覆盖项：

- `MODEL_PATH`
- `VAE_MODEL_PATH`
- `DATA_JSON_PATH`
- `N_GPUS_VIDEO_INIT`

当前脚本里这两个路径是直接写死的，不是通过 `${VAR:-default}` 读取外部环境变量：

- `VIDEOALIGN_CKPT_PATH=/share/models/dancegrpo/videoalign_ckpt`
- `VIDEOALIGN_BASE_MODEL_PATH=/workspace/models/Qwen2-VL-2B-Instruct`

## 2. 配置命中的是哪条分支

`examples/diffusion/config_video_diffusion_case4_dance.yaml` 的关键配置是：

- `trainer.diffusion: true`
- `trainer.disaggregate: false`
- `trainer.pipelined_micro_batch: false`
- `algorithm.adv_estimator: grpo`
- `actor_rollout_ref.actor.dance_case4_mode: true`
- `actor_rollout_ref.actor.extra.diffusion: true`
- `actor_rollout_ref.actor.extra.dance.model_type: hunyuan_hf`
- `actor_rollout_ref.actor.extra.dance.use_videoalign: true`

因此当前 Case4 会命中专门的硬分支，前提校验是：

- `trainer.diffusion == true`
- `trainer.disaggregate == false`
- `trainer.pipelined_micro_batch == false`
- `algorithm.adv_estimator == grpo`

只要其中任意一项不满足，trainer 和 worker 两侧都会直接报错，不会退回通用 PPO/diffusion 训练逻辑。

## 3. 从脚本到 trainer 的总链路

真实调用顺序可以概括成：

```text
case4_dance.sh
  -> python -m verl.trainer.main_ppo
  -> run_ppo()
  -> TaskRunner.run()
  -> RayPPOTrainer(...)
  -> RayPPOTrainer.init_workers()
  -> ActorRolloutRefWorker.init_model()
  -> ActorRolloutRefWorker._build_model_optimizer_dance()
  -> RayPPOTrainer.fit()
  -> RayPPOTrainer.fit_dance_case4()
  -> actor_rollout_wg.generate_sequences()
  -> ActorRolloutRefWorker._generate_sequences_dance()
  -> actor_rollout_wg.update_actor()
  -> ActorRolloutRefWorker._update_actor_dance()
```

## 4. `main_ppo.py` 这一层实际做了什么

### 4.1 `run_ppo()`

`run_ppo()` 先 `ray.init(...)`，再创建 `TaskRunner`，最后远程执行 `TaskRunner.run(config)`。

### 4.2 `TaskRunner.run()`

这里有几个和 Case4 直接相关的分支：

1. `_build_tokenizer_and_processor(config)`  
   因为 `dance_case4_mode=true`，这里直接返回 `(None, None)`，不会走通用 diffusion tokenizer/processor 初始化。

2. reward manager  
   因为 `dance_case4_mode=true`，这里会直接设：
   - `reward_fn = None`
   - `val_reward_fn = None`

3. dataset/collate/sampler  
   同样因为 `dance_case4_mode=true`，这里传给 `RayPPOTrainer` 的是：
   - `train_dataset = None`
   - `val_dataset = None`
   - `collate_fn = None`
   - `train_sampler = None`

4. 然后创建 `RayPPOTrainer(...)`，并执行：
   - `trainer.init_workers()`
   - `trainer.fit()`

也就是说，Case4 不依赖 `load_reward_manager()`，而是把奖励留在 worker 内部的 dance 专用分支里做。

## 5. `RayPPOTrainer` 初始化阶段

### 5.1 `RayPPOTrainer.__init__()`

构造函数里会先做两件事：

1. `_validate_config()`
2. `_create_dataloader(...)`

### 5.2 `_create_dance_case4_dataloader()`

由于 `dance_case4_mode=true`，`_create_dataloader(...)` 不会创建通用 RLHF dataset，而是直接走：

```text
_create_dance_case4_dataloader()
  -> LatentDataset(data_json_path, num_latent_t=t, cfg_rate=cfg)
  -> StatefulDataLoader(..., collate_fn=latent_collate_function)
```

这里有两个很重要的实际行为：

1. 必须提供 `data.data_json_path`，否则直接报错。
2. dataloader 的 `batch_size` 取的是：
   - `data.gen_batch_size`
   - 如果没配，才回退到 `data.train_batch_size`

Case4 的 dataloader 输出必须是三元组：

```text
(encoder_hidden_states, encoder_attention_mask, caption)
```

## 6. Worker 初始化阶段

### 6.1 `init_workers()`

在当前配置里：

- 不走 `fit_dis()`
- 不走 encoder/llm split
- 不走 async rollout

所以 `init_workers()` 会创建一个 colocated 的 `actor_rollout_wg`，角色是 `actor_rollout`。

然后根据 `N_GPUS_VIDEO_INIT` / `trainer.n_gpus_video_init` 选择初始化方式：

- `0`：整组直接 `init_model()`
- `1`：串行逐卡 `init_model()`
- `1 < n < world_size`：分批 `init_model()`

### 6.2 `ActorRolloutRefWorker.init_model()`

这里是当前 Case4 最关键的分叉点：

- 如果 `dance_case4_mode=true` 且条件满足：
  - 直接走 `_build_model_optimizer_dance()`
  - 然后 `return`
- 不会走通用的：
  - `_build_model_optimizer()`
  - `_build_model_optimizer_diffusion()`
  - `_build_rollout_diffusion()`

### 6.3 `_build_model_optimizer_dance()`

这个函数会完成当前 Case4 需要的全部模型初始化：

1. 读取 `actor.extra.dance` 配置。
2. 如果设置了 `algorithm.seed`，先 `set_seed(...)`。
3. 按 `model_type` 和 `pretrained_model_name_or_path` 加载生成 transformer。
4. 用 fastvideo 的 FSDP 工具把 transformer 包起来。
5. 创建 `AdamW` 和 lr scheduler。
6. 加载 VAE。
7. 如果 `use_videoalign=true`，额外构造 `VideoVLMRewardInference`。

当前配置里，实际加载的是：

- 生成模型：`hunyuan_hf`
- transformer：`HunyuanVideoTransformer3DModel.from_pretrained(..., subfolder="transformer")`
- VAE：`AutoencoderKLHunyuanVideo.from_pretrained(..., subfolder="vae")`

所以和旧文档不同，Hunyuan 路径现在已经在这条 Case4 主链路里实际接通了。

## 7. 训练阶段不是通用 `fit()`，而是 `fit_dance_case4()`

`trainer.fit()` 一进来就会先检查 `dance_case4_mode`。

如果为真，并且 Case4 条件满足，那么直接：

```text
fit()
  -> fit_dance_case4()
```

不会进入下面这些通用步骤：

- 通用 diffusion `gen_batch` 构造
- `compute_reward(...)`
- `compute_advantage_diffusion(...)`
- 通用 actor PPO update 流

### 7.1 `fit_dance_case4()` 的单步流程

每一步实际就是：

```text
next(train_dataloader)
  -> 拆成 (encoder_hidden_states, encoder_attention_mask, caption)
  -> DataProto.from_single_dict(
       {
         "encoder_hidden_states": ...,
         "encoder_attention_mask": ...
       },
       meta_info={"caption": caption},
     )
  -> actor_rollout_wg.generate_sequences(new_batch)
  -> actor_rollout_wg.update_actor(rollout_batch)
```

这里的 `caption` 不在 tensor 里，而是放进 `meta_info` 传给 worker。

## 8. `generate_sequences()` 实际走的是 `_generate_sequences_dance()`

### 8.1 输入

worker 从 `prompts` 里读取：

- `encoder_hidden_states`
- `encoder_attention_mask`
- `meta_info["caption"]`

如果 `rollout.use_group=true`，会按 `rollout.num_generations` 把条件和 caption 一起扩成多份。

### 8.2 采样过程

`_generate_sequences_dance()` 会：

1. 构造 `sigma_schedule`
2. 根据视频尺寸和帧数创建初始 latent noise
3. 逐 timestep 调用 transformer 前向
4. 在 `flux_step(...)` 里计算：
   - 下一步 latent
   - `pred_original_sample`
   - 该步 `log_prob`
5. 保存整条 rollout 轨迹

这里产出的 `log_probs` 就是后面 actor 更新时做 ratio 的旧策略 log-prob。

### 8.3 视频导出和奖励

采样结束后，worker 会：

1. 用 VAE 把最终 latent decode 成视频
2. 导出到：

```text
./videos/hunyuan_{rank}_{index}.mp4
```

3. 如果 `use_videoalign=true` 且 inferencer 初始化成功，就调用：

```text
self.inferencer.reward([video_path], [caption], use_norm=True)
```

得到：

- `VQ`
- `MQ`

否则 `vq_reward/mq_reward` 默认是 `-1.0`。

### 8.4 返回给 driver 的内容

`_generate_sequences_dance()` 返回的 `DataProto` 里包含：

- `timesteps`
- `latents`
- `next_latents`
- `log_probs`
- `vq_rewards`
- `mq_rewards`
- `encoder_hidden_states`
- `encoder_attention_mask`

同时在 `meta_info` 里带上：

- `sigma_schedule`

## 9. `update_actor()` 实际走的是 `_update_actor_dance()`

这一步同样不会进入通用 PPO actor 更新逻辑，而是直接在 worker 内部做 Dance Case4 的更新。

### 9.1 先在 worker 内部算 advantage

`_update_actor_dance()` 会按 `num_generations` 分组，分别对每组做：

- `vq_advantages = (vq_rewards - mean) / (std + 1e-8)`
- `mq_advantages = (mq_rewards - mean) / (std + 1e-8)`

也就是说，当前 Case4 的 advantage 不是 trainer 上的 `compute_advantage_diffusion()` 算的，而是在 worker 里根据 `VQ/MQ` 自己算的。

### 9.2 best-of-n 筛选

接着会根据：

```text
total_scores = vq_coef * vq_advantages + mq_coef * mq_advantages
```

做 `bestofn` 选择。当前实现是：

- 选一半 top
- 选一半 bottom
- 拼起来再打乱

### 9.3 重新计算新策略 log-prob 并做 PPO clip loss

之后会：

1. 对 timestep 做随机打乱
2. 按 `timestep_fraction` 只训练前一部分 timestep
3. 用当前 transformer 重新算 `new_log_probs`
4. 和 rollout 里存下来的 `log_probs` 做 ratio
5. 分别计算 `vq_loss` 和 `mq_loss`
6. 按 `vq_coef/mq_coef` 加权成最终 loss

### 9.4 反向传播与优化

每个 sample 更新时会执行：

- `final_loss.backward()`
- `self.transformer.clip_grad_norm_(...)`
- `self.optimizer.step()`
- `self.lr_scheduler.step()`
- `self.optimizer.zero_grad()`

梯度裁剪优先读取：

- `actor.grad_clip`

如果没配，再回退到：

- `actor.max_grad_norm`

最后返回：

- `meta_info["metrics"]["actor/loss"]`

## 10. 当前 Case4 主流程里明确“不走”的路径

为了避免后面再对着旧文档排查，这里把当前不属于主链路的路径单独列出来：

- 不走 `load_reward_manager()`
- 不走 `BatchRewardManager`
- 不走 `compute_reward(...)`
- 不走 `compute_advantage_diffusion(...)`
- 不走 `verl/workers/rollout/diffusion_rollout.py`
- 不走 `DataParallelPPOActor.update_policy_diffusion()`
- 不走 `_build_model_optimizer_diffusion()`
- 不走 `_build_rollout_diffusion()`

这些都是通用 diffusion / PPO 路径；当前能跑通的 Case4 走的是 `dance_case4_mode` 专用实现。

## 11. 一句话总结

现在仓库里的 Case4 实际执行流程是：

```text
case4_dance.sh
  -> main_ppo
  -> RayPPOTrainer
  -> dance_case4 专用 dataloader
  -> actor_rollout worker 专用初始化（Hunyuan + VAE + 可选 VideoAlign）
  -> fit_dance_case4()
  -> _generate_sequences_dance()
  -> _update_actor_dance()
```

也就是说，当前 `long-rl` 里的 Case4 已经不是“通用 diffusion PPO 上做适配”，而是“在 trainer 和 worker 两侧都显式开了 dance_case4 的专用硬分支”。
