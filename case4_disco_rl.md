# Case 4: `disaggregate=false`、`pipelined_micro_batch=false`、`disco=false` 时 DanceGRPO 执行链路

## 1. 结论与范围

本文只分析你关心的主训练链路（忽略 `fit()` 里的测试性提前退出逻辑），即：

`next(iter(train_dataloader)) -> DataProto.from_single_dict() -> actor_rollout_ref_wg.generate_sequences(new_batch) -> FSDPWorker.generate_sequences() -> actor_rollout_ref_wg.update_actor(batch) -> FSDPWorker.update_actor()`

在当前代码中，这组参数会走 **colocated 单组 Worker** 路径：

- 不走 `fit_dis()`
- 不走 `fit_disco_pipelined()`
- 走 `fit()`

关键分支位置：

- `verl/trainer/main.py:108-116`
- `verl/trainer/ray_trainer.py:789`

---

## 2. 入口与分支决策

### 2.1 入口调用

1. `main()` 组装配置并启动 Ray  
2. `Runner.run(config)` 创建 `RayPPOTrainer` 并进入训练  

关键位置：

- `verl/trainer/main.py:121`
- `verl/trainer/main.py:34`

### 2.2 参数如何决定执行路径

当 `trainer.disaggregate=false` 时，`Runner.run()` 走 `else` 分支：

- 角色映射为 `Role.ActorRolloutRef` + `Role.Critic`
- `trainer.init_workers()`
- `trainer.fit()`

关键位置：

- `verl/trainer/main.py:58-70`
- `verl/trainer/main.py:114-116`

注意：虽然角色映射里声明了 `Critic`，但 `adv_estimator=grpo` 时 `self.use_critic=False`，实际不会创建 critic worker。

关键位置：

- `verl/trainer/ray_trainer.py:221-224`
- `verl/trainer/ray_trainer.py:290-296`

---

## 3. 初始化阶段调用链

### 3.1 Driver 侧

1. `RayPPOTrainer.init_workers()`
2. 创建资源池与 colocated worker class
3. `spawn()` 后拿到 `self.actor_rollout_ref_wg`
4. 调用 `self.actor_rollout_ref_wg.init_model()`

关键位置：

- `verl/trainer/ray_trainer.py:274-331`

### 3.2 Worker 侧

1. `FSDPWorker.init_model()`（`@register(ONE_TO_ALL)`）
2. 对于 `actor_rollout_ref`，`self.colocated=True`，走 `_build_model_optimizer_dance()`
3. 加载 transformer、FSDP 包装、optimizer、scheduler、VAE、(可选) VideoAlign reward inferencer

关键位置：

- `verl/workers/fsdp_workers.py:1317`
- `verl/workers/fsdp_workers.py:1398-1401`
- `verl/workers/fsdp_workers.py:654-725`

### 3.3 模型载入补充（VideoAlign + Hunyuan）

#### A. VideoAlign 奖励模型如何载入

1. 仅在 `role in {actor_rollout_ref, rollout_ref}` 且 `worker.reward.use_videoalign=true` 时加载。  
2. `_build_model_optimizer_dance()` 内部直接构造：
   `VideoVLMRewardInference("/workspace/DanceGRPO/videoalign_ckpt", device=self.device, dtype=torch.bfloat16)`。  
3. `VideoVLMRewardInference.__init__()` 会：
   - 读取 `model_config.json`
   - 创建 `DataConfig/ModelConfig/PEFTLoraConfig/TrainingConfig`
   - 调 `create_model_and_processor(...)` 构建模型与 processor
   - 调 `load_model_from_checkpoint(...)` 从 `load_from_pretrained` 目录恢复 checkpoint
   - `model.eval().to(device)`，供 rollout 后奖励打分使用
4. 在 `reward` 模型构建函数中，底座 VLM 是 `./Qwen2-VL-2B-Instruct`（本地路径）。

关键位置：

- `verl/workers/fsdp_workers.py:662-669`
- `fastvideo/models/videoalign/inference.py:30-65`
- `fastvideo/models/videoalign/train_reward.py:89-108`

#### B. Hunyuan 生成模型如何载入

1. `_build_model_optimizer_dance()` 调用：
   `load_transformer(model_type, None, pretrained_model_name_or_path, master_weight_type)`。  
2. 当 `model_type="hunyuan_hf"`（示例配置默认）时：
   - `HunyuanVideoTransformer3DModel.from_pretrained(pretrained_model_name_or_path, subfolder="transformer", torch_dtype=...)`
   - 随后被 FSDP 包装成 `self.transformer`
3. VAE 通过 `load_vae(model_type, vae_model_path)` 加载；当 `model_type="hunyuan_hf"` 时：
   - `AutoencoderKLHunyuanVideo.from_pretrained(vae_model_path, subfolder="vae", torch_dtype=torch.float32).to("cuda")`
   - 返回 `fps=24`
4. 当前这条 Case 4 链路并不会在 worker 内加载 Hunyuan 文本编码器；`generate_sequences()` 直接消费 dataloader 提供的 `encoder_hidden_states/encoder_attention_mask`。

关键位置：

- `verl/workers/fsdp_workers.py:670-677`
- `verl/workers/fsdp_workers.py:690`
- `verl/workers/fsdp_workers.py:724`
- `fastvideo/utils/load.py:253-300`
- `fastvideo/utils/load.py:303-345`
- `verl/workers/fsdp_workers.py:1999-2001`

---

## 4. 每个训练 step 的主调用链（函数级）

`fit()` 内每步核心链路：

1. `next(iter(self.train_dataloader))` 取 batch  
2. `DataProto.from_single_dict(batch_dict, meta_info={"caption": caption})`  
3. `self.actor_rollout_ref_wg.generate_sequences(new_batch)`  
4. `self.actor_rollout_ref_wg.update_actor(batch)`  

关键位置：

- `verl/trainer/ray_trainer.py:821-832`

---

## 5. 分发与回收机制（RayWorkerGroup + register）

`generate_sequences` 与 `update_actor` 都是 `Dispatch.DP_COMPUTE_PROTO`：

- 输入 `DataProto` 会被 `chunk(world_size)` 平均切分到各 rank
- 每个 rank 独立执行同名 worker 函数
- 输出通过 `DataProto.concat()` 拼回 driver 侧单个 `DataProto`

关键位置：

- `verl/workers/fsdp_workers.py:1947`
- `verl/workers/fsdp_workers.py:1534`
- `verl/single_controller/base/decorator.py:106-123`
- `verl/protocol.py:540-566`
- `verl/protocol.py:584-600`
- `verl/single_controller/ray/base.py:373-392`

这意味着你在 `fit()` 里看到的一次 `actor_rollout_ref_wg.generate_sequences(...)`，底层是“多 rank 并行 rollout + driver 汇总”。

---

## 6. `FSDPWorker.generate_sequences()` 的 GRPO rollout 细节

函数入口：

- `verl/workers/fsdp_workers.py:1948`

### 6.1 输入整理

1. 从 `prompts.batch` 取：
   - `encoder_hidden_states`
   - `encoder_attention_mask`
2. 从 `prompts.meta_info` 取 `caption`
3. 若 `use_group=true`，按 `num_generations` 对条件与 caption 扩展

关键位置：

- `verl/workers/fsdp_workers.py:1999-2017`

### 6.2 噪声日程与采样循环

1. 构造 `sigma_schedule = linspace(1,0,sampling_steps+1)`  
2. 经过 `sd3_time_shift()` 变换  
3. 对每个样本逐步采样：
   - 前向得到 `model_pred`
   - `flux_step(...)` 计算 `z_{t+1}` 与该步 `log_prob`
   - 保存全路径 latent 与 logprob

关键位置：

- `verl/workers/fsdp_workers.py:2021-2024`
- `verl/workers/fsdp_workers.py:2070-2114`
- `verl/workers/fsdp_workers.py:1954-1993`

`flux_step(grpo=True)` 的核心是用高斯形式计算 `log_prob(prev_sample | prev_sample_mean, std_dev_t)`，这就是后续 PPO ratio 的“旧策略 log_prob”来源。

### 6.3 解码与奖励

1. 取最终 `pred_original` 反标定后进 VAE decode  
2. 导出视频 `./videos/hunyuan_{rank}_{index}.mp4`  
3. 调 `self.inferencer.reward(...)` 得到 `VQ/MQ`（可带归一化）

关键位置：

- `verl/workers/fsdp_workers.py:2111-2131`
- `verl/workers/fsdp_workers.py:2132-2149`
- `fastvideo/models/videoalign/inference.py:227-254`

### 6.4 返回结构

返回 `DataProto` 包含：

- `timesteps`
- `latents`
- `next_latents`
- `log_probs`
- `vq_rewards`
- `mq_rewards`
- `encoder_hidden_states`
- `encoder_attention_mask`
- `meta_info["sigma_schedule"]`

关键位置：

- `verl/workers/fsdp_workers.py:2162-2183`

---

## 7. `FSDPWorker.update_actor()` 的 GRPO/PPO 更新细节

函数入口：

- `verl/workers/fsdp_workers.py:1535`

### 7.1 预处理

1. `optimizer.zero_grad()`
2. 将 `data.batch` 张量搬到当前 device
3. 从 `meta_info["sigma_schedule"]` 还原 tensor

关键位置：

- `verl/workers/fsdp_workers.py:1608-1617`

### 7.2 分组标准化 advantage

按 `num_generations` 分组，对每组分别计算：

- `vq_advantages = (vq_rewards - mean) / std`
- `mq_advantages = (mq_rewards - mean) / std`

关键位置：

- `verl/workers/fsdp_workers.py:1625-1648`

### 7.3 best-of-n 筛选

1. `total_scores = vq_coef * vq_advantages + mq_coef * mq_advantages`
2. 选 top `bestofn/2` + bottom `bestofn/2`
3. 打乱后保留（当 `num_generations != bestofn` 时）

关键位置：

- `verl/workers/fsdp_workers.py:1650-1665`

### 7.4 时间步随机化与训练步数截断

1. 对每个样本的 diffusion timestep 做随机排列 `perms`
2. `train_timesteps = int(T * timestep_fraction)`，只训练前这部分随机步

关键位置：

- `verl/workers/fsdp_workers.py:1666-1676`
- `verl/workers/fsdp_workers.py:1686`

### 7.5 GRPO one-step + PPO clipped objective

对每个样本、每个训练 timestep：

1. `grpo_one_step(...)` 用当前 actor 重算 `new_log_probs`  
2. `ratio = exp(new_log_probs - old_log_probs)`  
3. 分别对 `vq_adv`、`mq_adv` 算 clipped loss  
4. `final_loss = vq_coef * vq_loss + mq_coef * mq_loss`  
5. `final_loss.backward()`

关键位置：

- `verl/workers/fsdp_workers.py:1692-1704`
- `verl/workers/fsdp_workers.py:1710-1733`
- `verl/workers/fsdp_workers.py:1577-1606`

### 7.6 梯度累计与参数更新

每 `gradient_accumulation_steps` 个 sample：

- `clip_grad_norm_`
- `optimizer.step()`
- `lr_scheduler.step()`
- `optimizer.zero_grad()`

关键位置：

- `verl/workers/fsdp_workers.py:1734-1738`

返回：

- `DataProto(non_tensor_batch={"actor_loss": ...})`

关键位置：

- `verl/workers/fsdp_workers.py:1739-1741`

---

## 8. 总体调用图（Case 4）

1. `main -> Runner.run`
2. `Runner.run -> RayPPOTrainer.init_workers -> actor_rollout_ref_wg.init_model -> FSDPWorker.init_model -> _build_model_optimizer_dance`
3. `Runner.run -> RayPPOTrainer.fit`
4. `fit -> DataLoader -> DataProto.from_single_dict`
5. `fit -> actor_rollout_ref_wg.generate_sequences`  
6. `WG dispatch(DP_COMPUTE_PROTO) -> per-rank FSDPWorker.generate_sequences -> concat`
7. `fit -> actor_rollout_ref_wg.update_actor`
8. `WG dispatch(DP_COMPUTE_PROTO) -> per-rank FSDPWorker.update_actor -> concat`
9. `fit` 进入下一 step

---

## 9. 对这条链路的实现语义总结

1. 这是“**rollout 与 actor colocated**”的路径：采样和更新都在同一组 `actor_rollout_ref` worker 上执行。  
2. `generate_sequences` 产出的是扩散轨迹级训练样本（含每步 `log_probs` 与奖励），`update_actor` 再基于这些轨迹做 PPO 风格 GRPO 更新。  
3. 分组标准化 advantage 与 best-of-n 筛选都发生在 worker 本地分片上，然后再由外层聚合返回。  
