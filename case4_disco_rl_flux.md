# Case 4（DIscoRL Flux）：`disaggregate=false`、`pipelined_micro_batch=false` 时 Flux DanceGRPO 执行链路

## 1. 结论与范围

本文只分析 DIscoRL 中 **Flux + DanceGRPO** 在 Case4 参数组合下的主训练链路，即：

```text
flux_grpo_nodes*.sh / python -m verl.trainer.main
  -> Runner.run(config)
  -> RayPPOTrainerDance.init_workers()
  -> FSDPWorkerDance.init_model()
  -> FSDPWorkerDance._build_model_optimizer_dance()
  -> RayPPOTrainerDance.fit()
  -> actor_rollout_ref_wg.generate_sequences(new_batch)
  -> FSDPWorkerDance.generate_sequences()
  -> actor_rollout_ref_wg.update_actor(batch)
  -> FSDPWorkerDance.update_actor()
```

在这组参数下，DIscoRL 会走 **colocated 单组 Worker** 路径：

- 不走 `init_workers_dis()`
- 不走 `fit_dis()`
- 不走 `fit_disco_pipelined()`
- 走 `init_workers() + fit()`

注意：DIscoRL 现成 Flux YAML 主要是 `disaggregate=true + pipelined_micro_batch=false + worker.actor.disco=true`，也就是更接近 Case2/Case3 的开关组合。Case4 需要把 `trainer.disaggregate=false`，且 `worker.actor.disco` 不能参与主路径。

关键位置：

- `/Users/bytedance/codegfile/DIscoRL/verl/trainer/main.py`
- `/Users/bytedance/codegfile/DIscoRL/verl/trainer/ray_trainer_dance.py`
- `/Users/bytedance/codegfile/DIscoRL/verl/workers/fsdp_workers_dance.py`
- `/Users/bytedance/codegfile/DIscoRL/examples/config_flux_dance.yaml`

---

## 2. 入口与分支决策

### 2.1 入口调用

DIscoRL 的 Flux 脚本采用旧式 OmegaConf CLI 风格，典型入口类似：

```bash
python3 -m verl.trainer.main \
  config=examples/config_flux.yaml \
  trainer.model_name=flux \
  ...
```

`verl.trainer.main` 内部流程是：

1. `OmegaConf.from_cli()` 读取命令行。
2. 如果带 `config=...`，加载 YAML 并 merge 到 structured `PPOConfig`。
3. `ray.init(runtime_env=...)`。
4. `Runner.remote()` 后执行 `Runner.run(config)`。

关键位置：

- `verl/trainer/main.py:140-182`

### 2.2 Flux/Dance 如何被选中

`Runner.run()` 先读取：

```text
trainer.grpo_variant
```

当 `trainer.grpo_variant=dance` 时：

- trainer class 选 `RayPPOTrainerDance`
- worker class 选 `FSDPWorkerDance`
- 不走 `RayPPOTrainerFlow`
- 不走 `FSDPWorkerFlow`

关键位置：

- `verl/trainer/main.py:37-52`

### 2.3 Case4 参数如何决定拓扑

当：

```text
trainer.disaggregate=false
trainer.pipelined_micro_batch=false
trainer.model_name=flux
algorithm.adv_estimator=grpo
```

`Runner.run()` 走 colocated 分支：

- 角色映射为 `Role.ActorRolloutRef` + `Role.Critic`
- 资源池只有 `global_pool`
- `Role.ActorRolloutRef` 与 `Role.Critic` 都映射到 `global_pool`
- 创建 `RayPPOTrainerDance`
- 执行 `trainer.init_workers()`
- 执行 `trainer.fit()`

虽然角色映射里声明了 `Critic`，但 GRPO 下 `RayPPOTrainerDance.__init__()` 会令 `self.use_critic=False`，实际主链不依赖 critic worker。

关键位置：

- `verl/trainer/main.py:74-88`
- `verl/trainer/main.py:124-135`
- `verl/trainer/ray_trainer_dance.py:170-176`

---

## 3. 数据协议：Flux 与 Hunyuan 的第一处关键差异

DIscoRL 的 `LatentDataset` 对 Flux 会返回 4 元组：

```text
(prompt_embed, pooled_prompt_embeds, text_ids, caption)
```

对应到 trainer 中的变量是：

```text
(encoder_hidden_states, pooled_prompt_embeds, text_ids, caption)
```

因此 Flux Case4 在 `fit()` 中构造的 `DataProto` 包含：

- `encoder_hidden_states`
- `pooled_prompt_embeds`
- `text_ids`
- `meta_info["caption"]`

这和 Hunyuan Case4 的 3 元组不同：

- Hunyuan 是 `(encoder_hidden_states, encoder_attention_mask, caption)`
- Flux 不走 `encoder_attention_mask`
- Flux 需要额外的 `pooled_prompt_embeds` 与 `text_ids`

关键位置：

- `fastvideo/dataset/latent_rl_datasets.py:19-109`
- `verl/trainer/ray_trainer_dance.py:721-735`

---

## 4. 初始化阶段调用链

### 4.1 Driver 侧

Case4 下 driver 侧初始化为：

```text
RayPPOTrainerDance.init_workers()
  -> resource_pool_manager.create_resource_pool()
  -> create_colocated_worker_cls(...)
  -> spawn colocated actor_rollout_ref worker group
  -> self.actor_rollout_ref_wg.init_model()
```

因为是 colocated worker，采样与 actor 更新都落在同一组 `actor_rollout_ref` worker 上。

关键位置：

- `verl/trainer/ray_trainer_dance.py:521-579`

### 4.2 Worker 侧

`FSDPWorkerDance.init_model()` 会判断 `self.colocated`：

```text
if self.colocated:
    self._build_model_optimizer_dance()
else:
    self._build_model_optimizer_dance_dis()
```

Case4 是 colocated，因此走 `_build_model_optimizer_dance()`。

关键位置：

- `verl/workers/fsdp_workers_dance.py:1433-1440`

### 4.3 Flux 模型与奖励初始化

当 `trainer.model_name="flux"` 时，`_build_model_optimizer_dance()` 会走 Flux 分支：

1. 可选设置 `algorithm.seed`。
2. 如果角色是 `actor_rollout_ref` 或 `rollout_ref`，按奖励配置加载 HPSv2 或 PickScore。
3. 加载 `FluxTransformer2DModel.from_pretrained(..., subfolder="transformer")`。
4. 用 `get_dit_fsdp_kwargs(...)` 生成 FSDP 配置。
5. 按 `trainer.fsdp_strategy` 选择 FSDP 或 FSDP2。
6. 创建 `AdamW`。
7. 创建 lr scheduler。
8. 加载 `AutoencoderKL.from_pretrained(..., subfolder="vae")`。
9. 设置 `self.noise_scheduler = None`。

关键位置：

- `verl/workers/fsdp_workers_dance.py:961-1088`

### 4.4 HPSv2 / PickScore 奖励模型

Flux dance 侧不是 VideoAlign 的 `VQ/MQ` 双奖励，而是图像奖励：

- `worker.reward.use_hpsv2=true` 时加载 HPSv2 的 OpenCLIP 模型与 tokenizer。
- `trainer.use_pickscore=true` 时加载 PickScore processor/model。
- Case4 rollout decode 出图片后，用 caption 计算单个 `rewards` 张量。

DIscoRL 源代码里 HPSv2 路径是硬编码：

```text
/workspace/DanceGRPO/hps_ckpt/open_clip_pytorch_model.bin
/workspace/DanceGRPO/hps_ckpt/HPS_v2.1_compressed.pt
```

这正是迁移到 long-rl 时需要替换为环境变量或 `/home/qzy/models` 路径的点。

关键位置：

- `verl/workers/fsdp_workers_dance.py:974-1003`
- `verl/workers/fsdp_workers_dance.py:1004-1010`
- `verl/workers/fsdp_workers_dance.py:3134-3181`

---

## 5. 每个训练 step 的主调用链

Case4 下 `RayPPOTrainerDance.fit()` 每 step 的 Flux 主链是：

1. 从 dataloader 取一批：
   ```text
   encoder_hidden_states, pooled_prompt_embeds, text_ids, caption = next(iter(self.train_dataloader))
   ```
2. 构造：
   ```text
   DataProto.from_single_dict(
       {
           "encoder_hidden_states": encoder_hidden_states,
           "pooled_prompt_embeds": pooled_prompt_embeds,
           "text_ids": text_ids,
       },
       meta_info={"caption": caption},
   )
   ```
3. 调用：
   ```text
   self.actor_rollout_ref_wg.generate_sequences(new_batch)
   ```
4. 调用：
   ```text
   self.actor_rollout_ref_wg.update_actor(batch)
   ```

关键位置：

- `verl/trainer/ray_trainer_dance.py:681-753`

补充：DIscoRL 这个 `fit()` 里有写 `output_colocate1.txt` 后 `SystemExit(0)` 的计时性退出逻辑。本文按主训练语义分析，不把该测试出口作为算法流程的一部分。

关键位置：

- `verl/trainer/ray_trainer_dance.py:756-765`

---

## 6. 分发与回收机制

`generate_sequences` 与 `update_actor` 都是：

```text
@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
```

因此 driver 侧的一次：

```text
actor_rollout_ref_wg.generate_sequences(new_batch)
```

底层语义是：

1. 输入 `DataProto` 被按 worker group 的 world size 切分。
2. 每个 rank 执行 `FSDPWorkerDance.generate_sequences()`。
3. 各 rank 返回的 `DataProto` 在 driver 侧 concat。

`update_actor` 同理，只是执行的是每个 rank 的 `FSDPWorkerDance.update_actor()`。

关键位置：

- `verl/workers/fsdp_workers_dance.py:1677`
- `verl/workers/fsdp_workers_dance.py:2711`
- `verl/single_controller/base/decorator.py`
- `verl/single_controller/ray/base.py`

---

## 7. `generate_sequences()` 的 Flux rollout 细节

### 7.1 输入整理

Flux 分支从 `prompts.batch` 读取：

- `encoder_hidden_states`
- `pooled_prompt_embeds`
- `text_ids`

从 `prompts.meta_info` 读取：

- `caption`

如果 `worker.rollout.use_group=true`，按 `worker.rollout.num_generations` 扩展条件和 caption。

关键位置：

- `verl/workers/fsdp_workers_dance.py:2974-2998`

### 7.2 latent 形状与 image ids

Flux 分支内部定义了三个辅助函数：

- `prepare_latent_image_ids(...)`
- `pack_latents(...)`
- `unpack_latents(...)`

初始 latent 是图像 latent：

```text
(B, 16, latent_h, latent_w)
```

进入 Flux transformer 前会被 `pack_latents(...)` 转成 patch 序列形态：

```text
(B, num_patches, channels * 4)
```

同时生成 `image_ids`，作为 Flux transformer 的 `img_ids` 输入。

关键位置：

- `verl/workers/fsdp_workers_dance.py:2952-2972`
- `verl/workers/fsdp_workers_dance.py:2999-3060`

### 7.3 sigma schedule 与采样循环

Flux rollout 会：

1. 构造 `sigma_schedule = linspace(1, 0, sampling_steps + 1)`。
2. 经过 `sd3_time_shift(shift, sigma_schedule)`。
3. 每个采样 step 调 Flux transformer：
   - `hidden_states=z`
   - `encoder_hidden_states=batch_encoder_hidden_states`
   - `timestep=timesteps / 1000`
   - `guidance=[rollout.guidance_scale]`
   - `txt_ids=batch_text_ids.repeat(...)`
   - `pooled_projections=batch_pooled_prompt_embeds`
   - `img_ids=image_ids`
4. 调 `flux_step(..., grpo=True, sde_solver=True)` 得到：
   - 下一步 latent
   - `pred_original`
   - 当前 step 的 `log_prob`
5. 保存整条 latent 轨迹和 `log_probs`。

这里保存的 `log_probs` 就是后续 PPO ratio 的 old log-prob。

关键位置：

- `verl/workers/fsdp_workers_dance.py:3000-3024`
- `verl/workers/fsdp_workers_dance.py:3061-3128`

### 7.4 解码与奖励

采样结束后，Flux 分支会：

1. `unpack_latents(pred_original, h, w, 8)`。
2. 做 Flux VAE 的反标定：
   ```text
   latents_img = (latents_img / 0.3611) + 0.1159
   ```
3. 用 VAE decode 成 image。
4. 用 `VaeImageProcessor(16)` 后处理。
5. 如果启用 HPSv2：
   - `preprocess_val(decoded[0])`
   - `reward_processor([caption])`
   - `reward_model(img, txt)`
   - `image_features @ text_features.T`
   - diagonal 作为 score
6. 如果启用 PickScore，则走 PickScore 的 image/text feature 相似度。

返回奖励字段是单个：

```text
rewards
```

不是 Hunyuan/VideoAlign 的：

```text
vq_rewards / mq_rewards
```

关键位置：

- `verl/workers/fsdp_workers_dance.py:3134-3181`

### 7.5 返回结构

Flux `generate_sequences()` 返回的 `DataProto` 包含：

- `timesteps`
- `latents`
- `next_latents`
- `log_probs`
- `rewards`
- `image_ids`
- `text_ids`
- `encoder_hidden_states`
- `pooled_prompt_embeds`

`meta_info` 包含：

- `sigma_schedule`
- `prompt_caption`
- compare/debug 相关 meta

关键位置：

- `verl/workers/fsdp_workers_dance.py:3183-3229`

---

## 8. `update_actor()` 的 Flux GRPO/PPO 更新细节

### 8.1 输入整理

Flux `update_actor()` 会：

1. `self.optimizer.zero_grad()`。
2. 把 `data.batch` 搬到当前 device。
3. 从 `data.meta_info["sigma_schedule"]` 恢复 tensor。
4. 读取：
   - `latents`
   - `next_latents`
   - `timesteps`
   - `log_probs`
   - `rewards`
   - `image_ids`
   - `text_ids`
   - `encoder_hidden_states`
   - `pooled_prompt_embeds`

关键位置：

- `verl/workers/fsdp_workers_dance.py:1977-1994`

### 8.2 组内 advantage

Flux Case4 用单个 `rewards` 做 GRPO 标准化：

```text
advantages = (rewards - group_mean) / (group_std + 1e-8)
```

分组大小来自：

```text
worker.rollout.num_generations
```

关键位置：

- `verl/workers/fsdp_workers_dance.py:1996-2007`

### 8.3 best-of-n 筛选

Flux 的 `total_scores` 直接使用：

```text
samples["advantages"]
```

随后：

1. `torch.argsort(total_scores)`
2. 取 top `bestofn/2`
3. 取 bottom `bestofn/2`
4. concat 后打乱
5. 如果 `num_generations != bestofn`，按 selected indices 筛选所有 sample 字段

关键位置：

- `verl/workers/fsdp_workers_dance.py:2009-2021`

### 8.4 新策略 log-prob 与 PPO clipped loss

更新时会先随机打乱 timestep：

```text
perms = torch.stack([torch.randperm(len(samples["timesteps"][0])) for _ in range(batch_size)])
```

然后只训练：

```text
train_timesteps = int(T * worker.actor.timestep_fraction)
```

每个 timestep 调 `grpo_one_step(...)`：

1. 用当前 `self.transformer` 重新 forward。
2. 重新计算 `new_log_probs`。
3. `ratio = exp(new_log_probs - old_log_probs)`。
4. 对 clipped PPO loss 取 max。
5. loss 除以 `gradient_accumulation_steps * train_timesteps`。
6. `loss.backward()`。

关键位置：

- `verl/workers/fsdp_workers_dance.py:1936-1975`
- `verl/workers/fsdp_workers_dance.py:2035-2085`

### 8.5 梯度累计与优化

每当满足：

```text
(i + 1) % worker.actor.gradient_accumulation_steps == 0
```

执行：

- `self.transformer.clip_grad_norm_(worker.actor.max_grad_norm)`
- `self.optimizer.step()`
- `self.lr_scheduler.step()`
- `self.optimizer.zero_grad()`

最后返回：

```text
DataProto(non_tensor_batch={"actor_loss": loss_np})
```

关键位置：

- `verl/workers/fsdp_workers_dance.py:2086-2116`

---

## 9. Case4 Flux 与 Case4 Hunyuan 的核心差异

1. 输入协议不同：
   - Hunyuan: `encoder_hidden_states + encoder_attention_mask`
   - Flux: `encoder_hidden_states + pooled_prompt_embeds + text_ids`
2. 模型不同：
   - Hunyuan: `HunyuanVideoTransformer3DModel`
   - Flux: `FluxTransformer2DModel`
3. latent 形态不同：
   - Hunyuan 是 video latent，带 temporal 维度
   - Flux 是 image latent，进 transformer 前需要 pack 成 patch 序列
4. 奖励不同：
   - Hunyuan 当前主链是 VideoAlign `VQ/MQ`
   - Flux 当前主链是 HPSv2/PickScore 风格单 `rewards`
5. update objective 不同：
   - Hunyuan 分别算 `vq_advantages/mq_advantages` 再加权
   - Flux 只基于 `advantages` 算单路 PPO clipped loss

---

## 10. 总体调用图（DIscoRL Flux Case4）

```text
python -m verl.trainer.main
  -> Runner.run
  -> RayPPOTrainerDance(...)
  -> init_workers
  -> actor_rollout_ref_wg.init_model
  -> FSDPWorkerDance.init_model
  -> _build_model_optimizer_dance(model_name="flux")
  -> RayPPOTrainerDance.fit
  -> dataloader returns (encoder_hidden_states, pooled_prompt_embeds, text_ids, caption)
  -> DataProto.from_single_dict(...)
  -> actor_rollout_ref_wg.generate_sequences
  -> FSDPWorkerDance.generate_sequences(model_name="flux")
  -> Flux rollout + VAE decode + HPSv2/PickScore reward
  -> actor_rollout_ref_wg.update_actor
  -> FSDPWorkerDance.update_actor(model_name="flux")
  -> GRPO advantage + best-of-n + PPO clipped update
```

---

## 11. 对迁移的直接启示

1. Flux Case4 迁移不能照搬 Hunyuan Case4 的 3 元组 dataloader 协议，必须支持 `pooled_prompt_embeds/text_ids`。
2. Flux Case4 迁移不能复用 VideoAlign `VQ/MQ` 更新逻辑，必须保留单 `rewards -> advantages -> PPO loss` 路径。
3. DIscoRL 入口里的 `trainer.grpo_variant`、`worker.actor.disco`、flow 分支都不应搬进 long-rl。
4. HPSv2/OpenCLIP 的硬编码路径必须迁移成 `/home/qzy/models` 或环境变量。
5. long-rl 目标实现应保持 Case4 的 colocated 语义：同一组 `actor_rollout_ref` worker 同时负责 Flux rollout 和 actor update。
