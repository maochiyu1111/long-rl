# Case4（long-rl Flux，目标态预演）：迁移成功后的实际执行流程

本文不是描述当前仓库已经实现的真实行为，而是按“Flux Case4 已按 `flux_migration_guide.md` 迁移完成”的目标态来写，用来回答：如果 long-rl 把 DIscoRL 的 Flux + DanceGRPO Case4 迁移干净，执行流程应该长什么样。

目标主链路可以先浓缩成一句话：

```text
case4_flux.sh
  -> python -m verl.trainer.main_flux
  -> TaskRunner.run()
  -> RayPPOTrainerDance._create_flux_dance_latent_dataloader("dance_case4_mode")
  -> RayPPOTrainerDance.init_workers()
  -> actor_rollout_wg.init_model()
  -> FSDPWorkerDance._build_model_optimizer_dance(model_name="flux")
  -> RayPPOTrainerDance.fit()
  -> RayPPOTrainerDance.fit_dance_case4()
  -> actor_rollout_wg.generate_sequences(new_batch)
  -> FSDPWorkerDance._generate_sequences_dance(model_name="flux")
  -> actor_rollout_wg.update_actor(rollout_batch)
  -> FSDPWorkerDance._update_actor_dance(model_name="flux")
```

---

## 1. 命中的是哪条目标分支

Flux Case4 的成功迁移不应复活 DIscoRL 的旧字段：

- 不使用 `trainer.grpo_variant`
- 不使用 `worker.actor.disco`
- 不使用 `actor_rollout_ref.actor.disco`
- 不迁移 flow 分支

固定命中条件应为：

```text
trainer.diffusion=true
trainer.disaggregate=false
trainer.pipelined_micro_batch=false
algorithm.adv_estimator=grpo
trainer.model_name=flux
actor_rollout_ref.actor.dance_case4_mode=true
```

这意味着：

- Flux Case4 是 diffusion 训练。
- Flux Case4 是 colocated 单 WorkerGroup 拓扑。
- Flux Case4 的 rollout 和 actor update 都在 long-rl 的 colocated `actor_rollout_wg` 内完成，worker role 是 `actor_rollout`。
- `actor_rollout_ref.*` 只表示 long-rl 配置 namespace，不是运行时 WorkerGroup 名。
- Flux Case4 不走 disaggregate。
- Flux Case4 不走 async dual-rollout。
- Flux Case4 不走通用 diffusion PPO 主链。

和 Hunyuan Case4 一样，trainer 的 `fit()` 应在入口处识别 `dance_case4_mode=true` 并直接转入：

```text
fit()
  -> fit_dance_case4()
```

---

## 2. 入口：`case4_flux.sh`

目标态下应新增独立入口：

```bash
bash case4_flux.sh
```

脚本职责应与已有 `case4_dance.sh` 一致，但换成 Flux 专用路径与入口：

1. 设置基本环境变量：
   - `PYTHONUNBUFFERED=1`
   - 离线环境变量按测试机需要设置
   - `OPEN_CLIP_CKPT_PATH`
   - `HPSV2_CKPT_PATH`
2. 指向 Flux 配置文件：
   - `CONFIG_PATH=${ROOT_DIR}/examples/diffusion`
   - `CONFIG_NAME=config_video_diffusion_case4_flux`
3. 使用 long-rl Hydra CLI 风格：
   ```bash
   python3 -m verl.trainer.main_flux \
     --config-path="${CONFIG_PATH}" \
     --config-name="${CONFIG_NAME}" \
     ...
   ```
4. 默认路径应落到 `/home/qzy/models`：
   - `FLUX_MODEL_PATH=/home/qzy/models/flux`
   - `DATA_JSON_PATH=/home/qzy/models/flux/rl_embeddings/videos2caption.json`
   - `OPEN_CLIP_CKPT_PATH=/home/qzy/models/open_clip_pytorch_model.bin`
   - `HPSV2_CKPT_PATH=/home/qzy/models/HPS_v2.1_compressed.pt`
5. 只使用 long-rl namespace 覆盖参数：
   - `actor_rollout_ref.actor.extra.dance.*`
   - `actor_rollout_ref.rollout.*`
   - `data.*`
   - `trainer.*`

不应出现：

- `python3 -m verl.trainer.main`
- `config=examples/config_flux.yaml`
- `worker.actor.*`
- `worker.rollout.*`
- `worker.actor.disco`
- `trainer.grpo_variant`

---

## 3. 配置：`config_video_diffusion_case4_flux.yaml`

Flux Case4 YAML 应以同号 Hunyuan 文件为骨架：

```text
examples/diffusion/config_video_diffusion_case4_dance.yaml
```

必须严格对齐的 Case4 开关是：

```yaml
trainer:
  diffusion: true
  disaggregate: false
  pipelined_micro_batch: false

algorithm:
  adv_estimator: grpo

actor_rollout_ref:
  actor:
    dance_case4_mode: true
```

Flux 专用字段则来自 DIscoRL Flux/Dance 配置，但要落到 long-rl namespace：

```yaml
trainer:
  model_name: flux

data:
  data_json_path: /home/qzy/models/flux/rl_embeddings/videos2caption.json
  cfg: 0.0

actor_rollout_ref:
  model:
    path: /home/qzy/models/flux
  actor:
    gradient_accumulation_steps: 8
    grad_clip: 0.01
    extra:
      diffusion: true
      dance:
        pretrained_model_name_or_path: /home/qzy/models/flux
        vae_model_path: /home/qzy/models/flux
        model_type: flux
        master_weight_type: fp32
        timestep_fraction: 1.0
        guidance_scale: 3.5
        clip_range: 1.0e-4
        adv_clip_max: 5.0
        timestep_micro_batch: 1
        use_hpsv2: true
        use_pickscore: false
  rollout:
    guidance_scale: 3.5
    height: 720
    width: 720
    num_frames: 1
    shift: 3
    eta: 0.3
    fps: 8
    num_generations: 2
    bestofn: 2
    sampling_steps: 16
    use_group: true
    use_same_noise: true
```

注意：`model_type: flux` 只能作为 Flux 专用 worker 的分支标记使用。迁移后的 `fsdp_workers_flux_dance.py` 必须在 `trainer.model_name=flux` 或 `extra.dance.model_type=flux` 时直接调用 `FluxTransformer2DModel.from_pretrained(...)` 与 `AutoencoderKL.from_pretrained(...)`，不能继续走当前 Hunyuan Case4 使用的 `fastvideo.utils.load.load_transformer()` / `load_vae()`，否则会因 `model_type=flux` 不受支持而失败。

这些 DIscoRL Flux actor/update 字段也必须落到 long-rl schema 或由 `extra.dance` 读取：

- `worker.actor.gradient_accumulation_steps` -> `actor_rollout_ref.actor.gradient_accumulation_steps`
- `worker.actor.max_grad_norm` -> `actor_rollout_ref.actor.grad_clip` 或兼容读取 `max_grad_norm`
- `worker.actor.timestep_fraction` -> `actor_rollout_ref.actor.extra.dance.timestep_fraction`
- `worker.actor.guidance_scale` -> `actor_rollout_ref.actor.extra.dance.guidance_scale`
- `worker.actor.clip_range` -> `actor_rollout_ref.actor.extra.dance.clip_range`
- `worker.actor.adv_clip_max` -> `actor_rollout_ref.actor.extra.dance.adv_clip_max`
- `worker.actor.timestep_micro_batch` -> `actor_rollout_ref.actor.extra.dance.timestep_micro_batch`

奖励配置应体现 Flux 的单 reward 语义：

- HPSv2: 使用 `HPSV2_CKPT_PATH` 与 `OPEN_CLIP_CKPT_PATH`
- PickScore: 只有确认要迁移时才开启
- 不使用 VideoAlign `VQ/MQ` 作为 Flux 主链奖励
- 推荐把 `use_hpsv2` / `use_pickscore` 放在 `actor_rollout_ref.actor.extra.dance.*` 或明确的 Flux reward namespace 中，并让 worker 只从 long-rl namespace 读取，不再读取 DIscoRL 的 `worker.reward.*`。

硬性禁止：

- 不写 `disco`
- 不写 `grpo_variant`
- 不写 flow 相关字段
- 不照抄 DIscoRL 的 `worker.actor.*` 顶层结构

---

## 4. `main_flux.py` 入口层目标行为

`main_flux.py` 应沿用 long-rl `main_ppo.py` 的框架风格：

- Hydra 装饰器
- `TaskRunner` Ray remote
- `--config-path / --config-name`
- `get_ppo_ray_runtime_env` 风格的 runtime env
- long-rl `ResourcePoolManager` / `RayWorkerGroup`

但它只支持 Flux + Dance：

```text
main_flux.py
  -> RayPPOTrainerDance
  -> FSDPWorkerDance
```

不应 import：

- `RayPPOTrainerFlow`
- `FSDPWorkerFlow`
- `main_flow`

当 `dance_case4_mode=true` 时，`TaskRunner.run()` 应像 Hunyuan Case4 一样：

1. `_build_tokenizer_and_processor(config)` 返回 `(None, None)`。
2. 不加载通用 reward manager：
   - `reward_fn = None`
   - `val_reward_fn = None`
3. 不创建通用 RLHF dataset：
   - `train_dataset = None`
   - `val_dataset = None`
   - `collate_fn = None`
   - `train_sampler = None`
4. 创建 `RayPPOTrainerDance(...)`。
5. 因 `trainer.disaggregate=false`，执行：
   ```text
   trainer.init_workers()
   trainer.fit()
   ```

---

## 5. Dataloader 协议必须升级为 Flux 4 元组

这是 Flux Case4 相对当前 Hunyuan Case4 最容易踩坑的地方。

Hunyuan Case4 的 dataloader 返回：

```text
(encoder_hidden_states, encoder_attention_mask, caption)
```

Flux Case4 的目标 dataloader 必须返回：

```text
(encoder_hidden_states, pooled_prompt_embeds, text_ids, caption)
```

因此不能直接复用当前只返回 3 元组的 Hunyuan latent dataloader 逻辑，否则 `fit_dance_case4()` 会在输入协议上直接错位。

目标态可以采用两种实现之一：

1. 新增 Flux 专用 dataloader helper：
   ```text
   _create_flux_dance_latent_dataloader("dance_case4_mode")
   ```
2. 扩展现有 `_create_dance_latent_dataloader(...)`，让它按 `trainer.model_name=flux` 选择 Flux 数据集协议。

这是必改点，不是单纯 YAML 配置问题。当前 long-rl 的 `_create_dance_latent_dataloader(...)` 固定导入：

```text
fastvideo.dataset.latent_rl_datasets.LatentDataset
```

该数据集当前只返回 Hunyuan 3 元组。Flux 4 元组逻辑已经存在于：

```text
fastvideo.dataset.latent_flux_rl_datasets.LatentDataset
```

因此迁移时必须显式切到 Flux 数据集，或把 Flux 4 元组逻辑合并进 dance dataloader；否则 `config_video_diffusion_case4_flux.yaml` 即使配置了 Flux 路径，也仍然会按 Hunyuan 3 元组协议取数。

无论用哪种方式，目标输出都必须保证：

- `encoder_hidden_states` 对应 Flux prompt embeddings
- `pooled_prompt_embeds` 对应 Flux pooled prompt embeddings
- `text_ids` 对应 Flux text ids
- `caption` 进入 `DataProto.meta_info`

---

## 6. `fit_dance_case4()` 的目标行为

Flux 迁移成功后，Case4 主循环仍应复用 Hunyuan Case4 的同步骨架：

```text
data_iterator = iter(self.train_dataloader)
while self.global_steps < max_train_steps:
    batch = next(data_iterator)
    new_batch = DataProto.from_single_dict(...)
    rollout_batch = self.actor_rollout_wg.generate_sequences(new_batch)
    actor_output = self.actor_rollout_wg.update_actor(rollout_batch)
```

但 batch 解包要按 `trainer.model_name` 分流：

```text
if trainer.model_name in {"hunyuan", "hunyuan_hf"}:
    encoder_hidden_states, encoder_attention_mask, caption = batch
    batch_dict = {
        "encoder_hidden_states": encoder_hidden_states,
        "encoder_attention_mask": encoder_attention_mask,
    }

elif trainer.model_name == "flux":
    encoder_hidden_states, pooled_prompt_embeds, text_ids, caption = batch
    batch_dict = {
        "encoder_hidden_states": encoder_hidden_states,
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "text_ids": text_ids,
    }
```

随后统一：

```text
DataProto.from_single_dict(batch_dict, meta_info={"caption": caption})
```

这是必改点。当前 long-rl 的 `fit_dance_case4()` 会先校验 `len(batch) == 3`，并固定组装 `encoder_attention_mask`；迁移 Flux Case4 时必须把这段改成按 `trainer.model_name` 分流，否则 Flux dataloader 返回 4 元组会在 trainer 入口直接失败。

这里应保持 Case4 的关键语义：

- 一次 dataloader batch 对应一次同步 rollout。
- rollout 返回的 batch 直接送 actor update。
- 不在 trainer 侧计算 reward。
- 不在 trainer 侧计算 `compute_advantage_diffusion(...)`。
- 不在 trainer 侧做异步 `ray.wait`。
- 不做 actor/rollout_ref 双 WorkerGroup 调度。

---

## 7. Worker 初始化：Flux colocated actor_rollout

目标态新增的 `verl/workers/fsdp_workers_flux_dance.py` 中，类名可以保持 DIscoRL 原名：

```text
FSDPWorkerDance
```

Case4 下 worker 初始化链路是：

```text
actor_rollout_wg.init_model()
  -> role="actor_rollout"
FSDPWorkerDance.init_model()
  -> colocated=true
  -> _build_model_optimizer_dance()
  -> model_name="flux"
```

Flux 分支应完成：

1. 读取 long-rl namespace：
   - `actor_rollout_ref.actor.extra.dance.pretrained_model_name_or_path`
   - `actor_rollout_ref.actor.extra.dance.vae_model_path`
   - `actor_rollout_ref.actor.extra.dance.master_weight_type`
   - `actor_rollout_ref.actor.extra.dance.timestep_fraction`
   - `actor_rollout_ref.actor.extra.dance.use_hpsv2`
   - `actor_rollout_ref.actor.extra.dance.use_pickscore`
   - `actor_rollout_ref.rollout.*`
2. 加载 `FluxTransformer2DModel.from_pretrained(..., subfolder="transformer")`。
3. 用 fastvideo FSDP 工具包装 transformer。
4. 创建 optimizer 与 lr scheduler。
5. 加载 `AutoencoderKL.from_pretrained(..., subfolder="vae")`。
6. 如果启用 HPSv2，加载 OpenCLIP 与 HPSv2 checkpoint。
7. 如果启用 PickScore，加载 PickScore processor/model。

路径必须改成：

- `OPEN_CLIP_CKPT_PATH` 或 `/home/qzy/models/open_clip_pytorch_model.bin`
- `HPSV2_CKPT_PATH` 或 `/home/qzy/models/HPS_v2.1_compressed.pt`
- `FLUX_MODEL_PATH` 或 `/home/qzy/models/flux`

---

## 8. `_generate_sequences_dance()` 的 Flux 目标流程

Flux 目标分支从 `DataProto` 读取：

- `encoder_hidden_states`
- `pooled_prompt_embeds`
- `text_ids`
- `meta_info["caption"]`

如果 `rollout.use_group=true`，按 `rollout.num_generations` 扩展：

- `encoder_hidden_states`
- `pooled_prompt_embeds`
- `text_ids`
- `caption`

随后执行 Flux rollout：

1. 构造 `sigma_schedule = linspace(1, 0, sampling_steps + 1)`。
2. 用 `sd3_time_shift(shift, sigma_schedule)` 调整时间步。
3. 创建初始 image latent：
   ```text
   (B, 16, height/8, width/8)
   ```
4. `pack_latents(...)` 转成 Flux patch 序列。
5. `prepare_latent_image_ids(...)` 生成 `image_ids`。
6. 每个 diffusion step 调 `self.transformer(...)`：
   - `hidden_states=z`
   - `encoder_hidden_states=...`
   - `timestep=timesteps / 1000`
   - `guidance=[rollout.guidance_scale]`
   - `txt_ids=text_ids.repeat(...)`
   - `pooled_projections=pooled_prompt_embeds`
   - `img_ids=image_ids`
7. 调 `flux_step(..., grpo=True, sde_solver=True)` 得到下一步 latent、`pred_original`、`log_prob`。
8. 保存 `latents`、`next_latents`、`log_probs`。
9. `unpack_latents(...)` 后用 VAE decode。
10. 用 HPSv2/PickScore 对 decode image + caption 打分。

返回 `DataProto` 的 tensor 字段应包含：

- `timesteps`
- `latents`
- `next_latents`
- `log_probs`
- `rewards`
- `image_ids`
- `text_ids`
- `encoder_hidden_states`
- `pooled_prompt_embeds`

`meta_info` 至少包含：

- `sigma_schedule`
- `prompt_caption`

注意：Flux Case4 不应返回 `vq_rewards/mq_rewards` 作为主链字段。

---

## 9. `_update_actor_dance()` 的 Flux 目标流程

Flux update 目标逻辑与 DIscoRL 的 Flux Case4 对齐，但配置读取要改成 long-rl namespace。

输入字段：

- `timesteps`
- `latents`
- `next_latents`
- `log_probs`
- `rewards`
- `image_ids`
- `text_ids`
- `encoder_hidden_states`
- `pooled_prompt_embeds`
- `meta_info["sigma_schedule"]`

### 9.1 advantage

按 `rollout.num_generations` 分组：

```text
advantages = (rewards - group_mean) / (group_std + 1e-8)
```

不要调用：

- `compute_advantage_diffusion(...)`
- Hunyuan 的 `vq_advantages/mq_advantages` 分支

### 9.2 best-of-n

使用单路 reward advantage：

```text
total_scores = advantages
```

然后：

1. 排序。
2. 取 top `bestofn/2`。
3. 取 bottom `bestofn/2`。
4. concat 后 shuffle。
5. 如果 `num_generations != bestofn`，筛选 batch。

### 9.3 PPO clipped loss

每个样本、每个训练 timestep：

1. 用当前 actor transformer 重新算 `new_log_probs`。
2. `ratio = exp(new_log_probs - old_log_probs)`。
3. `advantages` 按 `actor_rollout_ref.actor.extra.dance.adv_clip_max` 裁剪。
4. 计算：
   ```text
   unclipped_loss = -advantages * ratio
   clipped_loss = -advantages * clamp(ratio, 1 - clip_range, 1 + clip_range)
   loss = max(unclipped_loss, clipped_loss)
   ```
5. 按 `gradient_accumulation_steps * train_timesteps` 归一。
6. `loss.backward()`。

这里的 `clip_range`、`adv_clip_max`、`timestep_micro_batch` 不应继续硬编码或读取 DIscoRL `worker.actor.*`，而应从 long-rl namespace 读取，例如：

- `actor_rollout_ref.actor.extra.dance.clip_range`
- `actor_rollout_ref.actor.extra.dance.adv_clip_max`
- `actor_rollout_ref.actor.extra.dance.timestep_micro_batch`

### 9.4 optimizer step

Case4 是同步 colocated 更新，目标行为应是：

- 按 `actor_rollout_ref.actor.gradient_accumulation_steps` 累积。
- 使用 `actor_rollout_ref.actor.grad_clip` 或兼容读取 `actor_rollout_ref.actor.max_grad_norm` 做 clip。
- `optimizer.step()`。
- `lr_scheduler.step()`。
- `optimizer.zero_grad()`。

返回 metrics 应保持 long-rl 风格：

```text
DataProto(meta_info={"metrics": {"actor/loss": ...}})
```

如果为了兼容 DIscoRL 源实现临时返回 `non_tensor_batch["actor_loss"]`，也应在 trainer 侧统一转换为 long-rl metrics，避免 Case4 报告逻辑拿不到 actor loss。

---

## 10. 当前 long-rl Hunyuan Case4 不能直接复用的点

现有 Hunyuan Case4 已经提供了很好的骨架，但 Flux 不能只改模型路径。

必须显式处理这些差异：

1. Dataloader 形态：
   - 当前 Hunyuan 是 3 元组
   - Flux 目标是 4 元组
2. Prompt 条件字段：
   - 当前 Hunyuan 使用 `encoder_attention_mask`
   - Flux 使用 `pooled_prompt_embeds/text_ids`
3. Transformer forward 参数：
   - Hunyuan 使用 video transformer 参数协议
   - Flux 使用 `txt_ids/pooled_projections/img_ids`
4. latent 结构：
   - Hunyuan 是 video latent
   - Flux 是 image latent patch sequence，需要 pack/unpack
5. 奖励字段：
   - Hunyuan 是 `vq_rewards/mq_rewards`
   - Flux 是单 `rewards`
6. PPO loss：
   - Hunyuan 是 `vq_coef/mq_coef` 双路 loss
   - Flux 是单路 reward advantage loss

---

## 11. Flux Case4 明确不走的路径

迁移成功后的 Flux Case4 不应走：

- `fit_dis()`
- `fit_dance_case3_dis()`
- `fit_dance_dual_rollout_dis_async(...)`
- `generate_sequences_dance_async(...)`
- `update_actor_dance_async(...)`
- `load_reward_manager()`
- `BatchRewardManager`
- `compute_reward(...)`
- `compute_advantage_diffusion(...)`
- `verl/workers/rollout/diffusion_rollout.py`
- `DataParallelPPOActor.update_policy_diffusion()`
- flow trainer / flow worker
- DIscoRL 的 `worker.actor.disco` 分支

这条链路应该是“Flux colocated DanceGRPO”：

```text
worker 内 rollout/reward
  -> worker 内 GRPO/PPO update
```

而不是通用 diffusion PPO，也不是 disaggregate async。

---

## 12. 推荐实现拆分

为了不影响现有 Hunyuan dance 栈，Flux Case4 推荐按 `flux_migration_guide.md` 的独立文件策略实现：

1. 新增 `verl/trainer/main_flux.py`。
2. 新增 `verl/trainer/ray_trainer_flux_dance.py`。
3. 新增 `verl/workers/fsdp_workers_flux_dance.py`。
4. 新增 `examples/diffusion/config_video_diffusion_case4_flux.yaml`。
5. 新增 `case4_flux.sh`。

Flux 文件内部可以复用 Hunyuan Case4 的函数命名风格，但不要修改：

- `case4_dance.sh`
- `config_video_diffusion_case4_dance.yaml`
- `verl/trainer/main_ppo.py`
- `verl/workers/fsdp_workers.py`

除非某个 schema 字段确实需要追加到共享 config 中。

---

## 13. 验收口径

Flux Case4 迁移完成后，至少应满足：

1. `case4_flux.sh` 使用 `python3 -m verl.trainer.main_flux`。
2. `config_video_diffusion_case4_flux.yaml` 中：
   - `dance_case4_mode=true`
   - `trainer.disaggregate=false`
   - `trainer.pipelined_micro_batch=false`
   - `trainer.model_name=flux`
3. 新增 Flux 文件中没有 `disco` 字段。
4. 新增 Flux 文件中没有 `grpo_variant` 字段。
5. 新增 Flux 文件中没有 flow import。
6. dataloader 能返回 `(encoder_hidden_states, pooled_prompt_embeds, text_ids, caption)`。
7. `fit_dance_case4()` 能按 Flux 4 元组组装 `DataProto`。
8. `generate_sequences()` 返回 `rewards`，不是 `vq_rewards/mq_rewards`。
9. `update_actor()` 基于 `rewards` 计算 `advantages`。
10. HPSv2/OpenCLIP 路径来自环境变量或 `/home/qzy/models`。

---

## 14. 一句话总结

Flux Case4 迁移成功后的 long-rl 目标流程应该是：

```text
case4_flux.sh
  -> main_flux
  -> Flux/Dance 专用 trainer + worker
  -> colocated actor_rollout worker
  -> Flux 4 元组 latent dataloader
  -> fit_dance_case4()
  -> Flux rollout + VAE decode + HPSv2/PickScore reward
  -> single-reward GRPO/PPO actor update
```

也就是说，Flux Case4 的迁移目标不是“把 Hunyuan Case4 的模型路径换成 Flux”，而是“在 long-rl 的 Case4 colocated 骨架下，完整承接 DIscoRL Flux 的输入协议、Flux transformer forward、image reward、单 reward PPO update 语义”。
