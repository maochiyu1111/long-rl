# Case 3: `disaggregate=true`、`pipelined_micro_batch=false`、`disco=false` 时 DanceGRPO 执行链路

## 1. 结论与范围

本文只分析你关心的主训练链路（忽略 `fit_dis()` 里的测试性提前退出逻辑），即：

`next(iter(train_dataloader)) -> DataProto.from_single_dict() -> [设计意图下先完成 actor -> rollout 权重对齐] -> rollout_ref_wg.generate_sequences(new_batch) -> FSDPWorker.generate_sequences() -> actor_wg.update_actor(batch) -> FSDPWorker.update_actor()`

在当前仓库中，这组参数对应的是 **disaggregate 双 WorkerGroup** 路径：

- 不走 `fit()`
- 不走 `fit_disco_pipelined()`
- 走 `init_workers_dis() + fit_dis()`

并且其设计语义是：

- `rollout_ref_wg` 负责 rollout / reward 一侧的采样与轨迹构造
- `actor_wg` 负责 actor 一侧的 GRPO/PPO 更新
- 两组 worker 之间通过 process group 与权重同步接口协作

关键分支位置：

- `verl/trainer/main.py:39-57`
- `verl/trainer/main.py:108-113`

---

## 2. 入口与分支决策

### 2.1 入口调用

1. `main()` 组装配置并启动 Ray
2. `Runner.run(config)` 创建 `RayPPOTrainer` 并进入训练

关键位置：

- `verl/trainer/main.py:121-151`
- `verl/trainer/main.py:34-116`

### 2.2 参数如何决定执行路径

当 `trainer.disaggregate=true` 时，`Runner.run()` 走 disaggregate 分支：

- 角色映射为 `Role.RolloutRef` + `Role.Actor` + `Role.Critic`
- 之后实际调用 `trainer.init_workers_dis()`
- 由于 `worker.actor.disco=false` 且 `trainer.pipelined_micro_batch=false`，进入 `trainer.fit_dis()`

关键位置：

- `verl/trainer/main.py:39-57`
- `verl/trainer/main.py:108-113`

注意：虽然 `role_worker_mapping` 里仍然声明了 `Role.Critic`，但 `adv_estimator=grpo` 时：

- `self.use_critic=False`
- 这条主链里不会进入 critic 训练分支

关键位置：

- `verl/trainer/ray_trainer.py:221-224`

---

## 3. 初始化阶段调用链

### 3.1 Driver 侧：创建两组 WorkerGroup

1. `RayPPOTrainer.init_workers_dis()`
2. 创建 `pool_fast` / `pool_slow` 对应资源池
3. 为 `Role.RolloutRef` 构造 `rollout_ref_cls`
4. 为 `Role.Actor` 构造 `actor_cls`
5. 对每个资源池调用 `create_colocated_worker_cls(...)`
6. `spawn()` 后拿到：
   - `self.rollout_ref_wg`
   - `self.actor_wg`

关键位置：

- `verl/trainer/ray_trainer.py:334-367`

### 3.2 `actor_setup_dist` / `rollout_ref_setup_dist` 是怎么来的

这里代码里看起来像“存在两个独立函数”：

- `w.actor_setup_dist.remote(...)`
- `w.rollout_ref_setup_dist.remote(...)`

但它们并不是 `FSDPWorker` 显式定义的两个方法，而是 `create_colocated_worker_cls(...)` 生成的前缀代理：

1. `create_colocated_worker_cls(...)` 会把被 `@register` 修饰的方法绑定成 `<prefix>_<method_name>` 形式  
   例如 `actor_setup_dist`、`rollout_ref_setup_dist`
2. `spawn(prefix_set=...)` 再把这些前缀方法重绑定到对应子 `WorkerGroup`
3. 在 `init_workers_dis()` 里，trainer 直接遍历底层 Ray actor handle，所以调用的是带前缀的 `.remote(...)`

关键位置：

- `verl/single_controller/ray/base.py:329-355`
- `verl/single_controller/ray/base.py:417-445`
- `verl/single_controller/ray/base.py:456-497`

### 3.3 `setup_dist()`：建立 actor / rollout_ref 两组通信域

`init_workers_dis()` 会分别给 actor 组和 rollout 组下发 rank 信息，最终都落到 `FSDPWorker.setup_dist()`：

1. 设置 `MASTER_ADDR` / `MASTER_PORT` / `RANK` / `WORLD_SIZE` / `LOCAL_RANK`
2. `dist.init_process_group(backend="nccl", init_method="env://")`
3. 创建：
   - `self.actor_pg`
   - `self.rollout_ref_pg`
   - `self.actor_rollout_pg`
4. 根据当前角色设置本组 ranks：
   - actor 侧用 `actor_group_ranks`
   - rollout_ref 侧用 `rollout_ref_group_ranks`

其中 `actor_rollout_pg` 的设计语义是：

- 让 actor 侧 rank 0 与 rollout/ref 组建立跨组广播通道
- 为后续 actor -> rollout 权重同步提供基础

关键位置：

- `verl/trainer/ray_trainer.py:369-418`
- `verl/workers/fsdp_workers.py:241-279`

### 3.4 Worker 侧：`init_model()` 进入 disaggregate 构建逻辑

在 Case 3 中：

- `role="actor"` 时 `self.colocated=False`
- `role="rollout_ref"` 时 `self.colocated=False`

因此两组 worker 的 `init_model()` 都不会走 colocated 的 `_build_model_optimizer_dance()`，而是统一进入：

- `_build_model_optimizer_dance_dis()`

关键位置：

- `verl/workers/fsdp_workers.py:156-160`
- `verl/workers/fsdp_workers.py:1317-1401`

### 3.5 两组 worker 的职责拆分

#### A. `rollout_ref_wg` 初始化链

设计语义下的 rollout 侧初始化链为：

1. `rollout_ref_setup_dist`
2. `FSDPWorker.setup_dist()`
3. `rollout_ref_wg.init_model()`
4. `FSDPWorker.init_model()`
5. `_build_model_optimizer_dance_dis()` with `role="rollout_ref"`

关键位置：

- `verl/trainer/ray_trainer.py:398-418`
- `verl/workers/fsdp_workers.py:726-736`
- `verl/workers/fsdp_workers.py:853-894`

#### B. `actor_wg` 初始化链

设计语义下的 actor 侧初始化链为：

1. `actor_setup_dist`
2. `FSDPWorker.setup_dist()`
3. `actor_wg.init_model()`
4. `FSDPWorker.init_model()`
5. `_build_model_optimizer_dance_dis()` with `role="actor"`

关键位置：

- `verl/trainer/ray_trainer.py:388-418`
- `verl/workers/fsdp_workers.py:739-890`

---

## 4. 模型载入补充（Case 3 的角色差异）

### 4.1 `rollout_ref` 侧加载什么

`_build_model_optimizer_dance_dis()` 在 `role="rollout_ref"` 时，主要加载：

1. 可选 VideoAlign 奖励模型 `self.inferencer`
2. `self.rollout = load_transformer(...)`
3. `self.vae = load_vae(...)`
4. 将 `self.rollout.eval()`，作为 rollout 前向模型使用

这意味着 rollout/ref 组承担：

- diffusion 采样
- VAE decode
- 奖励打分

关键位置：

- `verl/workers/fsdp_workers.py:728-736`
- `verl/workers/fsdp_workers.py:853-894`

### 4.2 `actor` 侧加载什么

`_build_model_optimizer_dance_dis()` 在 `role="actor"` 且 `disco=false` 时，主要加载：

1. `self.transformer = load_transformer(...)`
2. 用 `process_group=self.actor_pg` 做 FSDP 包装
3. 创建 `self.optimizer`
4. 创建 `self.lr_scheduler`

此时 actor 侧的设计职责是：

- 持有可训练 actor 参数
- 重算 `new_log_probs`
- 执行 GRPO/PPO 更新

默认不承担：

- rollout 采样
- VAE decode
- VideoAlign 奖励推理

关键位置：

- `verl/workers/fsdp_workers.py:739-890`

### 4.3 VideoAlign 奖励模型如何载入

仅在 `role="rollout_ref"` 且 `worker.reward.use_videoalign=true` 时加载：

1. `_build_model_optimizer_dance_dis()` 中构造 `VideoVLMRewardInference(...)`
2. `VideoVLMRewardInference.__init__()` 会：
   - 读取 `model_config.json`
   - 创建 `DataConfig/ModelConfig/PEFTLoraConfig/TrainingConfig`
   - 调 `create_model_and_processor(...)`
   - 调 `load_model_from_checkpoint(...)`
   - `model.eval().to(device)`

关键位置：

- `verl/workers/fsdp_workers.py:731-736`
- `fastvideo/models/videoalign/inference.py:29-65`

### 4.4 Hunyuan 生成模型与 VAE 如何载入

1. `load_transformer(...)` 根据 `model_type` 加载 transformer  
2. 当 `model_type="hunyuan_hf"` 时：
   - `HunyuanVideoTransformer3DModel.from_pretrained(..., subfolder="transformer", torch_dtype=...)`
3. `load_vae(...)` 在 `model_type="hunyuan_hf"` 时：
   - `AutoencoderKLHunyuanVideo.from_pretrained(..., subfolder="vae", torch_dtype=torch.float32).to("cuda")`
   - 返回 `fps=24`

关键位置：

- `fastvideo/utils/load.py:253-300`
- `fastvideo/utils/load.py:303-345`

---

## 5. 每个训练 step 的主调用链（函数级）

`fit_dis()` 内每步核心链路按设计意图可概括为：

1. `next(iter(self.train_dataloader))` 取 batch
2. `DataProto.from_single_dict(batch_dict, meta_info={"caption": caption})`
3. 先做 actor -> rollout 权重对齐
4. `self.rollout_ref_wg.generate_sequences(new_batch)`
5. `self.actor_wg.update_actor(batch)`

对应当前代码中的主干位置：

- `verl/trainer/ray_trainer.py:854-893`

这里与 Case 4 的核心差异是：

- Case 4：同一组 `actor_rollout_ref_wg` 完成 rollout + update
- Case 3：`rollout_ref_wg` 和 `actor_wg` 分别负责 rollout 与 update

---

## 6. 权重同步在 Case 3 中的设计语义

### 6.1 为什么 Case 3 需要单独的参数对齐

由于 Case 3 不是 colocated 路径：

- rollout 侧前向使用的是 `self.rollout`
- actor 侧更新使用的是 `self.transformer`

因此从设计语义上讲，每轮 rollout 之前都应把 actor 最新参数同步到 rollout 侧，否则 rollout 采样与 actor 更新会基于不同版本的策略。

### 6.2 仓库里提供了哪些同步接口

当前仓库已经提供了这套同步能力：

1. `actor_wg.get_actor_weights_info()`
2. `rollout_ref_wg.set_actor_weights_info(weights_info)`
3. `actor_wg.sync_rollout_weights()`
4. `rollout_ref_wg.sync_rollout_weights()`

其中 `sync_rollout_weights()` 的核心语义是：

- actor 侧从 `self.transformer.state_dict()` 抽取参数
- 通过 `actor_rollout_pg` 从 actor 侧 rank 0 广播到 rollout 组
- rollout 侧用 `rollout_load_weights(...)` 写入 `self.rollout`

要让这套同步真正可用，还需要先完成：

- actor 侧 `get_actor_weights_info()`
- rollout 侧 `set_actor_weights_info(weights_info)`

因为 `sync_rollout_weights()` 内部会检查 `_weights_info` 是否已经准备好。

关键位置：

- `verl/trainer/ray_trainer.py:1173-1181`
- `verl/workers/fsdp_workers.py:2917-2944`
- `verl/workers/fsdp_workers.py:2957-2986`

因此，本文在主链路里把“rollout 前先完成 actor -> rollout 权重对齐”视为 Case 3 的设计前提。

---

## 7. 分发与回收机制（Case 3 的两组独立并行）

`generate_sequences` 与 `update_actor` 都是 `Dispatch.DP_COMPUTE_PROTO`：

- 输入 `DataProto` 会先按调用方 `WorkerGroup.world_size` 做 `chunk(world_size)`
- 每个 rank 执行对应 worker 函数
- 输出再由 `DataProto.concat()` 聚合回 driver

关键位置：

- `verl/workers/fsdp_workers.py:1947`
- `verl/workers/fsdp_workers.py:1534`
- `verl/single_controller/base/decorator.py:106-123`
- `verl/protocol.py:540-566`
- `verl/protocol.py:584-600`

Case 3 与 Case 4 的差别在于：

1. `rollout_ref_wg.generate_sequences(...)` 只在 rollout/ref 组内部切分与回收
2. `actor_wg.update_actor(...)` 只在 actor 组内部切分与回收
3. 两个阶段分别经过不同的 `WorkerGroup.world_size`

所以 driver 看到的是：

- 一次 rollout group 的并行采样
- 接着一次 actor group 的并行更新

而不是同一组 worker 连续执行两个阶段。

---

## 8. `rollout_ref` 侧 `FSDPWorker.generate_sequences()` 的 GRPO rollout 细节

函数入口：

- `verl/workers/fsdp_workers.py:1948`

### 8.1 输入整理

1. 从 `prompts.batch` 取：
   - `encoder_hidden_states`
   - `encoder_attention_mask`
2. 从 `prompts.meta_info` 取 `caption`
3. 若 `use_group=true`，按 `num_generations` 扩展条件与 caption

关键位置：

- `verl/workers/fsdp_workers.py:1999-2017`

### 8.2 Case 3 与 Case 4 的关键差异：前向模型是谁

`generate_sequences()` 内部对 diffusion 模型前向有两条分支：

- `self.colocated=True` 时，用 `self.transformer(...)`
- `self.colocated=False` 时，用 `self.rollout(...)`

Case 3 中由于 `role="rollout_ref"` 且 `self.colocated=False`，实际会走：

- `self.rollout.eval()`
- `self.rollout(...)`

这正是 disaggregate 语义下“rollout 使用独立 rollout 模型副本”的关键差异。

关键位置：

- `verl/workers/fsdp_workers.py:2075-2106`

### 8.3 噪声日程与采样循环

1. 构造 `sigma_schedule = linspace(1,0,sampling_steps+1)`
2. 经过 `sd3_time_shift()` 变换
3. 对每个样本逐步采样：
   - 前向得到 `model_pred`
   - `flux_step(...)` 计算 `z_{t+1}` 与该步 `log_prob`
   - 保存全路径 latent 与 logprob

关键位置：

- `verl/workers/fsdp_workers.py:2021-2024`
- `verl/workers/fsdp_workers.py:2063-2114`

### 8.4 解码与奖励

1. 取最终 `pred_original` 反标定后进 VAE decode
2. 导出视频 `./videos/hunyuan_{rank}_{index}.mp4`
3. 调 `self.inferencer.reward(...)` 得到 `VQ/MQ`

关键位置：

- `verl/workers/fsdp_workers.py:2111-2149`

### 8.5 返回结构

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

## 9. `actor` 侧 `FSDPWorker.update_actor()` 的 GRPO/PPO 更新细节

函数入口：

- `verl/workers/fsdp_workers.py:1535`

### 9.1 预处理

1. `optimizer.zero_grad()`
2. 将 `data.batch` 张量搬到当前 device
3. 从 `meta_info["sigma_schedule"]` 还原 tensor

关键位置：

- `verl/workers/fsdp_workers.py:1607-1617`

### 9.2 分组标准化 advantage

按 `num_generations` 分组，对每组分别计算：

- `vq_advantages = (vq_rewards - mean) / std`
- `mq_advantages = (mq_rewards - mean) / std`

关键位置：

- `verl/workers/fsdp_workers.py:1625-1648`

### 9.3 best-of-n 筛选

1. `total_scores = vq_coef * vq_advantages + mq_coef * mq_advantages`
2. 选 top `bestofn/2` + bottom `bestofn/2`
3. 打乱后保留（当 `num_generations != bestofn` 时）

关键位置：

- `verl/workers/fsdp_workers.py:1649-1665`

### 9.4 时间步随机化与训练步数截断

1. 对每个样本的 diffusion timestep 做随机排列 `perms`
2. `train_timesteps = int(T * timestep_fraction)`，只训练前这部分随机步

关键位置：

- `verl/workers/fsdp_workers.py:1666-1686`

### 9.5 GRPO one-step + PPO clipped objective

对每个样本、每个训练 timestep：

1. `grpo_one_step(...)` 用当前 actor 的 `self.transformer` 重算 `new_log_probs`
2. `ratio = exp(new_log_probs - old_log_probs)`
3. 分别对 `vq_adv`、`mq_adv` 算 clipped loss
4. `final_loss = vq_coef * vq_loss + mq_coef * mq_loss`
5. `final_loss.backward()`

关键位置：

- `verl/workers/fsdp_workers.py:1577-1606`
- `verl/workers/fsdp_workers.py:1692-1733`

### 9.6 梯度累计与参数更新

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

## 10. 当前实现与设计意图的偏差

这里需要特别区分“仓库当前主循环写法”和“Case 3 的设计语义”。

### 10.1 当前 `fit_dis()` 主循环没有显式做权重同步

当前 `fit_dis()` 的真实代码主链是：

1. `DataProto.from_single_dict(...)`
2. `self.rollout_ref_wg.generate_sequences(new_batch)`
3. `self.actor_wg.update_actor(batch)`

中间没有显式调用：

- `self.sync_weights()`
- 或 `actor_wg.sync_rollout_weights()` / `rollout_ref_wg.sync_rollout_weights()`
- 也没有先做 `get_actor_weights_info()` / `set_actor_weights_info(...)`

关键位置：

- `verl/trainer/ray_trainer.py:870-893`
- `verl/trainer/ray_trainer.py:1173-1181`
- `verl/trainer/ray_trainer.py:1263-1264`

### 10.2 为什么本文主线仍按“先同步再 rollout”来写

因为从 disaggregate 训练语义上，Case 3 的合理主线应是：

- actor 组先更新出最新参数
- rollout 组在下一轮采样前拿到这份参数
- 然后 rollout 轨迹再回传给 actor 组更新

也就是说：

- `sync_weights()` 是代码库已经准备好的协作机制
- 只是当前 `fit_dis()` 没把它显式接进主循环

因此本文主线按“设计意图”描述，而把这一点单独列为实现偏差。

---

## 11. 总体调用图（Case 3）

1. `main -> Runner.run`
2. `Runner.run -> RayPPOTrainer.init_workers_dis`
3. `init_workers_dis -> create_colocated_worker_cls(...).spawn(...)`
4. `init_workers_dis -> actor_setup_dist / rollout_ref_setup_dist -> FSDPWorker.setup_dist`
5. `init_workers_dis -> rollout_ref_wg.init_model -> FSDPWorker.init_model -> _build_model_optimizer_dance_dis(role="rollout_ref")`
6. `init_workers_dis -> actor_wg.init_model -> FSDPWorker.init_model -> _build_model_optimizer_dance_dis(role="actor")`
7. `Runner.run -> RayPPOTrainer.fit_dis`
8. `fit_dis -> DataLoader -> DataProto.from_single_dict`
9. `[设计意图] fit_dis -> sync_weights`
10. `fit_dis -> rollout_ref_wg.generate_sequences`
11. `WG dispatch(DP_COMPUTE_PROTO) -> per-rank FSDPWorker.generate_sequences -> concat`
12. `fit_dis -> actor_wg.update_actor`
13. `WG dispatch(DP_COMPUTE_PROTO) -> per-rank FSDPWorker.update_actor -> concat`
14. `fit_dis` 进入下一 step

---

## 12. 对这条链路的实现语义总结

1. 这是“**rollout 与 actor 分离部署**”的路径：采样发生在 `rollout_ref_wg`，更新发生在 `actor_wg`。  
2. `generate_sequences` 负责产出扩散轨迹级训练样本（含每步 `log_probs` 与奖励），`update_actor` 再基于这些轨迹做 PPO 风格的 GRPO 更新。  
3. Case 3 相比 Case 4 的本质差异，不在 GRPO 数学形式，而在执行拓扑：  
   - Case 4 是同组 worker 内部闭环  
   - Case 3 是 rollout 组与 actor 组跨组协作  
4. 因此 Case 3 的关键设计语义是：**采样前参数对齐，采样后轨迹回传给 actor 更新。**
