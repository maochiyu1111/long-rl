# Case 1: `disaggregate=true`、`pipelined_micro_batch=true`、`disco=false` 时 DanceGRPO 执行链路

## 1. 结论与范围

本文参照 `case4_disco_rl.md` 的口径，只分析当前代码里这组参数对应的**实际主训练链路**，忽略 `fit_disco_pipelined()` 里用于计时写文件后 `SystemExit` 的测试性提前退出逻辑。

主链路是：

`next(iter(train_dataloader)) -> _make_batch_prompts() -> (actor_wg/rollout_ref_wg).generate_sequences_asyn_dance(...) -> _get_datapro.remote(data_future) -> ray.wait/ray.get -> driver 侧 advantage + best-of-n -> actor_wg.update_actor_asyn(step_weight)`

在当前代码中，这组参数会走 **disaggregate + pipelined 异步链路**：

- 不走 `fit()`
- 不走 `fit_dis()`
- 不走 `fit_pipelined_single_window()`
- 走 `init_workers_dis() + fit_disco_pipelined()`

但这里有一个需要前置记住的关键风险：按当前代码直读，actor 在 `disco=false` 时不会构建 `self.rollout`、`self.vae`、`self.inferencer`，trainer 却仍会把前几个 prompt 提交给 `actor_wg.generate_sequences_asyn_dance(...)`。所以这条“主链路”在控制流上是成立的，在对象能力上却存在明显缺口；这点会直接影响你后面做 Case1 迁移时对 worker 职责边界的判断。

关键分支位置：

- `verl/trainer/main.py:39-57`
- `verl/trainer/main.py:76-79`
- `verl/trainer/main.py:108-113`
- `verl/trainer/ray_trainer.py:1183`

---

## 2. 入口与分支决策

### 2.1 入口调用

1. `main()` 组装配置并启动 Ray
2. `Runner.run(config)` 创建 `RayPPOTrainer` 并进入训练

关键位置：

- `verl/trainer/main.py:121`
- `verl/trainer/main.py:34`

### 2.2 参数如何决定执行路径

当 `trainer.disaggregate=true` 时，`Runner.run()` 走 disaggregate 分支：

- 角色映射：`Role.RolloutRef` + `Role.Actor` + `Role.Critic`
- 资源池映射：`RolloutRef -> pool_fast`，`Actor/Critic -> pool_slow`
- 执行：`trainer.init_workers_dis()`

随后分支判断：

- `worker.actor.disco=true` 与 `trainer.pipelined_micro_batch=true` 互斥
- 只要 `trainer.pipelined_micro_batch=true`，就调用 `trainer.fit_disco_pipelined()`
- 不会进入 `fit_dis()`

关键位置：

- `verl/trainer/main.py:41-57`
- `verl/trainer/main.py:76-79`
- `verl/trainer/main.py:108-113`

补充：

- `RayPPOTrainer` 在 `adv_estimator != GAE` 时会令 `self.use_critic=False`
- `init_workers_dis()` 当前也只实际创建 `rollout_ref` 与 `actor` 两组 worker

关键位置：

- `verl/trainer/ray_trainer.py:221-224`
- `verl/trainer/ray_trainer.py:334-350`

---

## 3. 初始化阶段调用链

### 3.1 Driver 侧（`init_workers_dis`）

1. `RayPPOTrainer.init_workers_dis()`
2. 创建两个 WorkerGroup：`self.rollout_ref_wg`、`self.actor_wg`
3. 规划统一全局 rank：`actor_ranks + rollout_ranks`
4. 分别下发 `actor_setup_dist` / `rollout_ref_setup_dist`
5. 先 `rollout_ref_wg.init_model()`，再 `actor_wg.init_model()`

关键位置：

- `verl/trainer/ray_trainer.py:334-418`

### 3.2 Worker 侧分布式组初始化（`setup_dist`）

`setup_dist()` 会：

1. 设置 `MASTER_ADDR/MASTER_PORT/RANK/WORLD_SIZE/LOCAL_RANK`
2. `dist.init_process_group(backend="nccl", init_method="env://")`
3. 创建：
   - `actor_pg`
   - `rollout_ref_pg`
   - `actor_rollout_pg`（`actor` 的 rank0 + 全部 `rollout_ref`，后端是 `gloo`）
4. 按角色把 `self.ranks` 绑定到对应组

关键位置：

- `verl/trainer/ray_trainer.py:388-410`
- `verl/workers/fsdp_workers.py:240-279`

### 3.3 `init_model()` 实际走到哪里

`FSDPWorker.init_model()` 内部根据 `self.colocated` 分支：

- 本 Case 下 `actor` / `rollout_ref` 都是解耦 worker，`self.colocated=False`
- 因此统一走 `_build_model_optimizer_dance_dis()`

关键位置：

- `verl/workers/fsdp_workers.py:1317-1401`

注意：虽然代码里有 `init_model_disco()`，但 `init_workers_dis()` 并没有调用它，对应分支还被注释掉了，所以本 Case 实际不会走 `init_model_disco()`。

关键位置：

- `verl/trainer/ray_trainer.py:411-418`
- `verl/workers/fsdp_workers.py:1402-1452`

### 3.4 模型载入补充（disaggregate、但 `disco=false`）

#### A. `rollout_ref` 侧

`_build_model_optimizer_dance_dis()` 在 `role=="rollout_ref"` 时：

1. 若 `worker.reward.use_videoalign=true`，构造 `VideoVLMRewardInference`
2. 加载 `self.rollout = load_transformer(..., torch.bfloat16)`
3. `self.rollout = transformer.to(self.device)` 并在后续 `eval()`
4. 加载 VAE

关键位置：

- `verl/workers/fsdp_workers.py:729-737`
- `verl/workers/fsdp_workers.py:853-860`
- `verl/workers/fsdp_workers.py:891-893`

#### B. `actor` 侧

`_build_model_optimizer_dance_dis()` 在 `role=="actor"` 且 `disco=false` 时：

1. **不会**进入 `if self.config.worker.actor.disco == True` 分支
2. 因而**不会**额外构建 `self.rollout`
3. 也**不会**在该分支里加载 `self.vae` / `self.inferencer`
4. 实际只构建训练用 `self.transformer`（FSDP）+ optimizer + lr_scheduler

关键位置：

- `verl/workers/fsdp_workers.py:739-797`
- `verl/workers/fsdp_workers.py:797-890`

#### C. Hunyuan / VAE / VideoAlign 的底层载入

当 `model_type="hunyuan_hf"` 时：

- transformer: `HunyuanVideoTransformer3DModel.from_pretrained(..., subfolder="transformer")`
- vae: `AutoencoderKLHunyuanVideo.from_pretrained(..., subfolder="vae")`

VideoAlign 推理器构造时会：

- 读取 `model_config.json`
- 调 `create_model_and_processor(...)`
- 调 `load_model_from_checkpoint(...)`
- `model.eval().to(device)`

关键位置：

- `fastvideo/utils/load.py:253-300`
- `fastvideo/utils/load.py:303-345`
- `fastvideo/models/videoalign/inference.py:29-65`

---

## 4. 每个训练 step 的主调用链（`fit_disco_pipelined`，且 `pipelined_micro_batch=true`）

`fit_disco_pipelined()` 在本配置下每步核心链路是：

1. `_make_batch_prompts()` 从 dataloader 取 batch，并拆成多个 `DataProto`
2. 固定 `actor_batch_num=2`
3. 前 `actor_batch_num` 个 prompt 交给 `actor_wg.generate_sequences_asyn_dance(...)`
4. 其余 prompt 交给 `rollout_ref_wg.generate_sequences_asyn_dance(...)`
5. 所有异步返回值先经 `_get_datapro.remote(data_future)` 触发 `.get()`
6. `ray.wait` 按完成顺序回收 rollout 结果
7. 在 driver 侧计算：
   - `vq/mq` 标准化 advantage
   - `best-of-n` 筛选
8. 满足拼批条件后 `DataProto.concat(batches_wait)`
9. 将样本发给 `actor_wg.update_actor_asyn(samples)`，并用 `step_weight` 标记是否执行 optimizer step
10. 等全部 `accum_futs` 完成后，结束当前 global step

补充一个容易忽略但很影响语义的细节：这里是 `next(iter(train_dataloader))`，也就是每个 step 都重新构造一次 dataloader iterator，而不是持有一个跨 step 递进的持久 iterator。

关键位置：

- `verl/trainer/ray_trainer.py:1195-1246`
- `verl/trainer/ray_trainer.py:1276-1288`
- `verl/trainer/ray_trainer.py:1321-1445`

补充：

- 虽然这是 `pipelined_micro_batch=true` 的 Case，但当前入口仍然复用 `fit_disco_pipelined()`，并没有走专门的 `fit_pipelined_single_window()`
- 函数里带有 `output_pipe.txt` + `SystemExit` 的计时出口，本文按主训练语义忽略

关键位置：

- `verl/trainer/ray_trainer.py:1283-1305`

### 4.1 `_make_batch_prompts()` 如何拆 batch

`_make_batch_prompts()` 有两种拆法：

1. 若 `num_generations % rollout_ref_wg.world_size != 0`
   - 要求 `train_batch_size % rollout_ref_wg.world_size == 0`
   - 以 `rollout_ref_wg.world_size` 为单位切 batch
   - `meta_info["use_seed"] = False`
2. 否则
   - 对 train batch 中每条样本单独构造一个 prompt batch
   - 每条样本重复 `num_generations` 次
   - 写入显式 `seed`
   - `meta_info["use_seed"] = True`

这里描述的是 `_make_batch_prompts()` 自身显式写死的切批条件；真正要让后续异步链跑通，还要再叠加第 5 节提到的 actor 路 world size 约束，因为前几个 prompt 最终会被分发给 `actor_wg`。

关键位置：

- `verl/trainer/ray_trainer.py:1195-1243`

---

## 5. 分发与回收机制（DP_COMPUTE_PROTO + 非阻塞 Future）

`generate_sequences_asyn_dance` 与 `update_actor_asyn` 都是：

- `Dispatch.DP_COMPUTE_PROTO`
- `blocking=False`

因此调用语义是：

1. 输入 `DataProto` 会按目标 worker group 的 `world_size` 切分
2. 各 rank 异步执行同名 worker 函数
3. 返回值先聚合成 `DataProtoFuture`
4. 只有显式调用 `.get()` 时，才真正 `ray.get(...)` 并 `DataProto.concat(...)`

关键位置：

- `verl/workers/fsdp_workers.py:1744`
- `verl/workers/fsdp_workers.py:2231`
- `verl/single_controller/base/decorator.py:106-123`
- `verl/protocol.py:540-566`
- `verl/protocol.py:653-700`

这也是 `fit_disco_pipelined()` 里单独定义 `_get_datapro.remote(data_future)` 并在里面执行 `data.get()` 的原因。

关键位置：

- `verl/trainer/ray_trainer.py:1244-1246`

补充：因为前 `actor_batch_num` 个 prompt 会发到 `actor_wg`，后面的 prompt 会发到 `rollout_ref_wg`，所以 prompt batch 的 batch 维实际上需要同时满足两组 world size 的切分要求；这不是一个单纯只对 `rollout_ref_wg` 生效的约束。

---

## 6. `FSDPWorker.generate_sequences_asyn_dance()` 的 rollout 细节

函数入口：

- `verl/workers/fsdp_workers.py:2231-2494`

### 6.1 输入整理

1. 从 `prompts.batch` 取：
   - `encoder_hidden_states`
   - `encoder_attention_mask`
   - 可选 `seed`
2. 从 `prompts.non_tensor_batch` 取 `caption`
3. 根据 `meta_info["use_seed"]` 决定：
   - 读显式 seed
   - 或按 group 扩展并使用内部生成的 seed 规则
4. 若 `use_group=true` 且当前走 group 扩展逻辑，则按 `num_generations` 重复条件与 caption

关键位置：

- `verl/workers/fsdp_workers.py:2298-2324`
- `verl/workers/fsdp_workers.py:2361-2368`

### 6.2 扩散采样

1. 构造 `sigma_schedule = linspace(1, 0, sampling_steps + 1)`
2. 用 `sd3_time_shift()` 做时间变换
3. 逐 timestep：
   - 用 `self.rollout(...)` 预测 `model_pred`
   - 调 `flux_step(..., grpo=True, sde_solver=True)` 得到 `next_latents` 与该步 `log_prob`
   - 保存整条 latent 轨迹与逐步 logprob

关键位置：

- `verl/workers/fsdp_workers.py:2325-2335`
- `verl/workers/fsdp_workers.py:2375-2409`

### 6.3 解码与奖励

1. 取最终 `pred_original` 反标定后进 VAE decode
2. 导出视频 `./videos/hunyuan_{rank}_{index}.mp4`
3. 调 `self.inferencer.reward(...)` 得到 `VQ/MQ`（可带归一化）

关键位置：

- `verl/workers/fsdp_workers.py:2410-2448`

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

- `verl/workers/fsdp_workers.py:2461-2494`

---

## 7. `FSDPWorker.update_actor_asyn()` 的更新细节

函数入口：

- `verl/workers/fsdp_workers.py:1744-1937`

### 7.1 预处理

1. `optimizer.zero_grad()`
2. 将 `data.batch` 张量搬到当前 device
3. 读取 `meta_info["step_weight"]`
4. 从 `meta_info["sigma_schedule"]` 恢复 tensor

关键位置：

- `verl/workers/fsdp_workers.py:1830-1840`

### 7.2 时间步打乱与采样子集

1. 对每个样本的 timestep 随机打乱
2. 对 `timesteps/latents/next_latents/log_probs` 按相同置换重排
3. `train_timesteps = int(T * timestep_fraction)`

关键位置：

- `verl/workers/fsdp_workers.py:1848-1872`

### 7.3 PPO / GRPO 一步更新

对每个样本、每个选中的 timestep：

1. `grpo_one_step(...)` 用 `self.transformer` 重算 `new_log_probs`
2. 计算 `ratio = exp(new_log_probs - old_log_probs)`
3. 分别对 `vq_advantages` / `mq_advantages` 做 clipped PPO loss
4. `final_loss = vq_coef * vq_loss + mq_coef * mq_loss`
5. `final_loss.backward()`

关键位置：

- `verl/workers/fsdp_workers.py:1787-1816`
- `verl/workers/fsdp_workers.py:1873-1923`

### 7.4 `step_weight` 的作用

仅当 `step_weight=True` 时才执行：

- `clip_grad_norm_`
- `optimizer.step()`
- `lr_scheduler.step()`
- `optimizer.zero_grad()`

关键位置：

- `verl/workers/fsdp_workers.py:1924-1930`

注意：虽然 driver 侧用 `step_weight` 表达“累计若干批再 step”的意图，但 `update_actor_asyn()` 一进入函数就会先执行一次 `optimizer.zero_grad()`。因此跨多次 `update_actor_asyn()` 调用并不会真正保留前一次调用产生的梯度；从代码实际语义看，更像是“多次独立 backward，其中只有打了 `step_weight=True` 的那次会真正 `step()`”。

---

## 8. 总体调用图（Case 1）

1. `main -> Runner.run`
2. `Runner.run(disaggregate=true) -> RayPPOTrainer.init_workers_dis`
3. `init_workers_dis -> actor/rollout_ref setup_dist -> FSDPWorker.setup_dist`
4. `init_workers_dis -> rollout_ref_wg.init_model -> _build_model_optimizer_dance_dis(role=rollout_ref)`
5. `init_workers_dis -> actor_wg.init_model -> _build_model_optimizer_dance_dis(role=actor, disco=false)`
6. `Runner.run -> RayPPOTrainer.fit_disco_pipelined`
7. `fit_disco_pipelined -> _make_batch_prompts`
8. `fit_disco_pipelined -> actor_wg/rollout_ref_wg.generate_sequences_asyn_dance`
9. `fit_disco_pipelined -> _get_datapro.remote(data_future) -> ray.wait/ray.get`
10. `fit_disco_pipelined -> driver 侧 vq/mq advantage 标准化 + best-of-n`
11. `fit_disco_pipelined -> actor_wg.update_actor_asyn(step_weight)`

---

## 9. 这条链路的关键实现语义

1. 这是“**disaggregate 双组 worker + pipelined 异步提交**”路径，但 trainer 入口复用的是 `fit_disco_pipelined()`，不是单独的 pipeline trainer。
2. 与 `case4_disco_rl.md` 不同，本 Case 的 `vq/mq advantage` 标准化和 `best-of-n` 筛选发生在 **driver**，`update_actor_asyn()` 直接消费整理好的训练样本。
3. `generate_sequences_asyn_dance()` 始终依赖 `self.rollout`、`self.vae`，且奖励路径依赖 `self.inferencer`。
4. 但本 Case 的 actor 初始化分支在 `disco=false` 时并不会构建这些 rollout 侧对象；与此同时，`fit_disco_pipelined()` 又固定把前 `actor_batch_num=2` 个 prompt 提交给 `actor_wg.generate_sequences_asyn_dance(...)`。因此按当前代码直读，这条链路存在明显的运行时属性缺失风险，需要结合你本地是否还有未展示补丁一起看。
