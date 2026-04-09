# Case 2: `disaggregate=true`、`pipelined_micro_batch=false`、`disco=true` 时 DanceGRPO 执行链路

## 1. 结论与范围

本文只分析你关心的主训练链路（忽略 `fit_disco_pipelined()` 里用于计时写文件后 `SystemExit` 的测试性分支），即：

`next(iter(train_dataloader)) -> _make_batch_prompts() -> (actor_wg/rollout_ref_wg).generate_sequences_asyn_dance(...) -> _get_datapro.remote(data_future) -> ray.wait/ray.get -> driver 侧 advantage + best-of-n -> actor_wg.update_actor_asyn(...)`

在当前代码中，这组参数会走 **disaggregate + disco 异步链路**：

- 不走 `fit()`
- 不走 `fit_dis()`
- 走 `init_workers_dis()` + `fit_disco_pipelined()`（即使 `pipelined_micro_batch=false`）

关键分支位置：

- `verl/trainer/main.py:39-57`
- `verl/trainer/main.py:108-113`

---

## 2. 入口与分支决策

### 2.1 入口调用

1. `main()` 组装配置并启动 Ray
2. `Runner.run(config)` 创建 `RayPPOTrainer` 并进入训练

关键位置：

- `verl/trainer/main.py:96-116`
- `verl/trainer/main.py:121-151`

### 2.2 参数如何决定执行路径

当 `trainer.disaggregate=true` 时，`Runner.run()` 走 disaggregate 分支：

- 角色映射为 `Role.RolloutRef` + `Role.Actor` + `Role.Critic`
- 资源池映射为 `RolloutRef -> pool_fast`，`Actor/Critic -> pool_slow`
- 实际执行 `trainer.init_workers_dis()`

随后分支判断：

- 只要 `config.worker.actor.disco=true`，就调用 `trainer.fit_disco_pipelined()`
- 不会进入 `fit_dis()`

关键位置：

- `verl/trainer/main.py:39-57`
- `verl/trainer/main.py:108-113`

补充：

- 在 `RayPPOTrainer.__init__()` 中，`adv_estimator != GAE` 时 `self.use_critic=False`
- 虽然 `role_worker_mapping` 里声明了 `Critic`，但 `init_workers_dis()` 实际只创建 `actor` 与 `rollout_ref` 两组 worker

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

- `verl/workers/fsdp_workers.py:240-279`

### 3.3 `init_model()` 实际走到哪里

`FSDPWorker.init_model()` 内部根据 `self.colocated` 分支：

- 本 Case 下 `actor` / `rollout_ref` 都是解耦 worker，`self.colocated=False`
- 因此统一走 `_build_model_optimizer_dance_dis()`

关键位置：

- `verl/workers/fsdp_workers.py:1316-1401`

注意：代码里虽然有 `init_model_disco()`，但 `init_workers_dis()` 并没有调用它，对应分支还被注释掉了，所以该 Case 实际不会走 `init_model_disco()`。

关键位置：

- `verl/trainer/ray_trainer.py:411-418`
- `verl/workers/fsdp_workers.py:1402-1452`

### 3.4 模型载入补充（disaggregate + disco）

#### A. `rollout_ref` 侧

`_build_model_optimizer_dance_dis()` 在 `role=="rollout_ref"` 时：

1. 若 `worker.reward.use_videoalign=true`，构造 `VideoVLMRewardInference`
2. 加载 `self.rollout = load_transformer(..., torch.bfloat16)`
3. `self.rollout = transformer.to(self.device)` 并在后续 `eval()`
4. 加载 VAE

关键位置：

- `verl/workers/fsdp_workers.py:728-737`
- `verl/workers/fsdp_workers.py:853-860`
- `verl/workers/fsdp_workers.py:891-893`

#### B. `actor` 侧

`_build_model_optimizer_dance_dis()` 在 `role=="actor"` 且 `disco=true` 时：

1. 额外加载一份 `self.rollout`（bf16）供 actor 路 rollout
2. 对 rollout block 做 `fully_shard(...)` 并显式 `reshard()`
3. 加载训练用 `self.transformer`
4. 将训练用 transformer 包成 `FSDP(process_group=self.actor_pg)`
5. 初始化 optimizer / lr_scheduler
6. 加载 VAE
7. 若开启 `VideoAlign`，也会加载 `self.inferencer`

关键位置：

- `verl/workers/fsdp_workers.py:739-797`
- `verl/workers/fsdp_workers.py:797-852`
- `verl/workers/fsdp_workers.py:862-893`

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

## 4. 每个训练 step 的主调用链（`fit_disco_pipelined`，且 `pipelined_micro_batch=false`）

`fit_disco_pipelined()` 在本配置下每步核心链路是：

1. `_make_batch_prompts()` 从 dataloader 取 batch，并拆成多个 `DataProto`
2. 前 `actor_batch_num` 个 prompt 交给 `actor_wg.generate_sequences_asyn_dance(...)`
3. 其余 prompt 交给 `rollout_ref_wg.generate_sequences_asyn_dance(...)`
4. 所有异步返回值先经 `_get_datapro.remote(data_future)` 触发 `.get()`
5. `ray.wait` 按完成顺序回收 rollout 结果
6. 在 driver 侧计算：
   - `vq/mq` 标准化 advantage
   - `best-of-n` 筛选
7. 满足拼批条件后 `DataProto.concat(batches_wait)`
8. 将样本发给 `actor_wg.update_actor_asyn(samples)`
9. 等全部 `accum_futs` 完成后，结束当前 global step

关键位置：

- `verl/trainer/ray_trainer.py:1183-1246`
- `verl/trainer/ray_trainer.py:1318-1445`
- `verl/trainer/ray_trainer.py:1447-1462`

补充：虽然函数名叫 `fit_disco_pipelined()`，但这里并不是 `pipelined_micro_batch=true` 的 Case；当前只是因为 `disco=true`，也复用了这条异步训练主循环。

补充：函数里带有 `output_disco.txt` + `SystemExit` 的计时出口，本文按主训练语义忽略。

关键位置：

- `verl/trainer/ray_trainer.py:1283-1300`

---

## 5. 分发与回收机制（DP_COMPUTE_PROTO + 非阻塞 Future）

`generate_sequences_asyn_dance` 与 `update_actor_asyn` 都是：

- `Dispatch.DP_COMPUTE_PROTO`
- `blocking=False`

因此调用语义是：

1. 输入 `DataProto` 会按 `world_size` 在 batch 维切分
2. 各 rank 异步执行同名 worker 函数
3. 返回值先聚合成 `DataProtoFuture`
4. 只有显式调用 `.get()` 时，才真正 `ray.get(...)` 并 `DataProto.concat(...)` 变回单个 `DataProto`

关键位置：

- `verl/workers/fsdp_workers.py:2230`
- `verl/workers/fsdp_workers.py:1743`
- `verl/single_controller/base/decorator.py:118-123`
- `verl/single_controller/ray/base.py:42-49`
- `verl/protocol.py:540-566`
- `verl/protocol.py:653-700`

这也是 `fit_disco_pipelined()` 里单独定义 `_get_datapro.remote(data_future)` 并在里面执行 `data.get()` 的原因。

关键位置：

- `verl/trainer/ray_trainer.py:1244-1246`

---

## 6. `FSDPWorker.generate_sequences_asyn_dance()` 的 disco rollout 细节

函数入口：

- `verl/workers/fsdp_workers.py:2230-2494`

### 6.1 输入整理

1. 从 `prompts.batch` 取：
   - `encoder_hidden_states`
   - `encoder_attention_mask`
   - 可选 `seed`
2. 从 `prompts.non_tensor_batch` 取 `caption`
3. 根据 `meta_info["use_seed"]` 决定是使用显式 seed 还是按 group 扩展
4. 若 `use_group=true` 且当前走 group 扩展逻辑，则按 `num_generations` 重复条件与 caption

关键位置：

- `verl/workers/fsdp_workers.py:2298-2324`

### 6.2 扩散采样

1. 构造 `sigma_schedule = linspace(1, 0, sampling_steps + 1)`
2. 用 `sd3_time_shift()` 做时间变换
3. 逐 timestep：
   - 用 `self.rollout(...)` 预测 `model_pred`
   - 调 `flux_step(..., grpo=True, sde_solver=True)` 得到 `next_latents` 与该步 `log_prob`
   - 保存整条 latent 轨迹与逐步 logprob

关键位置：

- `verl/workers/fsdp_workers.py:2325-2335`
- `verl/workers/fsdp_workers.py:2375-2412`
- `verl/workers/fsdp_workers.py:2237-2276`

### 6.3 解码与奖励

1. 取最终 `pred_original`，做尺度还原后送入 VAE decode
2. 导出视频到 `./videos/hunyuan_{rank}_{index}.mp4`
3. 若 `use_videoalign=true`，调用 `self.inferencer.reward(...)` 得到 `VQ/MQ`

关键位置：

- `verl/workers/fsdp_workers.py:2417-2448`
- `fastvideo/models/videoalign/inference.py:227-254`

### 6.4 返回结构

返回 `DataProto`（随后会 `.to("cpu")`）包含：

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

- `verl/workers/fsdp_workers.py:2461-2493`

补充：当 `self.role=="actor"` 时，生成结束后会显式对 actor 侧 `self.rollout` 做 `reshard()`。

关键位置：

- `verl/workers/fsdp_workers.py:2475-2480`

---

## 7. `FSDPWorker.update_actor_asyn()` 的更新细节

函数入口：

- `verl/workers/fsdp_workers.py:1743-1937`

### 7.1 输入前提

该函数依赖上游在 driver 侧已经写好的：

- `samples.batch["vq_advantages"]`
- `samples.batch["mq_advantages"]`
- `samples.meta_info["step_weight"]`

也就是说，这个 Case 里 advantage 标准化和 best-of-n 不在 worker 内做，而是在 `fit_disco_pipelined()` 的 driver 侧完成后再送来更新。

关键位置：

- `verl/trainer/ray_trainer.py:1372-1407`
- `verl/trainer/ray_trainer.py:1432-1438`

### 7.2 训练核心

1. `optimizer.zero_grad()`
2. 将 `data.batch` 搬到当前 device
3. 从 `meta_info["sigma_schedule"]` 恢复 tensor
4. 随机打乱 timestep 顺序
5. `train_timesteps = int(T * timestep_fraction)`，只训练前一部分随机步
6. 对每个样本、每个训练 timestep：
   - `grpo_one_step()` 用当前 `self.transformer` 重算 `new_log_probs`
   - `ratio = exp(new_log_probs - old_log_probs)`
   - 分别基于 `vq_advantages`、`mq_advantages` 计算 clipped loss
   - `final_loss = vq_coef * vq_loss + mq_coef * mq_loss`
   - `final_loss.backward()`

关键位置：

- `verl/workers/fsdp_workers.py:1817-1848`
- `verl/workers/fsdp_workers.py:1851-1890`
- `verl/workers/fsdp_workers.py:1892-1923`

### 7.3 何时 step

- 仅当 `step_weight=True` 时执行：
  - `clip_grad_norm_`
  - `optimizer.step()`
  - `lr_scheduler.step()`
  - `optimizer.zero_grad()`

关键位置：

- `verl/workers/fsdp_workers.py:1924-1930`

返回：

- `DataProto(non_tensor_batch={"actor_loss": ...})`

关键位置：

- `verl/workers/fsdp_workers.py:1933-1937`

---

## 8. 总体调用图（Case 2）

1. `main -> Runner.run`
2. `Runner.run(disaggregate=true) -> RayPPOTrainer.init_workers_dis`
3. `init_workers_dis -> actor_setup_dist / rollout_ref_setup_dist`
4. `rollout_ref_wg.init_model -> FSDPWorker.init_model -> _build_model_optimizer_dance_dis(role=rollout_ref)`
5. `actor_wg.init_model -> FSDPWorker.init_model -> _build_model_optimizer_dance_dis(role=actor, disco=true)`
6. `Runner.run -> RayPPOTrainer.fit_disco_pipelined`
7. `fit_disco_pipelined -> _make_batch_prompts`
8. `fit_disco_pipelined -> actor_wg/rollout_ref_wg.generate_sequences_asyn_dance`
9. `DataProtoFuture.get -> ray.wait/ray.get -> driver 侧 advantage + best-of-n -> DataProto.concat`
10. `fit_disco_pipelined -> actor_wg.update_actor_asyn(step_weight)`
11. `wait accum_futs -> 进入下一 global_step`

---

## 9. 对这条链路的实现语义总结

1. 这是“**actor/rollout_ref 解耦 + disco 双路 rollout**”路径：rollout 任务会同时分发到 actor 组与 rollout_ref 组。  
2. 与 `case4_disco_rl.md` 不同，本 Case 的 `vq/mq advantage` 标准化和 `best-of-n` 筛选发生在 **driver**，`update_actor_asyn()` 直接消费整理好的训练样本。  
3. 这条链路的异步性来自“`DP_COMPUTE_PROTO + blocking=False + DataProtoFuture.get()`”：先并发 rollout，再按完成顺序回收，拼够一批后再异步推进 actor 更新。  
