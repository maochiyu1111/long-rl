# Case 4（long-rl）：`disaggregate=false`、`pipelined_micro_batch=false`、`disco=false` 时 DanceGRPO 调用链

## 1. 总结（先回答你最关心的 3 点）

1. `disco` 里的 `_build_model_optimizer_dance` 在 `long-rl` **没有同名实现**。  
   `long-rl` 走的是 `ActorRolloutRefWorker._build_model_optimizer()` -> `_build_model_optimizer_diffusion()`；
   - **VideoAlign**：有，但不在 worker 初始化里加载，而是迁到 `reward manager(batch)` 侧按配置加载。
   - **Hunyuan**：当前主链路里没有对应加载/rollout分支（仅 `Wan` / `StableDiffusion3`）。

2. `disco` 的 `fit -> actor_rollout_ref_wg.generate_sequences` 在 `long-rl`（本 case）对应为  
   `fit -> actor_rollout_wg.generate_sequences`，入口在 `verl/workers/fsdp_workers.py`，但这里只是包装层，真实扩散生成在 `verl/workers/rollout/diffusion_rollout.py`。

3. `disco` 的 `update_actor`（写在 fsdp_worker）在 `long-rl` 对应为  
   `fsdp_workers.py:update_actor()`（包装）-> `DataParallelPPOActor.update_policy()` -> `dp_actor.py:_update_policy_diffusion_dancegrpo()`（核心 dance 更新）。

---

## 2. 本 case 在 long-rl 的主链路（是否有对应调用链）

有对应链路，但实现拆层了。

### 2.1 入口与分支

- `main_ppo.py` 读取：
  - `fit_disaggregate = trainer.disaggregate`
  - `actor_disco = actor.disco`
  - `pipelined_micro_batch = trainer.pipelined_micro_batch`  
  见 `verl/trainer/main_ppo.py:177-184`。

- 当 `fit_disaggregate=false`（即你这个 case）：
  - `trainer.init_workers()`
  - `trainer.fit()`  
  见 `verl/trainer/main_ppo.py:349-359`。

### 2.2 初始化链

- `init_workers()` 中，colocated 路径创建 `actor_rollout_wg` 并 `init_model()`：
  - `verl/trainer/ppo/ray_trainer.py:2291-2299`
  - `verl/trainer/ppo/ray_trainer.py:2392-2393`

- Worker 侧 `init_model()` 调用：
  - `_build_model_optimizer()`（若 diffusion=True 会转 `_build_model_optimizer_diffusion()`）
  - `_build_rollout()`（若 diffusion=True 会转 `_build_rollout_diffusion()`）  
  见 `verl/workers/fsdp_workers.py:1292-1360`、`719-737`、`974-1123`、`1125-1129`、`1231-1245`。

### 2.3 训练 step 主链（fit）

- `fit()` 内：
  - `DataProto.from_single_dict(batch_dict)`
  - `gen_batch = batch.pop(...)`
  - `actor_rollout_wg.generate_sequences(gen_batch)`
  - reward 计算
  - `compute_advantage_diffusion(...)`
  - `actor_rollout_wg.update_actor(batch)`  
  见 `verl/trainer/ppo/ray_trainer.py:2780-2840`、`2888-3034`、`3053-3079`。

---

## 3. 关注点 1：`_build_model_optimizer_dance` / VideoAlign / Hunyuan

### 3.1 long-rl 是否有 `_build_model_optimizer_dance`

没有。仓库中无该符号。模型初始化统一走：
- `_build_model_optimizer()`
- `_build_model_optimizer_diffusion()`（diffusion 分支）

对应位置：`verl/workers/fsdp_workers.py:719`、`974`。

### 3.2 VideoAlign 在 long-rl 有没有

有，但位置改变：

- 不是 worker 初始化时直接构造 inferencer；
- 而是 `reward manager` 侧按 `reward_kwargs` 决定 backend，并在 `batch` reward manager 中初始化 `VideoVLMRewardInference`。

关键路径：
- `main_ppo.py` 调 `load_reward_manager(...)`：`verl/trainer/main_ppo.py:319-324`
- `load_reward_manager(...)`：`verl/trainer/ppo/reward.py:103-159`
- `BatchRewardManager._init_videoalign_backend()`：`verl/workers/reward_manager/batch.py:77-133`

### 3.3 Hunyuan 在 long-rl 有没有

当前这条主链路里没有对应支持：

- `_build_model_optimizer_diffusion()` 仅按 `model_path` 路由：
  - 含 `wan` -> `WanPipeline`
  - 否则 -> `StableDiffusion3Pipeline`  
  见 `verl/workers/fsdp_workers.py:990-996`。

- rollout 构建也仅支持：
  - `WanRollout`
  - `StableDiffusionRollout`  
  见 `verl/workers/fsdp_workers.py:1238-1243`。

- `fastvideo` 目录当前仅有 `videoalign`，无 `hunyuan` 子模块。

### 3.4 结论：相较 disco 还缺什么

相较 disco 的 `_build_model_optimizer_dance` 语义，long-rl 当前缺/变更点：

- 缺 Hunyuan 模型加载与对应 rollout 路由。
- VideoAlign 从 worker 内同步加载改成 reward manager 异步/外部加载链。
- 生成侧不再是 disco 那种 fsdp_worker 内部直接做“采样+videoalign reward 一体化”。

---

## 4. 关注点 2：`fit -> ...generate_sequences` 在 long-rl 写在哪里？差异是什么？

### 4.1 写在哪里

- 训练循环调用点（本 case）：
  - `self.actor_rollout_wg.generate_sequences(gen_batch)`  
    `verl/trainer/ppo/ray_trainer.py:2839`

- Worker 入口：
  - `ActorRolloutRefWorker.generate_sequences()`  
    `verl/workers/fsdp_workers.py:1468-1509`

- 真正扩散 rollout 实现：
  - `StableDiffusionRollout.generate_sequences()`：`verl/workers/rollout/diffusion_rollout.py:147-236`
  - `WanRollout.generate_sequences()`：`verl/workers/rollout/diffusion_rollout.py:282-360`

### 4.2 与 disco 的差异

- `disco`：`generate_sequences` 在 fsdp_worker 内实现完整 dance rollout（含轨迹、logprob、视频导出、VideoAlign 奖励）。
- `long-rl`：`fsdp_workers.py` 只做包装/分发，核心生成在 `diffusion_rollout.py`。
- 奖励分离：
  - rollout 先返回轨迹与占位 reward（`vq_rewards/mq_rewards` 初始可为占位），
  - 再由 trainer 侧 `compute_reward(...)` + `_inject_dual_rewards_from_sources(...)` 注入真实 VQ/MQ，
  - 最后 `compute_advantage_diffusion(...)` 计算 dual advantage。  
  见 `ray_trainer.py:2888-3034`、`328-505`。

- 另一个关键差异（当前代码现状）：
  - `diffusion_rollout._is_dance_mode()` 依赖 `prompts.meta_info["diffusion_algo"]`（`diffusion_rollout.py:37-39`），
  - 但 `fit()` 主路径里 diffusion 分支仅写了 `global_steps`，未写 `diffusion_algo/use_seed`（`ray_trainer.py:2793-2795`）。
  - 而 `fit_dis` 辅助路径 `_make_batch_data_dis()` 是会写 `diffusion_algo/use_seed` 的（`ray_trainer.py:1514-1517`）。

---

## 5. 关注点 3：`update_actor` 在 long-rl 写在哪里？差异是什么？

### 5.1 写在哪里

- 训练循环调用点：
  - `self.actor_rollout_wg.update_actor(batch)`  
    `verl/trainer/ppo/ray_trainer.py:3079`

- Worker 包装入口：
  - `ActorRolloutRefWorker.update_actor()`  
    `verl/workers/fsdp_workers.py:1418-1464`

- 核心更新实现：
  - `DataParallelPPOActor.update_policy_diffusion()` ->
    - `flow_grpo`: `_update_policy_diffusion_flow_grpo()`
    - `dancegrpo`: `_update_policy_diffusion_dancegrpo()`  
  见 `verl/workers/actor/dp_actor.py:825-832`、`569-649`、`651-823`。

### 5.2 与 disco 的差异

- `disco`：`fsdp_worker.update_actor()` 内含 dance 主要算法实现。
- `long-rl`：fsdp worker 只做设备/并行封装，dance 核心迁到 `dp_actor.py`。

- `disco` 中很多“reward->adv->筛选->更新”在 worker 内部串起来；
- `long-rl` 中改成：
  1. rollout 返回轨迹；
  2. trainer 侧 reward 注入 + dual advantage（`compute_advantage_diffusion`）；
  3. actor 侧只消费 `vq_advantages/mq_advantages` 做 best-of-n + clipped PPO 损失更新。  

- 所以 `update_actor` 输入契约也变了：dance 路径要求 batch 里已有 `vq_advantages/mq_advantages`，否则直接报错（`ray_trainer.py:3057-3063`，`dp_actor.py:653-667`）。

---

## 6. 一句话结论

`long-rl` 有对应 Case4 调用链，但它不是 disco 的“fsdp_worker 一体化实现”；而是“worker 包装 + rollout模块 + trainer奖励/优势 + actor更新”分层实现。VideoAlign 仍可用（在 reward manager），Hunyuan 在当前主链路中缺失。
