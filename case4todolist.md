# Case4 硬迁移 ToDo（基于 `transfer_strategy11.md`）

## 0. 迁移总原则（硬约束，先读再动手）

> 本次是**硬迁移**，目标不是“在 long-rl 里重写一个相似实现”，而是“把 `disco_rl` 的 Case4 逻辑搬过来并跑通”。

- [ ] 原则 P0：`disco_rl` 源码是唯一真值（Source of Truth），迁移时优先复制，再做最小键名映射。
- [ ] 原则 P1：只允许两类改动：
  - [ ] 名字映射（配置键、角色名、函数调用名）。
  - [ ] 运行必需的最小适配（例如 `actor_rollout_ref` -> `actor_rollout` 角色判定兼容）。
- [ ] 原则 P2：禁止“自我发挥”：
  - [ ] 禁止把 `disco_rl` worker 内闭环拆回 long-rl 原有分层链路。
  - [ ] 禁止引入 `compute_reward` / `compute_advantage_diffusion` / critic 路径。
  - [ ] 禁止为了“代码风格统一”改动核心算法细节（采样、log_prob、adv 归一化、best-of-n、loss 形式）。
- [ ] 原则 P3：Case4 只覆盖单一路径，不保证其他路径兼容。

---

## 1. 目标与命中条件（先锁死）

- [ ] 固定 Case4 条件：
  - [ ] `trainer.disaggregate=false`
  - [ ] `trainer.pipelined_micro_batch=false`
  - [ ] `algorithm.adv_estimator=grpo`
  - [ ] `actor_rollout_ref.actor.dance_case4_mode=true`（新增硬开关）
  - [ ] 不再使用 `actor_rollout_ref.actor.disco` 旧键
- [ ] 在 trainer 和 worker 两侧都做命中判断，防止误入。
- [ ] 明确退出条件 A：`dance_case4_mode=false` 或未命中 Case4 时，必须回到原路径，不得污染其他 case。
- [ ] 明确退出条件 B：`dance_case4_mode=true` 但未命中 Case4（配置冲突）时，必须 fail-fast 报错并提示缺失条件。

---

## 2. 源码对照清单（迁移时逐项打勾）

### 2.1 Trainer 主循环来源

- [ ] 来源：`~/codegfile/disco_rl/verl/trainer/ray_trainer.py::fit`
- [ ] 目标：`~/codegfile/long-rl/verl/trainer/ppo/ray_trainer.py::fit_dance_case4`
- [ ] 要求：主循环结构、batch 构造方式、调用顺序保持一致。

### 2.2 Worker 初始化来源

- [ ] 来源：`~/codegfile/disco_rl/verl/workers/fsdp_workers.py::_build_model_optimizer_dance`
- [ ] 目标：`~/codegfile/long-rl/verl/workers/fsdp_workers.py::_build_model_optimizer_dance`
- [ ] 要求：保留模型/优化器/vae/videoalign 初始化逻辑，只改配置键映射。

### 2.3 Worker rollout 来源

- [ ] 来源：`~/codegfile/disco_rl/verl/workers/fsdp_workers.py::generate_sequences`
- [ ] 目标：`~/codegfile/long-rl/verl/workers/fsdp_workers.py::_generate_sequences_dance`
- [ ] 要求：`flux_step + sigma_schedule + VAE decode + VideoAlign reward` 逻辑不改。

### 2.4 Worker update 来源

- [ ] 来源：`~/codegfile/disco_rl/verl/workers/fsdp_workers.py::update_actor`
- [ ] 目标：`~/codegfile/long-rl/verl/workers/fsdp_workers.py::_update_actor_dance`
- [ ] 要求：group 归一化 advantage、best-of-n、timestep 子采样、dual loss、clip ratio 逻辑一致。

### 2.5 数据集来源

- [ ] 来源：`~/codegfile/disco_rl/fastvideo/dataset/latent_rl_datasets.py`
- [ ] 目标：`~/codegfile/long-rl/fastvideo/dataset/latent_rl_datasets.py`
- [ ] 要求：输入输出字段协议不变。

---

## 3. 阶段 A：补齐 fastvideo 依赖（必须第一步）

- [ ] A1. 全量覆盖目录：
  - [ ] `~/codegfile/disco_rl/fastvideo` -> `~/codegfile/long-rl/fastvideo`
- [ ] A2. 快速自检 imports（至少包含）：
  - [ ] `fastvideo.utils.load`
  - [ ] `fastvideo.utils.fsdp_util`
  - [ ] `fastvideo.models.videoalign.inference`
- [ ] A3. 失败即停：若此阶段 import 不通，不进入后续代码改造。

验收口径：
- [ ] 可以在 `long-rl` 环境中完成上述模块导入，无 `ModuleNotFoundError`。

---

## 4. 阶段 B：数据集接入（对齐 disco 输入协议）

- [ ] B1. 迁移 `latent_rl_datasets.py` 到 `long-rl/fastvideo/dataset/`。
- [ ] B2. 在 Case4 模式下，dataloader 强制使用：
  - [ ] `LatentDataset`
  - [ ] `latent_collate_function`
- [ ] B3. 输出字段严格为：
  - [ ] `encoder_hidden_states`
  - [ ] `encoder_attention_mask`
  - [ ] `caption`
- [ ] B4. 禁止复用 long-rl 现有 diffusion dataset 输出（`prompt_embeds/...`）。

验收口径：
- [ ] `next(iter(train_dataloader))` 实际返回三字段协议，shape 与 dtype 可进入 worker。

---

## 5. 阶段 C：Worker 硬分叉（`verl/workers/fsdp_workers.py`）

### 5.1 `init_model()` 分支

- [ ] C1. 增加 `dance_case4_mode` 判定（worker 侧）。
- [ ] C2. 命中时执行 `_build_model_optimizer_dance()` 并 `return`。
- [ ] C3. Case4 下明确不走：
  - [ ] `_build_model_optimizer()`
  - [ ] `_build_model_optimizer_diffusion()`
  - [ ] `_build_rollout()` / `_build_rollout_diffusion()`
  - [ ] `DataParallelPPOActor.update_policy_diffusion`

### 5.2 新增 `_build_model_optimizer_dance()`

- [ ] C4. 从 disco 原样复制函数主体。
- [ ] C5. 仅做键映射：
  - [ ] `lr <- self.config.actor.optim.lr`
  - [ ] `weight_decay <- self.config.actor.optim.weight_decay`
  - [ ] checkpoint 开关取 `self.config.model.enable_gradient_checkpointing` 或 `self.config.actor.extra.gradient_checkpointing`
  - [ ] 模型参数来自 `self.config.actor.extra.dance.*`
- [ ] C6. role 判定放宽：`actor_rollout` 也视为可加载 inferencer 的角色。

### 5.3 `generate_sequences()` 分支

- [ ] C7. 顶部加分支：命中 Case4 时 `return self._generate_sequences_dance(prompts)`。
- [ ] C8. 新增 `_generate_sequences_dance`，复制 disco 逻辑。
- [ ] C9. 返回字段严格对齐 disco 协议：
  - [ ] `timesteps`
  - [ ] `latents`
  - [ ] `next_latents`
  - [ ] `log_probs`
  - [ ] `vq_rewards`
  - [ ] `mq_rewards`
  - [ ] `encoder_hidden_states`
  - [ ] `encoder_attention_mask`
  - [ ] `meta_info["sigma_schedule"]`

### 5.4 `update_actor()` 分支

- [ ] C10. 顶部加分支：命中 Case4 时 `return self._update_actor_dance(data)`。
- [ ] C11. 新增 `_update_actor_dance`，复制 disco 的更新逻辑。
- [ ] C12. 在 worker 内完成 advantage 计算，不依赖 trainer 预写 `vq_advantages/mq_advantages`。

验收口径：
- [ ] worker 在 Case4 路径内完成 rollout/reward/adv/update 闭环。

---

## 6. 阶段 D：Trainer 硬分叉（`verl/trainer/ppo/ray_trainer.py`）

### 6.1 命中判定

- [ ] D1. 新增 `_is_dance_case4_mode(self)`。
- [ ] D2. 判定条件完整覆盖：
  - [ ] `self.diffusion is True`
  - [ ] `self.diffusion_disaggregate is False`
  - [ ] `not config.trainer.pipelined_micro_batch`
  - [ ] `config.algorithm.adv_estimator == GRPO`
  - [ ] `config.actor_rollout_ref.actor.dance_case4_mode is True`
  - [ ] 不依赖 `config.actor_rollout_ref.actor.disco`

### 6.2 训练循环切换

- [ ] D3. 在 `fit()` 顶部硬切：
  - [ ] `if self._is_dance_case4_mode(): return self.fit_dance_case4()`
- [ ] D4. 新增 `fit_dance_case4()`，按 disco `fit` 主循环复制。
- [ ] D5. 输入构造严格对齐：
  - [ ] `DataProto.from_single_dict({"encoder_hidden_states":..., "encoder_attention_mask":...}, meta_info={"caption": caption})`
- [ ] D6. 调用顺序固定：
  - [ ] `self.actor_rollout_wg.generate_sequences(new_batch)`
  - [ ] `self.actor_rollout_wg.update_actor(batch)`

### 6.3 禁止调用（Case4 下）

- [ ] D7. 不调用 reward_fn / rm worker。
- [ ] D8. 不计算 old_log_prob / ref_log_prob。
- [ ] D9. 不调用 `compute_advantage_diffusion`。
- [ ] D10. 不进入 critic 更新。

验收口径：
- [ ] trainer 层只负责取 batch + 分发到 worker 闭环，不做 diffusion 旧链路处理。

---

## 7. 阶段 E：配置文件落地（Case4 专用 YAML）

- [ ] E1. **必须**新建独立配置：`examples/diffusion/config_video_diffusion_case4_dance.yaml`。
- [ ] E1.1 不允许复用或“就地修改”现有通用 diffusion 配置，避免误入非 Case4 分支。
- [ ] E2. 必填字段检查：
  - [ ] `trainer.diffusion=true`
  - [ ] `trainer.disaggregate=false`
  - [ ] `trainer.pipelined_micro_batch=false`
  - [ ] `algorithm.adv_estimator=grpo`
  - [ ] `algorithm.use_kl_in_reward=false`
  - [ ] `critic.enable=false`
  - [ ] `reward_model.enable=false`
  - [ ] `actor_rollout_ref.actor.dance_case4_mode=true`
  - [ ] `actor_rollout_ref.actor.use_kl_loss=false`
  - [ ] 不再配置 `actor_rollout_ref.actor.disco`
  - [ ] `actor_rollout_ref.actor.optim.{lr,weight_decay}`
  - [ ] `actor_rollout_ref.actor.extra.dance.{pretrained_model_name_or_path,vae_model_path,model_type,master_weight_type,use_videoalign,videoalign_ckpt_path,timestep_fraction}`
  - [ ] `actor_rollout_ref.rollout.{sampling_steps,shift,eta,width,height,num_frames,fps,num_generations,use_group,use_same_noise,bestofn,vq_coef,mq_coef}`
  - [ ] `data.{data_json_path,t,cfg,dataloader_num_workers}`
- [ ] E3. 初始冒烟建议：
  - [ ] `max_train_steps=1`
  - [ ] `checkpointing_steps` 先设大（减少干扰）

验收口径：
- [ ] YAML 能直接启动，且命中 Case4 分支。

---

## 8. 阶段 F：运行验收（必须有日志证据）

### 8.1 冒烟（1 step）

- [ ] F1. 启动 Case4 专用配置，跑 `max_train_steps=1`。
- [ ] F2. 检查日志必须出现：
  - [ ] `init_model` 命中 `_build_model_optimizer_dance`
  - [ ] step 顺序：`generate_sequences -> update_actor`
  - [ ] rollout 返回关键字段齐全
  - [ ] `optimizer.step()` 至少执行 1 次且 loss 非 NaN

### 8.2 稳定性（10 steps）

- [ ] F3. 在固定配置下连续跑 `>=10` steps。
- [ ] F4. 检查：
  - [ ] 无 NaN / Inf
  - [ ] 无 shape mismatch
  - [ ] 无关键字段缺失

---

## 9. 差异审计（防止“迁移变重写”）

- [ ] G1. 对每个迁移函数做 diff 审计：
  - [ ] 与 `disco_rl` 对照，非必要差异必须收敛。
- [ ] G2. 若存在差异，必须记录“差异理由 + 影响评估”。
- [ ] G3. 重点审计项：
  - [ ] `sigma_schedule` 生成与使用
  - [ ] `flux_step` 及 `log_prob` 计算
  - [ ] advantage 归一化策略
  - [ ] dual loss 组合方式
  - [ ] best-of-n/timestep 子采样逻辑

---

## 10. 高风险点与回退策略

- [ ] R1. `fastvideo` 缺失或版本不一致：
  - [ ] 立刻回到“整目录覆盖”，不做碎片补丁。
- [ ] R2. dataloader 字段不对齐：
  - [ ] 禁止在 worker 里兜底猜字段，直接修 dataset/collate。
- [ ] R3. 误入 long-rl 原 diffusion 链路：
  - [ ] 在 `fit()` 与 `init_model()` 顶部加 assert/日志：`dance_case4_mode=true` 且未命中时立即报错；未开启开关时回原路径。
- [ ] R4. role 名不一致：
  - [ ] 在 dance 逻辑中显式兼容 `actor_rollout`。

---

## 11. 完成定义（DoD）

- [ ] DoD-1：Case4 配置稳定运行 `>=10` steps。
- [ ] DoD-2：调用链与策略一致：`fit -> generate_sequences -> update_actor`。
- [ ] DoD-3：worker 内闭环完整，不依赖 trainer 的 reward/adv 旧路径。
- [ ] DoD-4：关键算法逻辑与 `disco_rl` 对齐，未做语义重写。
- [ ] DoD-5：日志与代码审计均可证明“尊重源代码迁移”。
