# Case4 审计报告（dancegrpo.yaml）

## 审计范围
- 目标配置：`disaggregate=false`、`pipelined_micro_batch=false`、`disco=false`。
- 本次配置文件：`examples/diffusion/dancegrpo.yaml`。
- 基线参考：`/Users/bytedance/codegfile/disco_rl`。
- 审计方式：静态代码审计（未执行端到端训练；当前终端环境缺少训练依赖，如 `omegaconf`）。

## 路径核对（Case4）
1. 入口：`verl/trainer/main_ppo.py` 在该开关组合下走 `trainer.init_workers()` + `trainer.fit()`（非 `fit_dis` / 非 `fit_disco_pipelined`）。
2. `fit()` diffusion 分支会注入 `diffusion_algo` 与 `use_seed`（`verl/trainer/ppo/ray_trainer.py:2797`, `660-666`），rollout 能识别 dance 模式。
3. dance 训练主链完整：rollout -> reward -> dual reward 注入 -> dual advantage -> actor update（`verl/trainer/ppo/ray_trainer.py:2875-3069`, `3094-3117`）。

结论：主链路结构已接通，但存在以下高风险断点/偏差。

## Findings（按严重度）

### P0：VideoAlign 任一不可用/异常会导致训练直接中断
- 当前迁移逻辑：
  - `BatchRewardManager` 在 VideoAlign 不可用或异常时，返回占位分数 `VQ/MQ=-1`（`verl/workers/reward_manager/batch.py:192`, `228-230`, `244-246`）。
  - trainer 在 dance 双奖励注入时，检测到 `-1` 会直接 `raise ValueError`（`verl/trainer/ppo/ray_trainer.py:761-767`）。
- 与基线差异：
  - `disco_rl` 在同类异常时也写入 `-1`，但流程继续（`/Users/bytedance/codegfile/disco_rl/verl/workers/fsdp_workers.py:2146-2149`）。
- 影响：
  - 只要 VideoAlign 初始化失败、单样本推理异常或超时，Case4 全流程会硬中断。

### P1：算法步骤与基线不一致，`sampling_steps/shift/eta` 未驱动当前 rollout 采样
- 当前迁移逻辑：
  - Wan rollout 直接走 `wan_pipeline_with_logprob(... num_inference_steps=self.config.num_steps ...)`（`verl/workers/rollout/diffusion_rollout.py:324-329`）。
- 基线逻辑：
  - 显式构造 `sigma_schedule`，使用 `sampling_steps + shift`，并在步进中使用 `eta`（`/Users/bytedance/codegfile/disco_rl/verl/workers/fsdp_workers.py:2020-2023`, `2107`）。
- 影响：
  - 迁移后 Case4 的关键变量/算法步骤与源实现不等价，`sampling_steps/shift/eta` 对实际采样行为不再起主导作用。

### P1：`dancegrpo` 双奖励来源契约与基线不兼容（只接受 `reward_extra_info`）
- 当前迁移逻辑：
  - 强制 `vq/mq` 必须来自 `reward_extra_info`，明确禁用 batch 字段回退（`verl/trainer/ppo/ray_trainer.py:735-742`）。
- 基线逻辑：
  - `disco_rl` 直接在 rollout 输出 batch 中携带 `vq_rewards/mq_rewards`（`/Users/bytedance/codegfile/disco_rl/verl/workers/fsdp_workers.py:2171-2172`）。
- 影响：
  - 若沿用基线式上游输出契约（batch 内双奖励），在迁移代码中会被直接判定为错误来源并中断。

## P1 深度成因分析（补充，面向后续修复）

### 1) `sampling_steps/shift/eta` 未驱动 rollout 采样：根因不是“漏传参数”，而是“采样器实现已切换”
- 现状证据（longrl）：
  - 配置层保留了 `sampling_steps/shift/eta` 字段（`verl/workers/rollout/config.py:63-65`；`verl/trainer/config/rollout/rollout.yaml:104-111`）。
  - 训练器仅做日志上报，未参与采样实参构造（`verl/trainer/ppo/ray_trainer.py:637-657`）。
  - Wan rollout 实际只传 `num_inference_steps=self.config.num_steps`（`verl/workers/rollout/diffusion_rollout.py:324-329`）。
  - `wan_pipeline_with_logprob` 内部也仅按 `num_inference_steps` 驱动 scheduler（`verl/workers/diffusion_helper.py:698`, `833-834`）。
- 基线证据（disco_rl）：
  - 显式用 `sampling_steps` 构造 `sigma_schedule`，并应用 `shift`（`/Users/bytedance/codegfile/disco_rl/verl/workers/fsdp_workers.py:2020-2023`）。
  - 每一步 `flux_step` 显式消费 `eta`（`/Users/bytedance/codegfile/disco_rl/verl/workers/fsdp_workers.py:2107`）。
- 结构性结论：
  - 这是“采样内核从 Hunyuan 自定义 sigma/eta 逻辑迁移到 Wan/SD3 diffusers 统一逻辑”的语义分叉，不是单点调用遗漏。
  - 当前 longrl diffusion rollout 仅支持 Wan/SD3 路径，不存在 Hunyuan rollout 分支（`verl/workers/fsdp_workers.py:1238-1243`）。
  - 因此你提出的“是否因为 wan2.1 和 hunyuan 不一样”判断成立；更准确是“模型+采样器双重切换导致参数语义未做一一映射”。

### 2) `dancegrpo` 只接受 `reward_extra_info`：根因是“防占位符污染”的契约收口
- 现状证据（longrl）：
  - diffusion rollout 默认写入占位双奖励 `vq_rewards/mq_rewards=-1`（`verl/workers/rollout/diffusion_rollout.py:77-80`, `345-359`）。
  - trainer 在 dance 路径强制从 `reward_extra_info` 取双奖励，明确禁用 batch/non_tensor 回退（`verl/trainer/ppo/ray_trainer.py:721-742`）。
  - trainer 还对 `-1` 占位值做 fail-fast（`verl/trainer/ppo/ray_trainer.py:761-767`），避免静默退化。
  - reward manager 在 diffusion 分支把 VideoAlign 的 `VQ/MQ` 放到 `reward_extra_info` 返回（`verl/workers/reward_manager/batch.py:361-387`），并由 `compute_reward` 统一透传（`verl/trainer/ppo/reward.py:172-182`）。
  - 回归测试已把“仅有 batch 占位符时必须报错”固化（`tests/trainer/ppo/test_diffusion_reward_metric_mapping_on_cpu.py:89-96`）。
- 基线证据（disco_rl）：
  - rollout 直接产出真实 `vq_rewards/mq_rewards` 到 batch（`/Users/bytedance/codegfile/disco_rl/verl/workers/fsdp_workers.py:2171-2172`）。
  - trainer 直接在 batch 上做 vq/mq advantage（`/Users/bytedance/codegfile/disco_rl/verl/trainer/ray_trainer.py:1372-1390`）。
- 结构性结论：
  - 这是“奖励职责从 rollout worker 下沉/集中到 reward manager + trainer 注入”的架构变化，不是偶发实现偏差。
  - 你提出的“longrl 中 batch 在 fit 聚合，所以读取 reward_extra_info”判断基本成立；补充一点：该设计同时是为了阻断 rollout 占位符 `-1` 误入优化环。

## 后续修复建议（给 code agent）
- 针对 P1-采样参数：
  - 方案 A（等价对齐）：在 `wan_pipeline_with_logprob` 增加可选 `sigmas/shift/eta` 映射，把 `sampling_steps/shift/eta` 显式注入采样流程；并补齐协议测试。
  - 方案 B（语义收敛）：若决定只保留 `num_steps`，应在配置与日志中废弃或标注 `sampling_steps/shift/eta` 为“当前 Wan 路径不生效”，避免误配。
- 针对 P1-双奖励契约：
  - 方案 A（保持当前安全策略）：继续强制 `reward_extra_info`，同时在文档/配置中明确“dancegrpo 不接受 rollout batch 内 vq/mq 奖励”。
  - 方案 B（兼容基线输入）：允许 batch 回退但必须有强校验（禁止 `-1` 占位值、长度一致性、来源标记），以兼容历史上游。

## 总结
- 在 `disaggregate=false, pipelined_micro_batch=false, disco=false` 下，Case4 的主流程路径已连通。
- 但就“可稳定跑通”和“对齐 `disco_rl` 函数/变量/算法步骤”而言，仍存在 1 个硬中断风险（P0）+ 2 个基线偏差（P1）。
- 建议优先处理 P0（VideoAlign 占位分数与 trainer 严格校验的冲突），否则线上训练稳定性无法保证。

---

## 增补与勘误（2026-03-06，仅针对两个 P1）

> 本轮按你的要求忽略 P0（VideoAlign 不可用导致中断）讨论，聚焦两个 P1 的深层成因。

### 勘误：关于“只接受 `reward_extra_info`”
- 需要更正：当前 longrl 代码并非“只接受 `reward_extra_info`”。
- 当前实际优先级是：`reward_extra_info` -> `batch` -> `non_tensor_batch`。
  - 证据：`verl/trainer/ppo/ray_trainer.py:700-766`（`_resolve_dual_reward_tensor`）。
- 因此该 P1 更准确描述应为：
  - **双奖励来源契约与 disco_rl 不同，且存在多来源仲裁带来的语义歧义/退化风险**，而不是“硬性只接受 reward_extra_info”。

### P1-1 深层成因：`sampling_steps/shift/eta` 未驱动 rollout 采样

1. 这不是单点漏传，而是“采样内核语义迁移”
- longrl 配置层保留了 `sampling_steps/shift/eta`：
  - `verl/workers/rollout/config.py:63-65`
  - `verl/trainer/config/rollout/rollout.yaml:104-111`
- 但 rollout 实参仍由 `num_steps` 驱动：
  - SD3 路径：`verl/workers/rollout/diffusion_rollout.py:194-201`
  - WAN 路径：`verl/workers/rollout/diffusion_rollout.py:324-329`
- SD3 采样步进中未消费 `shift`，`eta` 也未配置化：
  - `sd3_pipeline_with_logprob` 签名无 `shift/eta`：`verl/workers/diffusion_helper.py:145-182`
  - `sde_step_with_logprob` 内 `std_dev_t` 仍固定乘 `0.7`：`verl/workers/diffusion_helper.py:109`

2. 模型路由维度与算法路由维度是正交的
- rollout 构建是按模型路径分支，不按 dance/flow 算法分支：
  - `verl/workers/fsdp_workers.py:1231-1243`
- 目前仅支持：`wan` -> `WanRollout`；`stable-diffusion` -> `StableDiffusionRollout`。
- 这说明你判断“wan2.1 与 hunyuan 不同导致采样逻辑不一致”是成立的，而且问题本质是：
  - **hunyuan 的 sigma/shift/eta 语义没有在 longrl 的模型路由 + 采样器抽象里落位**。

3. 与 disco_rl 的关键分叉
- disco_rl 是显式 `sampling_steps -> sigma_schedule`，然后 `shift` 变换，再在 step 里消费 `eta`：
  - `/Users/bytedance/codegfile/disco_rl/verl/workers/fsdp_workers.py:2020-2023, 2107`
- longrl 目前是 diffusers 风格的 `num_inference_steps` 主导。

4. 你提出的方案可行性（“把 hunyuan 逻辑放到 sd3 分支，避免影响 wan”）
- 方向可行，但要加两个边界：
  - **不要把 wan rollout 切到 sd3**；wan 继续走 `WanRollout` 现有路径。
  - 在 `StableDiffusionRollout`（或新增 `HunyuanRollout`）里引入 hunyuan-compatible 采样逻辑，并用显式开关/模型族判断开启。
- 推荐落地方式：
  - 新增采样策略枚举（例如 `rollout.sampling_policy = wan_native|hunyuan_compatible`），默认 `wan_native`。
  - `hunyuan_compatible` 仅在 SD3/Hunyuan 路径生效，消费 `sampling_steps/shift/eta`。
  - `WanRollout` 保持 `num_steps` 语义不变，避免对现有 wan2.1 训练造成行为漂移。

### P1-2 深层成因：dancegrpo 双奖励来源契约与基线不一致

1. 这是“职责拆分”导致的架构差异，不是简单字段名不一致
- longrl 中：
  - rollout 先写占位双奖励（`-1`）保证 schema：`verl/workers/rollout/diffusion_rollout.py:77-80, 345-359`
  - reward manager 在 diffusion 路径返回 `reward_tensor + reward_extra_info`：
    - `verl/workers/reward_manager/batch.py:324-389`
  - trainer 在 fit 阶段汇总并注入双奖励：
    - `verl/trainer/ppo/reward.py:162-182`
    - `verl/trainer/ppo/ray_trainer.py:2987-3033`
- disco_rl 中：rollout 直接产出 `vq_rewards/mq_rewards` 到 batch：
  - `/Users/bytedance/codegfile/disco_rl/verl/workers/fsdp_workers.py:2171-2172`
  - trainer 直接用 batch 做 advantage：`/Users/bytedance/codegfile/disco_rl/verl/trainer/ray_trainer.py:1372-1390`

2. 与你判断一致：根因确实包含“fit 阶段批处理聚合”
- 你的判断“longrl 中 batch 在 fit 聚合，所以读取 reward_extra_info”本质是对的。
- 更完整地说：
  - longrl 把奖励计算职责从 rollout 侧抽到 reward manager/fit 侧，形成了“rollout 占位 + fit 注入”的两阶段契约。

3. 当前真正风险点
- 由于当前允许 fallback 到 batch/non_tensor（见 `ray_trainer.py:700-766`），若 `reward_extra_info` 缺失或异常，理论上可能回退到 rollout 占位值来源，导致奖励语义退化。
- 即：问题不是“只能 reward_extra_info”，而是“多来源仲裁 + 占位值共存”带来的不确定性。

### 给后续修复的最小影响原则（供 code agent 使用）

1. 采样参数 P1（优先）
- 仅在 SD3/Hunyuan 侧新增 hunyuan-compatible 采样策略，消费 `sampling_steps/shift/eta`。
- WAN 分支保持现状，默认行为不变。
- 在 rollout 构建处扩展模型族识别（支持 hunyuan），不要用 `diffusion_algo` 直接决定模型 rollout。

2. 双奖励契约 P1
- 明确单一真源（推荐 `reward_extra_info`）或至少做严格来源标记。
- 若保留 fallback：必须显式拒绝占位值、加长度一致性校验、加来源埋点，防止 silent fallback。

3. 测试补齐
- 采样：新增“同配置下 wan 不变、sd3/hunyuan 改变”的 A/B 测试。
- 奖励：新增“reward_extra_info 缺失时是否触发预期策略（报错或受控降级）”测试。

---

## 追加 TODO（仅针对 Case4：`disaggregate=false`、`pipelined_micro_batch=false`、`disco=false`）

> 目标：在当前主路径（`init_workers()` + `fit()`）下跑通 Hunyuan，不依赖 `ray_trainer_orin.py`、不依赖 `disco=true` 分支。  
> 原则：优先对齐 disco_rl 的 Hunyuan 关键语义（`model_type` 加载、`sampling_steps/shift/eta` 采样逻辑），同时保持现有 `wan` 路径不变。

### T0. 固定范围与入口
- [ ] 在文档中明确本次只覆盖 Case4 主路径：`verl/trainer/main_ppo.py -> trainer.init_workers() -> trainer.fit()`。
- [ ] 明确不把“是否跑通”依赖到 `fit_disco_pipelined` 或 `init_model_disco`。

### T1. 配置与模型描述对齐（Case4 可直接使用）
- [ ] 在 `examples/diffusion/dancegrpo.yaml` 增加并使用 Hunyuan 必需字段：`actor.model.model_type`、`actor.model.pretrained_model_name_or_path`、`actor.model.vae_model_path`（命名按 longrl 配置体系落位）。
- [ ] 提供 Case4 专用示例（可命名为 `dancegrpo_hunyuan_case4.yaml`），固定为：`disaggregate=false`、`pipelined_micro_batch=false`、`disco=false`。
- [ ] 在配置注释中写清：`model_path` 可继续用于基础路由标识；Hunyuan 真实加载以 `model_type` + `pretrained_model_name_or_path` 为准。

### T2. 非 disco 主路径的模型加载能力补齐（按 model_type）
- [ ] 在 `verl/workers/fsdp_workers.py::_build_model_optimizer_diffusion` 增加 `model_type` 分支能力，支持 `hunyuan_hf/hunyuan`，避免“非 wan 即 SD3”。
- [ ] 将 disco_rl 中 `fastvideo/utils/load.py` 的必要加载能力迁入 longrl（当前 longrl 的 `fastvideo` 仅 `videoalign`，缺 Hunyuan 加载依赖）。
- [ ] 保持基础 rollout 类不新增 `HunyuanRollout`，继续使用现有类体系。

### T3. 在当前 rollout 主链路内注入 Hunyuan 采样语义
- [ ] 在 `verl/workers/rollout/diffusion_rollout.py` 的现有生成路径中，增加“当 `model_type` 为 Hunyuan 时”的分支，按 disco_rl 语义使用：`sampling_steps -> sigma_schedule -> shift -> eta`。
- [ ] 输出协议保持与现有 trainer 兼容：至少保证 `timesteps/latents/next_latents/log_probs(old_log_probs)/vq_rewards/mq_rewards` 可被后续阶段消费。
- [ ] 严格保持 Wan 现有 `num_steps` 行为不变，防止回归。

### T4. 训练闭环对齐（Case4 fit 路径）
- [ ] 核对 `verl/trainer/ppo/ray_trainer.py` 在 `fit()` diffusion 分支的输入/输出契约，确认 Hunyuan 分支不要求 `disco` 专用字段才能跑通。
- [ ] 核对 `verl/workers/actor/dp_actor.py` 的 diffusion logprob/update 路径，确保能消费 Hunyuan 分支产生的 `timesteps`/轨迹字段。
- [ ] 若字段契约存在差异，优先在 `fit()` 路径做最小兼容适配，不引入 `fit_disco_pipelined` 依赖。

### T5. 奖励与双奖励契约
- [ ] 维持当前 longrl 的奖励注入机制（`reward_extra_info` 主源），保证 Hunyuan 跑通时不会回退到占位 `-1`。
- [ ] 对 `vq/mq` 增加来源与有效性校验日志，便于定位 Case4 训练中断点。

### T6. 最小回归测试（围绕 Case4）
- [ ] 新增路由测试：Case4 配置下 `model_type=hunyuan_hf` 可走通构建与一次 rollout 生成。
- [ ] 新增采样测试：验证 Hunyuan 分支真实消费 `sampling_steps/shift/eta`（配置变更会改变轨迹长度/时间步）。
- [ ] 新增对照测试：同样配置下 Wan 路径行为不变。
- [ ] 新增一条最小训练冒烟：`disaggregate=false`、`pipelined_micro_batch=false`、`disco=false` 下完成至少 1 step（rollout->reward->update）。
