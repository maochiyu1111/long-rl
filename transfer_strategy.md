# long-rl 迁移 DanceGRPO 细化策略

## 1. 迁移总原则

1. 底座不变：以 `long-rl(verl0.5 + diffusion + NPU)` 为主线。
2. 算法先行：优先迁移 DanceGRPO 的训练语义闭环，不先绑定 VideoAlign/fastvideo 全栈。
3. 分层落地：先打通数据协议与更新路径，再接奖励后端与性能优化。

## 2. DanceGRPO 算法闭环映射（作为迁移对照）

1. Group rollout：同 prompt 多生成（`use_group/num_generations/use_same_noise`）。
2. 双奖励：每样本 `vq_reward`、`mq_reward`。
3. 双路组内优势：分别标准化得到 `vq_adv`、`mq_adv`。
4. Best-of-N 采样：按 `vq_coef * vq_adv + mq_coef * mq_adv` 选择 top/bottom。
5. 时间步子采样：step permutation + `timestep_fraction` 子集训练。
6. Dance update：逐 step 做 ratio/clip，双损失加权更新。

后续各阶段均围绕这 6 个环节展开。

## 3. 分阶段实施方案（细化版）

### 阶段A：算法开关与配置骨架

目标：在不影响现有 `flow_grpo` 的前提下，提供 DanceGRPO 独立配置入口。

要做的事：
1. 增加算法选择开关：`trainer.diffusion_algo: flow_grpo | dancegrpo`（默认 `flow_grpo`）。
2. 新增 dance 配置组：`use_group/num_generations/use_same_noise/bestofn/vq_coef/mq_coef/timestep_fraction/sampling_steps/shift/eta`。
3. 配置校验规则：
   1. `bestofn <= num_generations`。
   2. `0 < timestep_fraction <= 1`。
   3. `vq_coef/mq_coef` 非负。
4. 日志与监控字段预留：打印算法模式、关键超参、是否启用双奖励。

对应 DanceGRPO 算法部分：
1. 对应闭环 `1/4/5/6` 的参数入口与调度开关。

阶段产出：
1. 可通过配置切换算法路径，但暂不改变 rollout/update 逻辑。

验收标准：
1. `flow_grpo` 路径行为完全不变。
2. `dancegrpo` 模式启动不报配置错误，关键参数可见于日志。

---

### 阶段B：Rollout 协议对齐（轨迹字段 + group 生成）

目标：让 long-rl rollout 具备 Dance 更新所需的最小充分轨迹信息。

要做的事：
1. rollout 输出补齐 step 级字段：
   1. `timesteps`
   2. `latents`
   3. `next_latents`
   4. `log_probs`
2. 引入 group 生成策略：
   1. 按 `num_generations` 扩展 prompt。
   2. 支持 `use_same_noise` 与 seed 控制，确保可复现实验。
   3. 对齐异步分支条件：`use_group` 仅在 `gen_seed=True` 时触发 repeat（与 `use_seed` 配套）。
3. 明确 step 数语义：
   1. 统一训练端消费的 step 数与采样端输出长度。
   2. 避免 `sampling_steps` 与真实轨迹长度错位（dance 源码当前存在 `[:, :-1]` 二次截断，实际训练 step 少 1）。
4. rollout 数据结构中预留奖励槽位：`vq_rewards/mq_rewards`（先可空）。

对应 DanceGRPO 算法部分：
1. 对应闭环 `1`（group rollout）。
2. 为闭环 `5/6`（step 子采样与 dance update）提供输入。

阶段产出：
1. dance 模式下可返回完整 step 轨迹张量，形状与 batch 对齐。

验收标准：
1. 单测覆盖轨迹字段存在性与 shape 一致性。
2. 同 seed 复跑时，`use_same_noise` 路径结果可重复。

---

### 阶段C：奖励协议与双路优势

目标：在 trainer 侧建立与 DanceGRPO 一致的双奖励、双优势计算逻辑。

要做的事：
1. reward 接口扩展：
   1. 统一支持 `vq_reward`、`mq_reward`、`overall_reward`（兼容旧逻辑）。
   2. 缺失值回退策略（例如默认值或跳过样本）要显式配置。
2. 按 prompt group 计算双路优势：
   1. 对 `vq_reward` 分组标准化得 `vq_adv`。
   2. 对 `mq_reward` 分组标准化得 `mq_adv`。
   3. 异步路径需显式保证“每个 rollout batch 对应单 prompt 组”，否则不能用整 batch 均值方差替代分组标准化。
3. 记录统计指标：
   1. `vq/mq` 均值、方差、标准化后分布。
   2. group 维度样本数与异常组比例。

对应 DanceGRPO 算法部分：
1. 对应闭环 `2/3`（双奖励 + 双路组内优势）。

阶段产出：
1. dance 模式训练 batch 中稳定产出 `vq_adv/mq_adv`。

验收标准：
1. 构造固定 reward 输入时，优势计算与预期数值一致。
2. 不影响原单奖励路径（`overall_reward` 仍可独立训练）。

---

### 阶段D：Dance 更新主路径（Best-of-N + step 子采样 + 双损失）

目标：实现 DanceGRPO 的核心策略更新逻辑，形成可训练闭环。

要做的事：
1. Best-of-N 选择器：
   1. `score = vq_coef * vq_adv + mq_coef * mq_adv`。
   2. 选 top/bottom 后拼接并打乱。
   3. `num_generations == bestofn` 时走直通逻辑，并显式覆盖 `batch_size`（避免同步路径出现未定义变量）。
   4. `bestofn` 必须为偶数（源码按 `bestofn//2` 取 top/bottom）。
2. 时间步采样：
   1. 每样本 step permutation。
   2. `train_steps = int(total_steps * timestep_fraction)`。
3. Dance loss 分支：
   1. 逐 step 计算 `ratio = exp(new_logp - old_logp)`。
   2. 分别计算 `vq_loss/mq_loss`（clip 规则与旧 GRPO 保持一致风格）。
   3. `final_loss = vq_coef * vq_loss + mq_coef * mq_loss`。
4. 更新调度：
   1. 梯度累计、梯度裁剪、optimizer/scheduler step 与现有训练框架对齐。
   2. 保留 `flow_grpo` 分支不变，dance 分支独立。

对应 DanceGRPO 算法部分：
1. 对应闭环 `4/5/6`（Best-of-N、timestep_fraction、dance policy update）。

阶段产出：
1. 在不接 VideoAlign 的情况下，dance 算法可端到端训练。

验收标准：
1. 前向/反向无 shape 或 NaN 错误。
2. 指标可见：best-of-n 命中率、有效训练 step 比例、`vq/mq/final_loss` 曲线。
3. 与 baseline 对比，至少完成稳定跑通若干 step/epoch。

---

### 阶段E：VideoAlign 与外部奖励后端插件化接入

目标：把高耦合奖励链路隔离为可插拔后端，避免污染 NPU 主干。

要做的事：
1. 定义 reward backend 抽象接口（输入视频/元信息，输出 `vq/mq[/ta/overall]`）。
2. 将 VideoAlign 作为一个后端实现，按配置启用，不进入默认主路径。
3. 增加 fallback 与容错：
   1. 后端不可用时的回退值策略。
   2. 超时/异常样本的跳过与统计。
4. 保持 NPU 兼容策略：
   1. 避免把 `fastvideo` 的 CUDA 训练栈全量引入核心 loop。
   2. 仅在后端边界做必要适配。

对应 DanceGRPO 算法部分：
1. 对应闭环 `2`（双奖励来源），并为后续 TA/Overall 扩展留口。

阶段产出：
1. reward backend 可切换：`builtin` / `videoalign`。

验收标准：
1. 不启用 VideoAlign 时主链路零回归。
2. 启用 VideoAlign 时 `vq/mq` 字段完整回传且训练可继续。

---

### 阶段F：异步流水线对齐与性能优化

目标：在语义对齐完成后，再迁移 dance 异步调度与吞吐优化。

要做的事：
1. 将同步闭环拆分为异步 rollout + trainer 聚合 + actor update。
2. 对齐批处理边界，确保 group 语义在异步场景不被破坏。
3. 引入 `step_weight`/更新频率控制，平衡吞吐与稳定性。
4. 建立性能基线：tokens/s、samples/s、显存/内存峰值、奖励端延迟。

对应 DanceGRPO 算法部分：
1. 对应闭环的工程调度变体，不改变 `1~6` 的算法语义。

阶段产出：
1. 异步模式可选启用，并与同步模式指标可对齐对比。

验收标准：
1. 异步与同步在同配置下收敛趋势一致（允许速度差异）。
2. 吞吐提升且无明显稳定性回退。

## 4. 推荐执行顺序与里程碑

1. M1（可配置但未生效）：完成阶段A。
2. M2（数据协议就绪）：完成阶段B+C，能产出 `vq/mq adv`。
3. M3（算法闭环跑通）：完成阶段D，dance 模式端到端训练。
4. M4（奖励后端增强）：完成阶段E，VideoAlign 可插拔。
5. M5（吞吐优化）：按需推进阶段F。

## 5. 风险控制与回滚策略

1. 风险：`flow_grpo` 被新逻辑污染。
   1. 控制：严格按 `diffusion_algo` 分支隔离 + 回归测试。
2. 风险：轨迹字段 shape 不一致导致更新崩溃。
   1. 控制：阶段B 建立 shape/step 语义单测。
3. 风险：双奖励质量波动影响稳定性。
   1. 控制：阶段C 增加 reward 统计与异常回退。
4. 风险：VideoAlign/fastvideo 耦合拖累 NPU 主路径。
   1. 控制：阶段E 插件化，不进入默认链路。

## 6. 最小可交付范围（建议）

若要尽快拿到“可训练、可验证”的首版，建议先完成：
1. 阶段A + 阶段B + 阶段C + 阶段D。
2. 阶段E/F 作为后续增强分批上线。

这对应“先算法语义对齐，再奖励和性能增强”的最低风险路径。

## 7. 源码对齐确认（disco_rl）

以下约束来自 `disco_rl` 真实代码，建议作为迁移时的硬约束：

1. Group rollout 与轨迹字段：
   1. `generate_sequences`/`generate_sequences_asyn_dance` 产出 `timesteps/latents/next_latents/log_probs/vq_rewards/mq_rewards`。
   2. 同步路径按 `use_group` repeat；异步路径是 `use_group and gen_seed` 才 repeat。
2. 双路优势与 best-of-n：
   1. 同步 `update_actor` 是按 `num_generations` 分组标准化。
   2. 异步 `fit_disco_pipelined` 是对 rollout batch 做标准化，依赖 batch 构造语义。
3. step 子采样与 dance loss：
   1. `train_timesteps = int(len(timesteps)*timestep_fraction)`。
   2. loss 为 `vq/mq` 双路 ratio+clip 后按 `vq_coef/mq_coef` 加权。
4. VideoAlign 接入：
   1. 仅在 `use_videoalign=true` 时调用 `VideoVLMRewardInference.reward(...)`，并读取 `VQ/MQ`。
   2. 失败时奖励回退 `-1/-1`。
