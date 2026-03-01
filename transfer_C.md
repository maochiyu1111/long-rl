# 阶段C：奖励协议与双路优势迁移 TODO（可直接执行）

## 0. 阶段定位

- 对应 `transfer_strategy.md` 的阶段C。
- 目标：建立与 `disco_rl` 一致的双奖励消费和双路优势计算；不进入策略更新。

## 1. 源码尊重硬约束（必须先满足）

1. 仅复用 `disco_rl` 已有双奖励与标准化语义。
2. 不新增自定义优势公式，不改现有单奖励路径。
3. 缺失值处理必须显式配置，禁止隐式兜底。

## 2. 输入契约（来自阶段B）

1. batch/rollout 可提供 `vq_rewards/mq_rewards`（或通过 reward extra info 映射得到）。
2. 轨迹与元信息透传可用，不丢字段。
3. `num_generations` 可从配置读到。

## 3. 输出契约（阶段C完成后，交付阶段D）

1. trainer 在 dual reward 场景稳定产出：`vq_advantages/mq_advantages`。
2. 保留单奖励 `advantages/returns` 逻辑，行为不变。
3. 双路统计指标可见：reward 均值/方差/标准化后分布。
4. dual reward 缺失值按配置执行：`error | skip | fill`。

## 4. 函数级 TODO（按顺序执行）

| 顺序 | long-rl 改动点 | 本阶段只做什么 | disco_rl 对齐依据 |
|---|---|---|---|
| C1 | `verl/workers/reward_manager/batch.py` `BatchRewardManager.__call__` | 扩展 diffusion reward 协议透传：`overall` 主通道不变，`vq_reward/mq_reward` 与 `VQ/MQ` 保留在 extra info。 | `disco_rl/verl/workers/reward/function.py:137`; `disco_rl/verl/workers/fsdp_workers.py:2141,2143` |
| C2 | `verl/workers/reward_manager/naive.py` `NaiveRewardManager.__call__` | 兼容 `score/overall`，保持文本路径旧行为。 | `disco_rl/verl/workers/reward/function.py:104` |
| C3 | `verl/trainer/ppo/reward.py` `compute_reward` | 不改 fallback 逻辑，确保 `reward_extra_info` 原样保留给 trainer。 | `disco_rl/verl/trainer/ray_trainer.py:1372` |
| C4 | `verl/trainer/ppo/ray_trainer.py` `fit`/`fit_dis` adv 前处理 | 注入 `vq_rewards/mq_rewards`：优先 batch 字段，其次 extra info（含 `VQ/MQ` 别名）。 | `disco_rl/verl/trainer/ray_trainer.py:1372-1390` |
| C5 | `verl/trainer/ppo/ray_trainer.py` `compute_advantage_diffusion` | 增加双路优势：`group` 按 `num_generations` 标准化；`batch` 仅在显式单组标记下按整 batch 标准化。 | `disco_rl/verl/workers/fsdp_workers.py:1625-1648`; `disco_rl/verl/trainer/ray_trainer.py:1372-1390` |
| C6 | `verl/trainer/ppo/ray_trainer.py`（校验） | `group` 模式下 batch 长度需可整除 `num_generations`；`batch` 模式必须显式单组标记；不满足直接报错。 | 同 C5 语义前提 |
| C7 | `verl/trainer/config/algorithm.py` + `ppo_trainer.yaml` | 新增缺失策略配置：`dual_reward_missing_strategy`、`dual_reward_fill_value`、`dual_adv_mode_default`。 | `transfer_strategy.md` 阶段C约束 |
| C8 | `verl/trainer/ppo/ray_trainer.py`（metrics 汇总） | 新增 `vq/mq` reward 与 advantage 统计指标。 | `disco_rl/verl/workers/fsdp_workers.py:1638,1648` |
| C9 | `tests/trainer/ppo/test_diffusion_dual_advantage_on_cpu.py` | 固定输入数值测试：组内标准化、batch 标准化、单路回归、约束报错。 | C5-C6 对齐语义 |
| C10 | `tests/trainer/config/test_algo_config_on_cpu.py` | dual reward 策略配置加载与分支行为测试。 | C7 |

## 5. 阶段C自检清单

1. 固定 reward 输入时，`vq_advantages/mq_advantages` 与手算一致。
2. 无 `vq/mq` 时，单奖励路径结果与改造前一致。
3. `group/batch` 模式切换依赖显式配置或标记，不允许隐式推断。

## 6. 交付给阶段D的固定上下文

1. 训练 batch 已稳定含 `vq_advantages/mq_advantages`。
2. 字段命名固定为 `vq_advantages/mq_advantages`，阶段D直接消费，不再改名。
3. dual reward 缺失值语义已固定，阶段D不再处理奖励缺失策略。

## 7. 非目标（阶段C不做）

1. 不做 `bestofn` 选择与样本重排。
2. 不做 `timestep_fraction` 子采样。
3. 不做 `vq/mq` 双损失反向更新。
4. 不接 VideoAlign 实现细节。
