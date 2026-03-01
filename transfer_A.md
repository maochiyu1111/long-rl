# 阶段A：算法开关与配置骨架迁移 TODO（可直接执行）

## 0. 阶段定位

- 对应 `transfer_strategy.md` 的阶段A。
- 目标：只建立 `dancegrpo` 的配置入口和分支骨架，不改任何现有训练算法逻辑。

## 1. 源码尊重硬约束（必须先满足）

1. 只迁移 `disco_rl` 已存在的配置项、字段名、分支语义。
2. 不新增 `disco_rl` 不存在的算法函数、变量、训练步骤。
3. `flow_grpo` 默认路径必须保持现状，行为不变。
4. 能原样复制的配置与命名，直接原样复制。

## 2. 输入契约（阶段A开始前）

1. long-rl 当前 `flow_grpo` 训练路径可用。
2. 配置系统（Hydra + dataclass）可正常加载现有 trainer/actor/rollout 配置。

## 3. 输出契约（阶段A完成后，交付阶段B）

1. 存在统一算法开关：`trainer.diffusion_algo: flow_grpo | dancegrpo`（默认 `flow_grpo`）。
2. Dance 所需配置键可被完整加载（但此阶段不接入算法）：
   - rollout 侧：`use_group/use_same_noise/num_generations/bestofn/vq_coef/mq_coef/sampling_steps/shift/eta`
   - actor 侧：`timestep_fraction`
3. 配置校验可拦截非法值：
   - `bestofn <= num_generations`
   - `bestofn` 为偶数
   - `0 < timestep_fraction <= 1`
   - `vq_coef >= 0` 且 `mq_coef >= 0`
4. 日志中可见算法模式与关键超参，但 rollout/update 逻辑不变。

## 4. 函数级 TODO（按顺序执行）

| 顺序 | long-rl 改动点 | 本阶段只做什么 | disco_rl 对齐依据 |
|---|---|---|---|
| A1 | `verl/trainer/config/ppo_trainer.yaml` | 新增 `trainer.diffusion_algo`，默认 `flow_grpo`。 | `disco_rl/verl/workers/fsdp_workers.py:2231` |
| A2 | `verl/trainer/ppo/ray_trainer.py` `RayPPOTrainer.__init__` | 读取并缓存 `self.diffusion_algo`，仅做模式标记。 | `disco_rl/verl/trainer/ray_trainer.py:1392` |
| A3 | `verl/trainer/ppo/ray_trainer.py` `fit`/`fit_dis` 入口 | 放置 `flow_grpo/dancegrpo` 路由骨架；本阶段可先共用旧实现。 | `disco_rl/verl/workers/fsdp_workers.py:2231` |
| A4 | `verl/trainer/ppo/ray_trainer.py` `_validate_config` | 校验 `diffusion_algo` 枚举值。 | `disco_rl/verl/workers/fsdp_workers.py:1996,2231` |
| A5 | `verl/trainer/ppo/ray_trainer.py` `_validate_config` | 在 `dancegrpo` 下增加校验：`bestofn/num_generations/timestep_fraction/vq_coef/mq_coef`。 | `disco_rl/verl/workers/fsdp_workers.py:1650,1686` |
| A6 | `verl/workers/rollout/config.py` + `verl/trainer/config/rollout/rollout.yaml` | 增加 dance rollout 配置字段（只定义，不接入逻辑）。 | `disco_rl/verl/workers/rollout/config.py:63-73` |
| A7 | `verl/workers/config/actor.py` + `verl/trainer/config/actor/actor.yaml` | 增加 `timestep_fraction` 配置字段。 | `disco_rl/verl/workers/actor/config.py:124` |
| A8 | `verl/trainer/main_ppo.py` + `ray_trainer.py` | 预留/打印监控字段：`training/diffusion_algo`、`training/dance/*`、`training/dual_reward_enabled`。 | `disco_rl/verl/trainer/main.py:125` |
| A9 | `tests/trainer/config/test_algo_config_on_cpu.py` | 配置测试：正例加载 + 负例校验（越界、奇偶、负系数）。 | 同 A5 配置约束 |
| A10 | `tests/trainer/...` smoke | 回归测试：默认 `flow_grpo` 不受影响。 | 阶段A验收要求 |

## 5. 阶段A自检清单

1. 不开 `dancegrpo` 时，训练路径与原来完全一致。
2. 开 `dancegrpo` 仅新增“可配置、可校验、可见日志”，不改变 rollout/update 数学行为。
3. 所有新增字段命名与 `disco_rl` 保持一致。

## 6. 交付给阶段B的固定上下文

1. `trainer.diffusion_algo` 已可用。
2. rollout/actor 的 dance 参数已能从配置读到。
3. 配置校验已就位，阶段B可专注做 rollout 协议，不再补配置骨架。

## 7. 非目标（阶段A不做）

1. 不改 `verl/workers/rollout/diffusion_rollout.py` 的生成与轨迹逻辑。
2. 不改 `verl/workers/actor/dp_actor.py` 的更新算法逻辑。
3. 不做 `vq/mq` 奖励、优势、best-of-n、dance loss 的任何实装。
