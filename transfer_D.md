# 阶段D：Dance 更新主路径迁移 TODO（可直接执行）

## 0. 阶段定位

- 对应 `transfer_strategy.md` 的阶段D。
- 目标：实装 DanceGRPO 核心更新闭环（Best-of-N + step 子采样 + 双损失）；`flow_grpo` 保持不变。

## 1. 源码尊重硬约束（必须先满足）

1. 仅迁移 `disco_rl` 已有更新公式与执行顺序。
2. 不修改 `flow_grpo` 的现有 loss/调度语义。
3. 不新增 `disco_rl` 不存在的算法分支。

## 2. 输入契约（来自阶段B/C）

1. rollout 字段可用：`timesteps/latents/next_latents/log_probs`。
2. trainer 字段可用：`vq_advantages/mq_advantages`。
3. 配置字段可用：`bestofn/num_generations/vq_coef/mq_coef/timestep_fraction`。

## 3. 输出契约（阶段D完成后，交付阶段E/F）

1. `dancegrpo` 可端到端执行前向+反向，无 shape/NaN 错误。
2. 指标可见：`vq_loss/mq_loss/final_loss`、有效训练 step 比例、best-of-n 选择信息。
3. `flow_grpo` 回归通过，未被新分支污染。

## 4. 函数级 TODO（按顺序执行）

| 顺序 | long-rl 改动点 | 本阶段只做什么 | disco_rl 对齐依据 |
|---|---|---|---|
| D1 | `verl/trainer/ppo/ray_trainer.py` `fit`/`fit_dis` | `diffusion_algo=dancegrpo` 时将 `vq_advantages/mq_advantages` 原样送入 actor；`flow_grpo` 保持旧输入。 | `disco_rl/verl/trainer/ray_trainer.py:1372-1390,1434-1438` |
| D2 | `verl/workers/actor/dp_actor.py` `update_policy_diffusion` | 在现有函数中增加 dance 分支，消费 `log_probs+vq/mq_advantages`；过渡期可将 `old_log_probs` 只读映射到 `log_probs`。 | `disco_rl/verl/workers/fsdp_workers.py:1672-1729` |
| D3 | `verl/workers/actor/dp_actor.py` `_forward_micro_batch_diffusion` | 支持“按指定 step 索引”计算 log-prob，供 step permutation 逐步更新。 | `disco_rl/verl/workers/fsdp_workers.py:1692-1701` |
| D4 | `verl/workers/actor/dp_actor.py` `update_policy_diffusion` dance分支 | 迁移 Best-of-N：`score=vq_coef*vq_adv+mq_coef*mq_adv`，取 top/bottom 后打乱。 | `disco_rl/verl/workers/fsdp_workers.py:1650-1658` |
| D5 | 同 D4 | fail-fast：`bestofn` 偶数且 `<= num_generations`；`vq/mq_adv` 统一为 1D 再排序。 | `disco_rl/verl/workers/fsdp_workers.py:1650-1655` |
| D6 | 同 D4 | `num_generations==bestofn` 走直通分支并显式赋 `batch_size`；所有训练键同步索引。 | `disco_rl/verl/workers/fsdp_workers.py:1661-1669`; `disco_rl/verl/trainer/ray_trainer.py:1404-1414` |
| D7 | `verl/workers/actor/dp_actor.py` dance分支 | 迁移 step permutation，仅重排 `timesteps/latents/next_latents/log_probs`。 | `disco_rl/verl/workers/fsdp_workers.py:1666-1676` |
| D8 | 同 D7 | 迁移 step 子采样：`train_timesteps=int(total_steps*timestep_fraction)`。 | `disco_rl/verl/workers/fsdp_workers.py:1686` |
| D9 | 同 D7 | 迁移双路 loss：ratio + clip（`clip_range=1e-4`,`adv_clip_max=5.0`），`final_loss=vq_coef*vq_loss+mq_coef*mq_loss`。 | `disco_rl/verl/workers/fsdp_workers.py:1689-1732` |
| D10 | `verl/workers/actor/dp_actor.py` + `_optimizer_step` | 对齐梯度调度：mini-batch 边界 step，step 后 `zero_grad()`，避免梯度泄漏。 | `disco_rl/verl/workers/fsdp_workers.py:1608,1734-1738` |
| D11 | `verl/workers/fsdp_workers.py` `update_actor` | 保持 worker 侧 scheduler 步进位置不变，仅接 dance metrics。 | `long-rl/verl/workers/fsdp_workers.py:1444-1447` |
| D12 | `verl/workers/actor/dp_actor.py` dance metrics | 上报 `actor/dance/vq_loss`、`actor/dance/mq_loss`、`actor/dance/final_loss`、`actor/dance/train_step_ratio`、`actor/dance/bestofn_hit_rate`。 | `disco_rl/verl/workers/fsdp_workers.py:1715,1729,1731` |
| D13 | `tests/workers/actor/test_special_dp_actor.py`（或新增） | 单测：奇偶校验、直通分支、shape 一致、`timestep_fraction` 生效、loss 有限。 | `disco_rl/verl/workers/fsdp_workers.py:1650-1739` |
| D14 | `tests/trainer/...` 回归 | `flow_grpo` 无回归；`dancegrpo` 可跑通并返回 D12 指标。 | 阶段D验收要求 |

## 5. 阶段D自检清单

1. Dance 路径能完整执行到优化器 step。
2. `bestofn` 与 `timestep_fraction` 都能从配置真实生效。
3. `flow_grpo` 同配置下行为不变。

## 6. 交付给阶段E/F的固定上下文

1. 更新主路径已闭环；后续阶段只增强“奖励来源”和“异步调度”。
2. 阶段E 不得改 D 的 loss 公式；阶段F 不得改 D 的更新数学定义。

## 7. 非目标（阶段D不做）

1. 不迁移异步流水线函数（留给阶段F）。
2. 不迁移 VideoAlign 后端（留给阶段E）。
3. 不改 `flow_grpo` 公式、clip、调度。
