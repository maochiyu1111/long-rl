# 阶段F：异步流水线对齐与性能优化迁移 TODO（可直接执行）

## 0. 阶段定位

- 对应 `transfer_strategy.md` 的阶段F。
- 目标：在不改算法数学定义前提下，迁移异步 dance 流水线并建立性能基线。

## 1. 源码尊重硬约束（必须先满足）

1. 只迁移 `disco_rl` 已有异步函数和调度语义。
2. 不修改阶段D已确定的 `best-of-n / timestep_fraction / dual-loss` 数学定义。
3. 异步能力只在对应开关下启用，默认行为不变。

## 2. 输入契约（来自阶段D/E）

1. 同步 dance 更新路径已稳定可跑。
2. reward backend 字段契约稳定：`vq_rewards/mq_rewards`。
3. 配置中已有 `disaggregate` 和 dance 关键参数。

## 3. 输出契约（阶段F完成后）

1. 同配置下可切换同步/异步并完成训练。
2. 异步路径保持 group 语义与同步一致。
3. 性能指标可对比：`time_per_step`、`tokens/s`、`samples/s`、奖励端时延、显存/内存峰值。

## 4. 函数级 TODO（按顺序执行）

| 顺序 | long-rl 改动点 | 本阶段只做什么 | disco_rl 对齐依据 |
|---|---|---|---|
| F1 | `verl/trainer/main_ppo.py` `TaskRunner.run` | 对齐 disaggregate 初始化与入口分流：`fit_dis` vs `fit_disco_pipelined`。 | `disco_rl/verl/trainer/main.py:76-79,108-113` |
| F2 | `verl/trainer/ppo/ray_trainer.py` `RayPPOTrainer.__init__` | 缓存 `self.disco`（读取 `actor.disco`），只作开关。 | `disco_rl/verl/trainer/ray_trainer.py:211` |
| F3 | `verl/trainer/ppo/ray_trainer.py` `_validate_config` | 校验 `actor.disco` 与 `trainer.pipelined_micro_batch` 互斥。 | `disco_rl/verl/trainer/main.py:76-79` |
| F4 | `verl/trainer/config/ppo_trainer.yaml` | 新增 `trainer.pipelined_micro_batch: False`。 | `disco_rl/verl/trainer/config.py:198-199` |
| F5 | `verl/workers/config/actor.py` + `actor.yaml` | 对齐 `disco/gradient_accumulation_steps/timestep_fraction`。 | `disco_rl/verl/workers/actor/config.py:104-105,123-124` |
| F6 | `verl/workers/rollout/config.py` + `rollout.yaml` | 核对异步 dance 依赖字段完整性（与前阶段同名）。 | `disco_rl/verl/workers/rollout/config.py:63-73` |
| F7 | `verl/trainer/ppo/ray_trainer.py` 新增 `fit_disco_pipelined` | 原样迁移主流程（含 `_make_batch_prompts`、`_get_datapro` 调用链）。 | `disco_rl/verl/trainer/ray_trainer.py:1183-1247` |
| F8 | `fit_disco_pipelined` 内 `_submit_one_rollout` | 对齐异步生成提交与 `meta_info` 透传（`role/round/global_step/use_seed`）。 | `disco_rl/verl/trainer/ray_trainer.py:1336-1343,1349-1358` |
| F9 | `fit_disco_pipelined` 内聚合段 | 对齐 `ray.wait` 聚合、双路标准化、bestofn 选择；禁止跨 group 混合后再标准化。 | `disco_rl/verl/trainer/ray_trainer.py:1367-1413` |
| F10 | `fit_disco_pipelined` actor更新节奏 | 对齐 `step_weight` 控制与 `update_actor_asyn` 调用频率。 | `disco_rl/verl/trainer/ray_trainer.py:1367,1432-1439` |
| F11 | `verl/workers/fsdp_workers.py` `generate_sequences_asyn_dance` | 原样迁移异步 dance rollout（`use_group and gen_seed`、seed/噪声逻辑、字段返回）。 | `disco_rl/verl/workers/fsdp_workers.py:2231,2301-2309,2351-2374` |
| F12 | `verl/workers/fsdp_workers.py` `update_actor_asyn` | 原样迁移异步 actor 更新，保留 `step_weight` 语义与 D 阶段数学定义。 | `disco_rl/verl/workers/fsdp_workers.py:1744,1833,1871-1872,1924-1929` |
| F13 | `verl/workers/fsdp_workers.py` `compute_ref_log_probs_asyn` | 补齐异步 ref logprob 入口，与同步函数并存。 | `disco_rl/verl/workers/fsdp_workers.py:2564-2607` |
| F14 | `verl/trainer/ppo/metric_utils.py` + `ray_trainer.py` | 接入性能基线输出：吞吐、时延、内存峰值。 | `disco_rl/verl/trainer/ray_trainer.py:1447-1465`; `fsdp_workers.py:2281-2297` |
| F15 | `examples/diffusion/*.yaml` + 脚本 | 增加同步/异步可对比运行配置与入口，便于验收。 | `transfer_strategy.md` 阶段F验收 |

## 5. 阶段F自检清单

1. 同超参下同步/异步都可跑通。
2. 异步路径没有破坏 group 语义和奖励字段语义。
3. 性能对比数据可直接查看并复现。

## 6. 迁移边界（必须保持）

1. 不改 D 阶段数学定义与执行顺序。
2. 不改 `flow_grpo` 行为。
3. 不新增 `disco_rl` 中不存在的算法函数。
