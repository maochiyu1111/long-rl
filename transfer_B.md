# 阶段B：Rollout 协议对齐（轨迹字段 + group 生成）迁移 TODO（可直接执行）

## 0. 阶段定位

- 对应 `transfer_strategy.md` 的阶段B。
- 目标：让 rollout 返回 Dance 更新必需字段；只做协议对齐，不进入优势和损失计算。

## 1. 源码尊重硬约束（必须先满足）

1. 只搬运 `disco_rl` 既有 group/seed/轨迹输出语义。
2. 不新增自定义采样分支，不改采样算法流程。
3. 同步与异步分支判定必须按源代码语义区分。

## 2. 输入契约（来自阶段A）

1. 已有 `trainer.diffusion_algo`。
2. 已有 rollout 配置键：`use_group/use_same_noise/num_generations/sampling_steps/shift/eta`。
3. 本阶段可直接读取这些键，不再新增命名。

## 3. 输出契约（阶段B完成后，交付阶段C/D）

1. rollout 样本稳定包含：`timesteps/latents/next_latents/log_probs`。
2. rollout 样本预留：`vq_rewards/mq_rewards`（占位值可按 disco 失败回退语义 `-1/-1`）。
3. group/seed 行为对齐：
   - 同步：按 `use_group` 控制 repeat。
   - 异步：仅 `use_group and gen_seed` 时 repeat。
4. step 维语义一致，避免训练侧再出现二次截断错位。

## 4. 函数级 TODO（按顺序执行）

| 顺序 | long-rl 改动点 | 本阶段只做什么 | disco_rl 对齐依据 |
|---|---|---|---|
| B1 | `verl/workers/rollout/config.py` + `verl/trainer/config/rollout/rollout.yaml` | 若阶段A遗漏，补齐并固定字段名（与 disco 完全同名）。 | `disco_rl/verl/workers/rollout/config.py:63-73` |
| B2 | `verl/workers/rollout/diffusion_rollout.py` `StableDiffusionRollout.generate_sequences` | 迁移同步 group repeat 逻辑。 | `disco_rl/verl/workers/fsdp_workers.py:2002` |
| B3 | `verl/workers/rollout/diffusion_rollout.py` `WanRollout.generate_sequences` | WAN 路径同 B2 对齐。 | `disco_rl/verl/workers/fsdp_workers.py:2002` |
| B4 | `verl/workers/rollout/diffusion_rollout.py`（同步/异步分支） | 对齐异步判定：异步只在 `use_group and gen_seed` repeat；同步保持 `use_group` 语义。 | `disco_rl/verl/workers/fsdp_workers.py:2002,2231,2308` |
| B5 | `verl/workers/rollout/diffusion_rollout.py`（噪声与seed） | 对齐 `use_same_noise` 与 seed 协议，只影响采样输入，不影响训练公式。 | `disco_rl/verl/workers/fsdp_workers.py:2045,2055,2351,2361,2365` |
| B6 | `verl/workers/rollout/diffusion_rollout.py`（返回结构） | 输出补齐 `timesteps/latents/next_latents/log_probs`；兼容保留 `old_log_probs`。 | `disco_rl/verl/workers/fsdp_workers.py:2163,2170,2462,2469` |
| B7 | `verl/workers/rollout/diffusion_rollout.py`（返回结构） | 预留 `vq_rewards/mq_rewards` 协议槽位（占位，不接后端）。 | `disco_rl/verl/workers/fsdp_workers.py:2147-2149,2171-2172,2446-2448,2470-2471` |
| B8 | `verl/workers/rollout/diffusion_rollout.py`（step维处理） | 统一 step 切片与 shape 断言，防止 `sampling_steps` 和真实轨迹错位。 | `disco_rl/verl/workers/fsdp_workers.py:2163,2170,2462,2469` |
| B9 | `verl/trainer/ppo/ray_trainer.py` `_make_batch_data/_make_batch_data_dis` | 注入 `meta_info["use_seed"]` 与可选 `batch["seed"]`，统一 sync/dis 路径协议。 | `disco_rl/verl/trainer/ray_trainer.py:1217,1230,1235` |
| B10 | `verl/workers/fsdp_workers.py` `generate_sequences` | 确保 preprocess/postprocess 不丢失新增轨迹字段与 meta_info。 | `disco_rl/verl/workers/fsdp_workers.py:1948` |
| B11 | `tests/workers/rollout/test_diffusion_rollout_protocol.py` | 新增协议测试：字段、shape、group倍率、sync/async repeat 条件、seed 复现。 | 阶段B验收 + `disco_rl` 对齐点 |

## 5. 阶段B自检清单

1. 训练侧可直接拿到 step 级轨迹，不需再拼接临时字段。
2. `use_group/use_same_noise/use_seed` 在 sync/dis 两条路径行为一致且可复现。
3. `vq_rewards/mq_rewards` 已存在，后续阶段可直接消费。

## 6. 交付给阶段C/D的固定上下文

1. 字段契约：`timesteps/latents/next_latents/log_probs/vq_rewards/mq_rewards`。
2. 语义契约：group 与 seed 判定已经固定，不允许后续阶段改语义。
3. C 阶段只做奖励消费与优势计算；D 阶段只做更新公式，不再回改 rollout 协议。

## 7. 非目标（阶段B不做）

1. 不做 `vq_advantages/mq_advantages` 计算。
2. 不做 `bestofn` 选择。
3. 不做 `timestep_fraction` 训练子采样与 dance loss。
4. 不接 VideoAlign 或任何外部奖励后端。
