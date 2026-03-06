# 全流程迁移审计（A~F 合并后）

## 审计范围与前提
- 目标：审计四组开关组合下流程是否可跑通，并对照 `disco_rl@sijie/dancegrpo` 的函数/变量/算法步骤。
- 假设：`trainer.diffusion=true`，且迁移目标为 `dancegrpo` 路径。
- 审计方式：静态代码审计（未跑端到端训练）。

## 四组 Case 结论

| Case | 配置 | 实际入口 | 结论 |
| --- | --- | --- | --- |
| Case1 | `disaggregate=true, pipelined_micro_batch=true, disco=false` | `fit_disco_pipelined()` | **进程可前进，但默认配置下高风险“只 rollout 不 update”（训练空转）** |
| Case2 | `disaggregate=true, pipelined_micro_batch=false, disco=true` | `fit_disco_pipelined()` | **与 Case1 同路径同风险，`disco` 未形成独立执行语义** |
| Case3 | `disaggregate=true, pipelined_micro_batch=false, disco=false` | `fit_dis()` | **存在明确断点：dual-adv group 校验很容易直接报错退出** |
| Case4 | `disaggregate=false, pipelined_micro_batch=false, disco=false` | `fit()` | **存在明确断点：rollout 未进入 dance 模式，后续 dual-adv 校验报错** |

---

## 关键问题（按严重度）

### P0-1 Case4 `fit()` diffusion 分支未透传 `diffusion_algo`，导致 rollout 不走 dance 语义
- 证据：
  - `fit()` 只写入 `gen_batch.meta_info["global_steps"]`，没有写入 `diffusion_algo`：`verl/trainer/ppo/ray_trainer.py:2782-2795`。
  - dance 判定依赖 `prompts.meta_info["diffusion_algo"] == "dancegrpo"`：`verl/workers/rollout/diffusion_rollout.py:37-38`。
  - 未进入 dance 后 `use_group` 不生效，输出 batch 不会按 `num_generations` 扩展。
  - dual-adv group 模式强制 `batch_size % num_generations == 0`：`verl/trainer/ppo/ray_trainer.py:431-438`。
- 结论：Case4 在当前代码下有硬中断风险，不能判定为“跑通”。

### P0-2 Case3 `fit_dis()` 对 diffusion rollout 结果按 `train_batch_size` 截断，破坏 group 完整性
- 证据：
  - `fit_dis` 采样后返回 `batch[: train_batch_size * rollout_repeat]`，diffusion 下 `rollout_repeat=1`：`verl/trainer/ppo/ray_trainer.py:1570-1587`。
  - dance group 优势计算要求 batch 按 `num_generations` 完整分组：`verl/trainer/ppo/ray_trainer.py:431-439`。
- 影响：当截断后不是完整组（或不可整除）时，直接在 dual-adv 处报错。
- 结论：Case3 有明确流程断点。

### P1-1 Case1/Case2 在默认配置下可能不触发任何 actor 更新（静默空转）
- 证据：
  - 两个 Case 均进入同一函数：`verl/trainer/main_ppo.py:349-353`。
  - `fit_disco_pipelined()` 仅在 `batch_size <= actor_wg.world_size` 时才把样本送 `update_actor_asyn`：`verl/trainer/ppo/ray_trainer.py:1385-1405`。
  - 当前示例配置里 actor/rollout_ref 切分为 `4/4`：`examples/diffusion/config_video_diffusion_npu.yaml:123-126`。
  - 默认 `bestofn=8`：`verl/trainer/config/rollout/rollout.yaml:95-97`。
- 风险链路：`bestofn(8) > actor_world_size(4)` 时，rollout 正常但不进入 update 分支。
- 结论：Case1/Case2 在默认参数组合下不是有效训练闭环。

### P1-2 与基线算法变量映射存在偏差：`sampling_steps/shift/eta` 配置未真正驱动 rollout
- 基线行为：`disco_rl` 在 dance rollout 中显式用 `sampling_steps + shift` 构造 `sigma_schedule`：`disco_rl/verl/workers/fsdp_workers.py:2326-2330`。
- 当前行为：`long-rl` rollout 调的是 `num_steps`，未使用 `sampling_steps/shift/eta`：`verl/workers/rollout/diffusion_rollout.py:200-201, 328-329`。
- 同时配置中仍暴露 `sampling_steps/shift/eta`：`verl/trainer/config/rollout/rollout.yaml:104-111`。
- 结论：迁移后“变量存在但不参与关键计算”，与基线算法步骤不一致。

### P2-1 `disco` 与 `pipelined_micro_batch` 当前仅作为“同一路径入口条件”
- 证据：
  - 路由上：满足任一开关都调用 `fit_disco_pipelined()`：`verl/trainer/main_ppo.py:349-353`。
  - `fit_disco_pipelined()` 内部没有 `self.disco` 分支行为。
- 结论：Case1/Case2 当前不是两条独立流程，而是同一路径的两个触发条件。

### P2-2 `fit_disco_pipelined` 默认 `do_gdr_sync=False`，与基线默认值不一致
- 当前：`verl/trainer/ppo/ray_trainer.py:1173-1179`。
- 基线：`disco_rl/verl/trainer/ray_trainer.py:1183-1189`（默认 `True`）。
- 结论：不是必崩点，但会增加 actor/rollout_ref 权重漂移风险。

---

## 对迁移原则的结论

1. 对齐基线函数/变量/算法步骤：**未完全对齐**。主要偏差在 `fit()` dance 标记透传、`fit_dis()` group 截断、以及 `sampling_steps/shift/eta` 未接入关键采样逻辑。
2. 流程可跑通性：**Case3/Case4 存在硬断点**；**Case1/Case2 存在静默空转风险**（默认参数下 actor 可能不更新）。

---

## 附注
- 本次未执行端到端训练，只做了静态代码路径审计。
