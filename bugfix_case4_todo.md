# Case4 Bugfix TODO（仅修复数据算法通路）

## 0. 目标与边界

- 目标 case：`disaggregate=false, pipelined_micro_batch=false, disco=false`。
- 目标入口：`fit()`（`verl/trainer/ppo/ray_trainer.py`）。
- 目标结果：case4 在 `trainer.diffusion=true && trainer.diffusion_algo=dancegrpo` 下可完成完整数据算法通路：rollout group 生成 -> dual-adv 计算 -> actor update。
- 边界约束：
  - 不改 `fit_dis()`、`fit_disco_pipelined()` 逻辑分支。
  - 不改 `flow_grpo` 行为。
  - 不引入与 case4 无关的新算法步骤。

## 1. 源码基线对齐约束（`disco_rl@sijie/dancegrpo`）

- dance 语义必须由明确的算法上下文触发，不能隐式猜测。
- group 生成语义保持一致：
  - sync：`use_group=True` 时按 `num_generations` 扩展。
  - async：`use_group and gen_seed` 时扩展。
- dual-adv 的 group 标准化前提必须成立：`batch_size % num_generations == 0`。
- 对齐参考位置：
  - 基线 sync group repeat：`/Users/bytedance/codegfile/disco_rl/verl/workers/fsdp_workers.py:2002-2016`
  - 基线 async group repeat：`/Users/bytedance/codegfile/disco_rl/verl/workers/fsdp_workers.py:2308-2313`
  - 当前 dance 判定入口：`verl/workers/rollout/diffusion_rollout.py:37-61`

## 2. 当前 case4 硬断点

- `fit()` diffusion 分支未给 rollout 透传 `gen_batch.meta_info["diffusion_algo"]`（当前只写了 `global_steps`）。
- `diffusion_rollout` 的 dance 判定依赖 `meta_info["diffusion_algo"] == "dancegrpo"`。
- 结果是 rollout 未按 group 扩展，后续 dual-adv group 校验触发 `batch_size % num_generations != 0` 报错。

## 3. Bugfix TODO（按优先级执行）

### P0-1 修复 `fit()` 的 dance 元信息透传（case4 主修复）

- [ ] 文件：`verl/trainer/ppo/ray_trainer.py`（`fit()` 的 diffusion 分支）。
- [ ] 动作：
  - 在 `gen_batch` 构建后补齐 `gen_batch.meta_info["diffusion_algo"] = self.diffusion_algo`。
  - 当 `self.diffusion_algo == "dancegrpo"` 时，补齐 `gen_batch.meta_info["use_seed"] = ("seed" in gen_batch.batch.keys())`。
  - 保留已有 `gen_batch.meta_info["global_steps"]`。
- [ ] 约束：
  - 仅在 `self.diffusion` 分支生效。
  - 不改文本路径与 `flow_grpo` 计算逻辑。
- [ ] 完成判定：
  - case4 下 rollout 输出 batch 大小满足 group 预期（可被 `num_generations` 整除）。
  - dual-adv 不再因为分组不可整除报错。

### P0-2 增加 case4 专属前置断言（防止静默偏离）

- [ ] 文件：`verl/trainer/ppo/ray_trainer.py`（`fit()` 的 advantage 前）。
- [ ] 动作：
  - 在 `self.diffusion && self.diffusion_algo == "dancegrpo"` 时，显式校验 `len(batch) % num_generations == 0`。
  - 报错信息需包含：`len(batch)`、`num_generations`、`diffusion_algo`、`use_group`。
- [ ] 约束：
  - 只加 fail-fast，不改任何数值计算。
- [ ] 完成判定：
  - 当未来再次漏传 dance meta 时，错误在进入 dual-adv 前即被定位。

### P1-1 统一元信息注入入口（防回归，可与 P0-1 同提交）

- [ ] 文件：`verl/trainer/ppo/ray_trainer.py`。
- [ ] 动作：
  - 抽取私有 helper（例如 `_prepare_diffusion_gen_meta(...)`），由 `fit()` 复用。
  - helper 行为与现有 `_make_batch_data()` / `_make_batch_data_dis()` 的 diffusion 元信息约定一致。
- [ ] 约束：
  - 不改变已有 `_make_batch_data*` 的外部行为。
- [ ] 完成判定：
  - `fit()` 与 `_make_batch_data*` 对 dance 元信息约定一致，消除分叉实现。

### P1-2 增加回归测试（覆盖 case4 断点）

- [ ] 文件建议：
  - `tests/trainer/ppo/test_case4_fit_dance_meta_on_cpu.py`（新增）。
  - 或在现有 `tests/workers/rollout/test_diffusion_rollout_protocol.py`、`tests/trainer/ppo` 相关测试中补充。
- [ ] 最小测试点：
  - dancegrpo 场景下，`fit()` 使用的 diffusion gen meta 含 `diffusion_algo=dancegrpo`。
  - `use_seed` 在有/无 `seed` 字段时都符合预期。
  - `flow_grpo` 路径不受影响（不要求 dual-adv 字段）。
- [ ] 完成判定：
  - 新增测试在 CPU 单测环境通过。

## 4. 非目标（本 TODO 明确不做）

- 不修 Case1/Case2 的 `fit_disco_pipelined` 更新节奏问题。
- 不修 Case3 的 `fit_dis` 截断组结构问题。
- 不在本轮处理 `sampling_steps/shift/eta` 与 `num_steps` 的对齐差异。

## 5. 验收清单（必须全部满足）

- [ ] case4：至少 1 个训练 step 完成 `adv + update_actor`，无 group 整除错误。
- [ ] case4：`training/diffusion_algo` 指标为 `dancegrpo`，且 dual-adv 指标可见。
- [ ] 其他 case 入口不变：Case1/2 仍走 `fit_disco_pipelined()`，Case3 仍走 `fit_dis()`。
- [ ] `flow_grpo` 相关现有单测全部通过。

## 6. 建议提交拆分

- Commit 1：`fix(case4): propagate dance diffusion meta in fit path`

