# transfer_A_result

## 1) 改动文件清单（含每个文件改动目的）

1. `verl/trainer/config/ppo_trainer.yaml`
   - 新增 `trainer.diffusion_algo: flow_grpo | dancegrpo`（默认 `flow_grpo`），提供阶段A算法开关入口（A1）。
2. `verl/trainer/ppo/ray_trainer.py`
   - `RayPPOTrainer.__init__` 读取并缓存 `self.diffusion_algo`（A2）。
   - `fit`/`fit_dis` 增加 `flow_grpo/dancegrpo` 路由骨架（阶段A先复用原实现，不改训练数学逻辑）（A3）。
   - `_validate_config` 增加 `diffusion_algo` 枚举校验（A4）与 dancegrpo 约束校验（A5）。
   - 增加 `training/diffusion_algo`、`training/dance/*`、`training/dual_reward_enabled` 的日志/指标预留（A8）。
3. `verl/workers/rollout/config.py`
   - 增加 dance rollout 配置字段定义：`use_group/use_same_noise/num_generations/bestofn/vq_coef/mq_coef/sampling_steps/shift/eta`（A6）。
4. `verl/trainer/config/rollout/rollout.yaml`
   - 暴露并赋默认值给上述 dance rollout 配置键（A6）。
5. `verl/workers/config/actor.py`
   - 增加 `timestep_fraction` 字段（默认 `0.6`）与可变字段注册（A7）。
6. `verl/trainer/config/actor/actor.yaml`
   - 暴露 `timestep_fraction` 配置键（A7）。
7. `verl/trainer/main_ppo.py`
   - 启动时打印 `training/diffusion_algo`、`training/dance/*`、`training/dual_reward_enabled`（A8）。
8. `tests/trainer/config/test_algo_config_on_cpu.py`
   - 新增阶段A配置测试：正例加载与负例校验（越界、奇偶、负系数、枚举值），以及 flow 默认路径不触发 dance 约束（A9/A10）。

## 2) 逐项对照 A1~A10 完成状态（PASS/FAIL）

- A1: PASS
- A2: PASS
- A3: PASS
- A4: PASS
- A5: PASS
- A6: PASS
- A7: PASS
- A8: PASS
- A9: FAIL（测试代码已补齐，但当前环境缺少 `torch/pytest`，未能执行验收）
- A10: FAIL（smoke 依赖 `pytest/torch`，当前环境不可执行，无法完成回归验收）

## 3) 与 disco_rl 的对齐证据（文件+函数/行号）

1. Dance rollout 字段命名与默认值对齐
   - 源：`disco_rl/verl/workers/rollout/config.py:63-72`
   - 迁移：
     - `long-rl/verl/workers/rollout/config.py:57-65`
     - `long-rl/verl/trainer/config/rollout/rollout.yaml:86-111`
   - 对齐点：`use_same_noise/use_group/num_generations/eta/vq_coef/mq_coef/bestofn/shift/sampling_steps`。

2. `timestep_fraction` 字段对齐
   - 源：`disco_rl/verl/workers/actor/config.py:124`
   - 迁移：
     - `long-rl/verl/workers/config/actor.py:121`
     - `long-rl/verl/trainer/config/actor/actor.yaml:132-133`

3. Best-of-N 与双系数、time-step fraction 语义约束来源
   - 源：`disco_rl/verl/workers/fsdp_workers.py:1650-1655,1661,1686`
   - 迁移校验：`long-rl/verl/trainer/ppo/ray_trainer.py:1071-1087`
   - 对齐点：`bestofn<=num_generations`、`bestofn` 偶数、`vq_coef/mq_coef`、`timestep_fraction`。

4. dance 分支入口语义（阶段A仅开关/骨架，不改训练算法）
   - 源：`disco_rl/verl/workers/fsdp_workers.py:2231`（`generate_sequences_asyn_dance` 入口）
   - 迁移骨架：
     - `long-rl/verl/trainer/ppo/ray_trainer.py:662-665`（`fit_dis` 路由）
     - `long-rl/verl/trainer/ppo/ray_trainer.py:2034-2037`（`fit` 路由）

5. group 语义命名对齐（仅配置层）
   - 源：`disco_rl/verl/workers/fsdp_workers.py:2308-2313`（`use_group` + `num_generations`）
   - 迁移：`long-rl/verl/workers/rollout/config.py:57-60`、`long-rl/verl/trainer/config/rollout/rollout.yaml:86-96`

## 4) 测试命令与结果摘要

执行命令与结果：

1. `pytest -q tests/trainer/config/test_algo_config_on_cpu.py`
   - 结果：FAIL
   - 原因：`pytest: command not found`

2. `pytest -q tests/trainer/ppo/test_core_algos_on_cpu.py`
   - 结果：FAIL
   - 原因：`pytest: command not found`

3. `python -m pytest ...`
   - 结果：FAIL
   - 原因：`python: command not found`

4. `python3 -m pytest -q tests/trainer/config/test_algo_config_on_cpu.py`
   - 结果：FAIL
   - 原因：`No module named pytest`

5. `python3 -m pytest -q tests/trainer/ppo/test_core_algos_on_cpu.py`
   - 结果：FAIL
   - 原因：`No module named pytest`

6. `python3 -m unittest tests/trainer/config/test_algo_config_on_cpu.py`
   - 结果：FAIL
   - 原因：`No module named torch`

7. `python3 -m unittest tests/trainer/ppo/test_core_algos_on_cpu.py`
   - 结果：FAIL
   - 原因：`No module named pytest`

8. `python3 -m py_compile <本次修改的py文件>`
   - 结果：PASS（语法检查通过）

## 5) commit hash

`e9d2320996be76c5d0c18e3865b63e03aa59b5e3`
