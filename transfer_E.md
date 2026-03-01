# 阶段E：VideoAlign 与外部奖励后端插件化接入迁移 TODO（可直接执行）

## 0. 阶段定位

- 对应 `transfer_strategy.md` 的阶段E。
- 目标：把奖励来源插件化接入，不污染默认主链路；保持阶段C/D的字段契约不变。

## 1. 源码尊重硬约束（必须先满足）

1. `fastvideo/models/videoalign/*` 只做原样迁移，不改方法名和签名。
2. 默认路径仍是 builtin reward；不开开关不应导入 VideoAlign 依赖。
3. 不改变阶段D已确定的 `vq/mq/final_loss` 更新逻辑。

## 2. 输入契约（来自阶段C/D）

1. trainer 已能消费 `vq_rewards/mq_rewards` 并计算 `vq_advantages/mq_advantages`。
2. dance 更新路径已闭环，依赖的字段名已固定。
3. reward 侧仍保留 `overall` 主通道。

## 3. 输出契约（阶段E完成后，交付阶段F）

1. reward backend 可切换：`builtin` / `videoalign`。
2. VideoAlign 启用时可回传 `VQ/MQ[/TA/Overall]` 并映射到训练字段。
3. VideoAlign 失败或不可用时回退 `VQ/MQ=-1/-1`，训练不中断。
4. 关闭 VideoAlign 时主路径零回归。

## 4. 函数级 TODO（按顺序执行）

| 顺序 | long-rl 改动点 | 本阶段只做什么 | disco_rl 对齐依据 |
|---|---|---|---|
| E1 | `fastvideo/models/videoalign/inference.py`（新增） | 原样迁移 `load_configs_from_json` 与 `VideoVLMRewardInference` 全部方法。 | `disco_rl/fastvideo/models/videoalign/inference.py:18-255` |
| E2 | `fastvideo/models/videoalign/prompt_template.py`（新增） | 原样迁移 `build_prompt`。 | `disco_rl/fastvideo/models/videoalign/prompt_template.py:100` |
| E3 | `fastvideo/models/videoalign/vision_process.py`（新增） | 原样迁移视频处理链路（含 `process_vision_info` 依赖函数）。 | `disco_rl/fastvideo/models/videoalign/vision_process.py:40-446` |
| E4 | `fastvideo/models/videoalign/data.py`（新增） | 原样迁移 `DataConfig` 与 `QWen2VLDataCollator`。 | `disco_rl/fastvideo/models/videoalign/data.py:16,131` |
| E5 | `fastvideo/models/videoalign/utils.py`（新增） | 原样迁移配置类与 `load_model_from_checkpoint`。 | `disco_rl/fastvideo/models/videoalign/utils.py:13-163` |
| E6 | `fastvideo/models/videoalign/train_reward.py`（新增） | 原样迁移 `create_model_and_processor`。 | `disco_rl/fastvideo/models/videoalign/train_reward.py:69` |
| E7 | `fastvideo/models/videoalign/trainer.py`（新增） | 原样迁移 `Qwen2VLRewardModelBT`。 | `disco_rl/fastvideo/models/videoalign/trainer.py:59` |
| E8 | `fastvideo/models/videoalign/__init__.py`（新增） | 与源仓一致保留模块初始化文件。 | `disco_rl/fastvideo/models/videoalign/__init__.py` |
| E9 | `verl/workers/reward_manager/batch.py` | 增加 backend 抽象选择：`builtin/videoalign`；保留 `use_videoalign` 兼容映射。 | `disco_rl/verl/workers/reward/config.py:33`; `fsdp_workers.py:664-669` |
| E10 | `BatchRewardManager.__init__` | 仅 `videoalign` 分支延迟导入并初始化 `VideoVLMRewardInference`。 | 同 E9 |
| E11 | `BatchRewardManager.__call__` diffusion分支 | `videoalign` 优先走 `reward_from_videos(...)`，有路径时兼容 `reward(...)`；读取 `VQ/MQ/TA/Overall`。 | `disco_rl/fastvideo/models/videoalign/inference.py:227-253` |
| E12 | `BatchRewardManager.__call__` 容错 | 失败回退 `VQ/MQ=-1.0`，并记录超时/异常/不可用计数。 | `disco_rl/verl/workers/fsdp_workers.py:2146-2149,2430-2448` |
| E13 | `BatchRewardManager.__call__` 返回结构 | 在 extra info 返回 `VQ/MQ/TA/Overall` + backend/异常计数字段。 | `disco_rl/fastvideo/models/videoalign/inference.py:248-253` |
| E14 | `verl/trainer/ppo/ray_trainer.py` `fit_dis` adv前 | 若 reward metrics 含 `VQ/MQ`，写入 `batch.batch["vq_rewards"/"mq_rewards"]`。 | `disco_rl/verl/workers/fsdp_workers.py:2171-2172` |
| E15 | `verl/trainer/ppo/ray_trainer.py` `fit` adv前 | 与 E14 保持同口径映射。 | `disco_rl/verl/workers/fsdp_workers.py:2470-2471` |
| E16 | `verl/trainer/config/reward_model/reward_model.yaml` | 增加 `reward_kwargs.use_videoalign: false`；可选 `reward_kwargs.backend: builtin`。 | `disco_rl/examples/config_video_diffusion.yaml:132-135` |
| E17 | `examples/diffusion/config_video_diffusion_npu.yaml` | 增加 VideoAlign 开关样例，补齐 `data.return_full_prompt: true`。 | 同 E16 |
| E18 | `examples/diffusion/config_video_diffusion.yaml` | 与 E17 保持一致，避免配置漂移。 | 同 E16 |
| E19 | `setup.py` / `requirements*.txt` | VideoAlign 依赖作为可选安装组；默认路径不强依赖。 | VideoAlign 依赖链来源 `train_reward.py/trainer.py` |
| E20 | `tests/workers/reward_manager/test_batch_reward_manager_videoalign_on_cpu.py` | 覆盖：关闭零回归、开启回传、内存视频路径、异常回退、计数正确。 | 阶段E验收 |
| E21 | `tests/trainer/ppo/*` | 覆盖 `fit/fit_dis` 的 `VQ/MQ -> vq_rewards/mq_rewards` 映射回归。 | `disco_rl/verl/workers/fsdp_workers.py:2171-2172,2470-2471` |

## 5. 阶段E自检清单

1. `use_videoalign=false` 时，不触发 VideoAlign 依赖导入。
2. `use_videoalign=true` 时，`VQ/MQ` 可稳定进入 trainer。
3. 异常时回退与统计可观测，训练不中断。
4. D 阶段更新公式与指标未被修改。

## 6. 交付给阶段F的固定上下文

1. 奖励来源可插拔，但输出字段契约不变：`vq_rewards/mq_rewards`。
2. 阶段F只做异步与性能，不改奖励字段语义。

## 7. 非目标（阶段E不做）

1. 不迁移 `fastvideo/train_grpo_*` 训练脚本。
2. 不改 `flow_grpo` 默认配置和语义。
3. 不重写 VideoAlign 内部算法实现。
