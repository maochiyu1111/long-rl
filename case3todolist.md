# Case3 迁移指导文件

本文不再把已删除的 `case4todolist.md` 当作规范来源，而是直接基于当前仓库的真实状态、`case3_disco_rl.md` 的基线语义，以及 `case4_long_rl.md` 已经验证成功的迁移思路来整理 Case3 的实施路线。

目标不是“写一个差不多的 Case3”，而是把 `disco_rl` 的 Case3 语义迁进 `long-rl`，同时保持和当前成功 Case4 一致的工程方法：

- 专用硬分叉
- 最小污染主路径
- fail-fast 明确报错
- 验收依赖真实执行证据

## 1. 当前仓库现状与差距

先把现状看清，后面的任务才不会跑偏。

### 1.1 当前已经有的能力

- 已有 `dance_case4_mode`
- 已有 Case4 专用入口逻辑
- 已有 Case4 latent dataloader 分支
- 已有 worker 侧 `_build_model_optimizer_dance / _generate_sequences_dance / _update_actor_dance`
- 已有 disaggregate 拓扑：
  - `init_workers_dis()`
  - `actor_wg`
  - `rollout_ref_wg`
  - `setup_dist()`
- 已有 disaggregate 的 GDR relay 调度框架

### 1.2 当前距离 Case3 还差什么

- 只有 `dance_case4_mode`，没有 `dance_case3_mode`
- `fit_dis()` 仍走当前通用 diffusion batch 组织逻辑
- 当前 `_make_batch_data_dis()` 依赖：
  - `prompt_embeds`
  - `negative_prompt_embeds`
- Case3 需要的却是 latent 三元组协议：
  - `encoder_hidden_states`
  - `encoder_attention_mask`
  - `caption`
- worker 侧专用硬分叉只有 Case4，没有 Case3 的 role-aware 分支
- 当前 GDR relay 默认面向 `rollout.pipeline.transformer`
- 但 Case3 dance disaggregate 目标态需要同步的是：
  - actor 侧 `self.transformer`
  - rollout_ref 侧 `self.rollout`

### 1.3 这意味着什么

Case3 不能只靠“复用现有 `fit_dis()` 再补一点 if 分支”完成。  
它需要像 Case4 一样，形成一条独立可识别的专用主链，只是拓扑从单组 worker 换成双组 worker。

## 2. 目标接口与配置变化

以下是 Case3 迁移完成后，仓库里应该新增或改变的外部接口。

### 2.1 新增配置开关

- [ ] 新增 `actor_rollout_ref.actor.dance_case3_mode`
- [ ] Case3 不再依赖 `actor_rollout_ref.actor.disco`
- [ ] 开关开启但条件不满足时，trainer 和 worker 两侧都必须 fail-fast

### 2.2 新增专用配置文件

- [ ] 新增 `examples/diffusion/config_video_diffusion_case3_dance.yaml`
- [ ] 该 YAML 只服务 Case3，不复用通用 diffusion YAML

### 2.3 新增专用启动脚本

- [ ] 新增 `case3_dance.sh`
- [ ] 负责组装 Case3 的 override、GPU split 和路径参数

### 2.4 新增专用 trainer 主循环

- [ ] 新增 `fit_dance_case3_dis()`
- [ ] `fit_dis()` 命中 Case3 后必须直接切进去

### 2.5 新增 Case3 dataloader 分支

- [ ] 新增 Case3 latent dataloader 构造逻辑
- [ ] 返回值固定为 `(encoder_hidden_states, encoder_attention_mask, caption)`

### 2.6 新增 worker role-aware 分支

- [ ] `init_model()` 在 Case3 下按 `role="actor"` / `role="rollout_ref"` 分流
- [ ] `generate_sequences()` 在 rollout_ref 侧走 Case3 dance 分支
- [ ] `update_actor()` 在 actor 侧走 Case3 dance 分支

## 3. 迁移任务清单

这一节按子系统拆任务，做完一块就能判断一块，不需要等到最后才发现方向错了。

### 3.1 入口与配置

- [ ] 在 `main_ppo.py` 新增 Case3 模式识别函数
- [ ] 命中条件至少覆盖：
  - [ ] `trainer.diffusion=true`
  - [ ] `trainer.disaggregate=true`
  - [ ] `trainer.pipelined_micro_batch=false`
  - [ ] `algorithm.adv_estimator=grpo`
  - [ ] `actor_rollout_ref.actor.dance_case3_mode=true`
- [ ] Case3 模式下跳过通用 tokenizer / processor 初始化
- [ ] Case3 模式下跳过通用 reward manager 初始化
- [ ] Case3 模式下跳过标准 RL dataset / sampler / collate_fn 初始化
- [ ] 非 Case3 模式保持现有路径不变

### 3.2 dataloader

- [ ] 参照 Case4 的思路新增 Case3 latent dataloader 构造函数
- [ ] 使用 `fastvideo.dataset.latent_rl_datasets.LatentDataset`
- [ ] 使用 `latent_collate_function`
- [ ] 输入配置至少包含：
  - [ ] `data.data_json_path`
  - [ ] `data.t`
  - [ ] `data.cfg`
  - [ ] `data.dataloader_num_workers`
- [ ] 输出必须严格是：
  - [ ] `encoder_hidden_states`
  - [ ] `encoder_attention_mask`
  - [ ] `caption`
- [ ] 不允许回退到 `prompt_embeds / negative_prompt_embeds`

### 3.3 trainer `fit_dis`

- [ ] 在 `fit_dis()` 顶部增加 Case3 模式分支
- [ ] 命中后直接进入 `fit_dance_case3_dis()`
- [ ] `fit_dance_case3_dis()` 每步顺序固定为：
  - [ ] 取 latent dataloader batch
  - [ ] 组装 `DataProto.from_single_dict(..., meta_info={"caption": caption})`
  - [ ] 先做 actor -> rollout_ref 同步
  - [ ] `rollout_ref_wg.generate_sequences(new_batch)`
  - [ ] `actor_wg.update_actor(rollout_batch)`
- [ ] Case3 模式下不进入：
  - [ ] `_make_batch_data_dis()`
  - [ ] `compute_reward(...)`
  - [ ] `compute_advantage_diffusion(...)`
  - [ ] ref log prob
  - [ ] critic update

### 3.4 worker `init_model`

- [ ] 在 `init_model()` 新增 Case3 命中判断
- [ ] 命中后直接进入 Case3 专用初始化并 `return`
- [ ] actor 侧职责：
  - [ ] 加载可训练 transformer
  - [ ] FSDP 包装
  - [ ] 创建 optimizer
  - [ ] 创建 lr scheduler
- [ ] rollout_ref 侧职责：
  - [ ] 加载 rollout 副本
  - [ ] 加载 VAE
  - [ ] 按需加载 VideoAlign inferencer
- [ ] 两边职责必须明确分离，不能再共享 Case4 的 colocated 语义

### 3.5 worker `generate_sequences`

- [ ] 在 `generate_sequences()` 顶部新增 Case3 分支
- [ ] 仅 rollout_ref 角色允许进入 Case3 rollout 逻辑
- [ ] 输入协议固定消费：
  - [ ] `encoder_hidden_states`
  - [ ] `encoder_attention_mask`
  - [ ] `meta_info["caption"]`
- [ ] 核心逻辑需对齐 `disco_rl`：
  - [ ] `sigma_schedule`
  - [ ] `flux_step`
  - [ ] rollout log prob
  - [ ] VAE decode
  - [ ] VideoAlign reward
- [ ] 返回字段保持与 disco 基线一致

### 3.6 worker `update_actor`

- [ ] 在 `update_actor()` 顶部新增 Case3 分支
- [ ] 仅 actor 角色允许进入 Case3 update 逻辑
- [ ] 核心逻辑需对齐 `disco_rl`：
  - [ ] group advantage 标准化
  - [ ] best-of-n
  - [ ] timestep shuffle / 子采样
  - [ ] PPO clip loss
  - [ ] `optimizer.step()`
- [ ] advantage 计算继续放在 worker 内，不回到 trainer

### 3.7 actor -> rollout 同步

- [ ] 复用当前 disaggregate 通信拓扑，不另造第四套同步框架
- [ ] 但同步 helper 需从“面向 pipeline.transformer”改成“面向 Case3 dance 模块”
- [ ] 明确同步源和同步目标：
  - [ ] actor 侧源权重：`self.transformer`
  - [ ] rollout_ref 侧目标权重：`self.rollout`
- [ ] rollout 前每步都必须完成同步
- [ ] 同步失败时要给出明确报错，不允许静默跳过

### 3.8 启动脚本与 YAML

- [ ] 新建 `case3_dance.sh`
- [ ] 新建 `config_video_diffusion_case3_dance.yaml`
- [ ] YAML 至少覆盖：
  - [ ] `trainer.diffusion=true`
  - [ ] `trainer.disaggregate=true`
  - [ ] `trainer.pipelined_micro_batch=false`
  - [ ] `algorithm.adv_estimator=grpo`
  - [ ] `algorithm.use_kl_in_reward=false`
  - [ ] `critic.enable=false`
  - [ ] `reward_model.enable=false`
  - [ ] `actor_rollout_ref.actor.dance_case3_mode=true`
  - [ ] `actor_rollout_ref.actor.use_kl_loss=false`
  - [ ] `actor_rollout_ref.actor.extra.dance.*`
  - [ ] `actor_rollout_ref.rollout.*`
  - [ ] `data.data_json_path`
  - [ ] `trainer.disaggregate_actor_n_gpus_per_node`
  - [ ] `trainer.disaggregate_rollout_ref_n_gpus_per_node`
- [ ] 明确处理 `dist_master_addr / dist_master_port`
- [ ] 不沿用当前默认硬编码地址作为最终可交付方案

### 3.9 测试与日志

- [ ] 新增 Case3 模式单测
- [ ] 复用 Case4 的测试风格，但不复用 Case4 的判断条件
- [ ] 增加 Case3 冒烟和稳定性日志留证要求
- [ ] 验证 Case4 现有测试不回归

## 4. 各子系统的明确完成标准

这一节不是任务重复，而是告诉我们“做到什么程度才算这块真的完成”。

### 4.1 入口层完成标准

- [ ] Case3 模式下，`main_ppo` 不再触发通用 tokenizer / processor 初始化
- [ ] Case3 模式下，`main_ppo` 不再构造通用 reward manager
- [ ] Case3 模式下，传给 trainer 的 dataset / collate_fn / sampler 为 Case3 专用途径

### 4.2 dataloader 完成标准

- [ ] `next(iter(train_dataloader))` 实际返回三元组
- [ ] 三元组能直接转成：

```python
DataProto.from_single_dict(
    {
        "encoder_hidden_states": ...,
        "encoder_attention_mask": ...,
    },
    meta_info={"caption": caption},
)
```

- [ ] 不需要任何通用 diffusion prompt 字段兜底

### 4.3 trainer 完成标准

- [ ] Case3 主循环只做三件事：
  - [ ] `sync`
  - [ ] `rollout`
  - [ ] `update`
- [ ] 看不到通用 diffusion reward / advantage / old_log_prob / ref_log_prob 逻辑进入主链

### 4.4 worker 完成标准

- [ ] actor 和 rollout_ref 职责分离正确
- [ ] actor 不承担 rollout / VAE / reward
- [ ] rollout_ref 不承担 optimizer / actor 更新
- [ ] 两侧都命中 Case3 专用 role-aware 初始化

### 4.5 sync 完成标准

- [ ] disaggregate dance 模式下能真实同步权重
- [ ] 同步目标是 Case3 dance 模块，不是旧 pipeline 结构假设
- [ ] rollout 前同步成功可从日志或断言验证

## 5. 运行验收

文档设计得再完整，最后还是要靠运行结果收口。

### 5.1 冒烟验收

- [ ] 用 Case3 专用脚本和 YAML 运行 `max_train_steps=1`
- [ ] 日志需要证明：
  - [ ] 命中 Case3 专用入口分支
  - [ ] 命中 Case3 专用 dataloader
  - [ ] 命中 Case3 专用 `fit_dance_case3_dis()`
  - [ ] 每步顺序为 `sync -> rollout_ref_wg.generate_sequences -> actor_wg.update_actor`

### 5.2 稳定性验收

- [ ] 固定配置连续运行 `>=10 steps`
- [ ] 验证：
  - [ ] 无 NaN
  - [ ] 无 Inf
  - [ ] 无 shape mismatch
  - [ ] 无关键字段缺失
  - [ ] actor 至少发生一次真实 `optimizer.step()`

### 5.3 结果协议验收

- [ ] rollout 返回字段与 `disco_rl` 基线一致
- [ ] trainer 没有重新写入通用 advantage 字段来驱动更新
- [ ] actor 更新使用的输入确实来自 rollout_ref 输出

## 6. 差异审计要求

Case3 迁移过程中可以有最小适配，但不能在不记录的情况下悄悄“重写语义”。

必须对照 `disco_rl` 审计这些函数：

- [ ] `fit_dis`
- [ ] `_build_model_optimizer_dance_dis`
- [ ] `generate_sequences`
- [ ] `update_actor`

对于任何保留差异，都必须记录：

- [ ] 差异点是什么
- [ ] 为什么不能完全照搬
- [ ] 影响范围是什么
- [ ] 是否会影响训练语义

重点审计项：

- [ ] `sigma_schedule`
- [ ] `flux_step`
- [ ] `log_prob` 计算
- [ ] advantage 归一化
- [ ] best-of-n
- [ ] timestep 子采样
- [ ] 梯度更新边界

## 7. 风险项

这些风险在 Case3 上都是真问题，必须在实施时主动看守。

### 7.1 误入通用 diffusion batch 协议

风险：

- `fit_dis()` 没有真正硬切
- `_make_batch_data_dis()` 仍在主链上
- 训练继续要求 `prompt_embeds / negative_prompt_embeds`

防护：

- [ ] Case3 模式下在 trainer 顶部硬切
- [ ] Case3 dataloader 单测直接验证三元组协议

### 7.2 同步 helper 仍绑定 pipeline 结构

风险：

- 当前 relay 同步默认面向 `rollout.pipeline.transformer`
- Case3 dance rollout 未必具备同样结构

防护：

- [ ] 显式适配 Case3 的 `self.transformer -> self.rollout`
- [ ] 用日志或断言证明同步的是对的模块

### 7.3 role 判定不完整导致 actor / rollout_ref 初始化错位

风险：

- actor 误加载 rollout 组件
- rollout_ref 误进入 actor optimizer 路径

防护：

- [ ] `init_model()` 的 role-aware 测试
- [ ] `generate_sequences()` / `update_actor()` 的角色断言

### 7.4 Case4 已有逻辑被回归破坏

风险：

- 为了加 Case3 修改了 Case4 已跑通分支

防护：

- [ ] 保持 Case4 专用逻辑不动
- [ ] 所有新增判断都以 `dance_case3_mode` 为边界
- [ ] 跑 Case4 现有测试做回归

## 8. 测试清单

下面这些测试场景需要在实现时显式补齐：

- [ ] `main_ppo` 在 Case3 模式下跳过通用文本组件初始化
- [ ] Case3 dataloader 返回 `(encoder_hidden_states, encoder_attention_mask, caption)`
- [ ] `fit_dis()` 在 Case3 模式下硬切到 `fit_dance_case3_dis()`
- [ ] Case3 每步顺序固定为 `sync -> rollout_ref_wg.generate_sequences -> actor_wg.update_actor`
- [ ] worker 侧 `actor` 和 `rollout_ref` 分别命中不同初始化职责
- [ ] sync helper 在 Case3 下同步的是 dance 模块权重
- [ ] Case3 开关开启但条件不满足时 fail-fast
- [ ] Case4 现有测试行为不回归

## 9. 最终验收标准

只有同时满足下面这些条件，Case3 才算迁移完成：

- [ ] 1 step 冒烟跑通
- [ ] 10 steps 稳定运行
- [ ] 无 NaN / shape mismatch / key missing
- [ ] rollout 返回字段与 `disco_rl` 基线一致
- [ ] actor 至少发生一次真实 `optimizer.step()`
- [ ] 日志能证明命中了 Case3 专用分支
- [ ] 日志能证明没有落入通用 diffusion reward / advantage 路径
- [ ] Case4 现有行为未被回归破坏

## 10. 一句话结论

Case3 的正确迁移路线，不是继续往当前通用 `fit_dis()` 里补兼容，而是沿用 Case4 已验证成功的哲学，单独拉出一条 disaggregate 版 Dance 专用主链：同样的 latent 数据协议，同样的 worker 内 GRPO 闭环，但额外补齐一层可靠的 actor -> rollout_ref 跨组权重同步。
