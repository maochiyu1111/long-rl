# Case3（long-rl，目标态预演）：理想中的迁移成功执行流程

本文不是描述当前仓库已经实现的真实行为，而是按“Case3 迁移已经完成且结构与 Case4 成功经验一致”的目标态来写，用来回答：如果 `long-rl` 真正把 `disco_rl` 的 Case3 迁移干净了，执行流程应该长什么样。

目标主链路可以先浓缩成一句话：

```text
case3_dance.sh
  -> python -m verl.trainer.main_ppo
  -> TaskRunner.run()
  -> RayPPOTrainer.init_workers_dis()
  -> actor/rollout_ref setup_dist + init_model
  -> RayPPOTrainer.fit_dance_case3_dis()
  -> actor -> rollout_ref 权重同步
  -> rollout_ref_wg.generate_sequences(new_batch)
  -> actor_wg.update_actor(rollout_batch)
```

## 1. 命中的是哪条目标分支

理想中的 Case3 迁移完成后，必须由一组明确的硬条件命中，不能再靠旧的 `disco` 语义残留去猜。

固定命中条件应为：

- `trainer.diffusion=true`
- `trainer.disaggregate=true`
- `trainer.pipelined_micro_batch=false`
- `algorithm.adv_estimator=grpo`
- `actor_rollout_ref.actor.dance_case3_mode=true`
- 不再使用 `actor_rollout_ref.actor.disco`

这意味着：

- Case3 仍然是 diffusion 训练
- 仍然是 disaggregate 双 WorkerGroup 拓扑
- 但它不再走当前通用 `fit_dis()` 的 diffusion batch 组织逻辑
- 而是像 Case4 一样，走一条专门为 DanceGRPO Case3 设计的硬分叉主链

## 2. 入口：`case3_dance.sh`

理想成功态下，Case3 会有一个专用入口：

```bash
bash case3_dance.sh
```

脚本职责与 `case4_dance.sh` 保持同一哲学，但适配 Case3：

1. 设置基本环境变量。
2. 指向新配置文件：
   - `examples/diffusion/config_video_diffusion_case3_dance.yaml`
3. 校验与 disaggregate 相关的关键参数：
   - `trainer.nnodes`
   - `trainer.n_gpus_per_node`
   - `trainer.disaggregate_actor_n_gpus_per_node`
   - `trainer.disaggregate_rollout_ref_n_gpus_per_node`
   - `data.train_batch_size`
   - `data.gen_batch_size`
4. 通过 Hydra override 启动：

```bash
python3 -m verl.trainer.main_ppo \
  --config-path=examples/diffusion \
  --config-name=config_video_diffusion_case3_dance \
  ...
```

和 Case4 一样，Case3 不应复用通用 diffusion 配置文件做“就地改参”，而应该有自己的独立 YAML，这样入口一眼就能看出是在跑专用硬分支。

## 3. `main_ppo.py` 入口层的目标行为

Case3 的成功迁移应复用 Case4 已证明可行的入口策略：先在 driver 侧识别专用模式，再跳过会把训练提前带回旧路径的通用初始化。

理想状态下，`TaskRunner.run()` 在 Case3 模式下应做这些事：

1. 识别 `dance_case3_mode=true`。
2. 跳过通用 tokenizer / processor 初始化。
3. 跳过通用 reward manager 初始化。
4. 跳过标准 RL dataset / sampler / collate_fn 构造。
5. 直接把 “Case3 latent dataloader + disaggregate trainer” 所需的最小对象传给 `RayPPOTrainer`。

这样做的原因和 Case4 一样直接：

- Case3 的输入协议不是通用 diffusion 的 `prompt_embeds/negative_prompt_embeds`
- 而是 `(encoder_hidden_states, encoder_attention_mask, caption)`
- 奖励也不在 trainer 侧走 `compute_reward(...)`
- 而是在 rollout/ref worker 内部由 VideoAlign 闭环完成

如果入口层继续强制走通用 tokenizer、reward manager、RL dataset 初始化，就会在还没进入 Case3 专用路径之前先偏到旧链路里。

## 4. Trainer 初始化：仍是 `init_workers_dis()`，但语义改成 Dance Case3

Case3 和 Case4 最大的结构差异不在 GRPO 数学形式，而在执行拓扑。

Case4 的成功迁移是 colocated 单组 worker。  
Case3 的目标态则继续复用当前仓库已经存在的 `init_workers_dis()` 拓扑：

1. `RayPPOTrainer.init_workers_dis()`
2. 创建 `actor_wg`
3. 创建 `rollout_ref_wg`
4. 分别对两组 worker 调 `actor_setup_dist` / `rollout_ref_setup_dist`
5. 进入各自 `init_model()`

这里不需要再发明第三种拓扑；Case3 的关键不是推翻 `init_workers_dis()`，而是把它从“通用 diffusion disaggregate 骨架”变成“Dance Case3 可用骨架”。

### 4.1 `setup_dist()` 在目标态里承担什么

`setup_dist()` 仍负责建立：

- default process group
- `actor_pg`
- `rollout_ref_pg`
- actor 与 rollout/ref 的跨组同步基础通信域

但 Case3 成功态有一个比当前更明确的要求：

- 同步能力必须服务于 Dance disaggregate
- 不再只适配 `rollout.pipeline.transformer`
- 而要能对接 Case3 的真实模块关系：
  - actor 侧：`self.transformer`
  - rollout_ref 侧：`self.rollout`

也就是说，`setup_dist()` 的问题不是要不要保留，而是要让后续同步逻辑真正和 Case3 的模型结构匹配。

## 5. Worker 初始化职责拆分

Case3 成功态下，两组 worker 会像 `disco_rl` 那样职责清晰分离，而不是共享一个 actor/rollout 闭环对象。

### 5.1 actor 侧

`role="actor"` 的 worker 应负责：

- 加载可训练 transformer
- 按 actor 组做 FSDP 包装
- 创建 optimizer
- 创建 lr scheduler
- 作为唯一的策略更新源头

它的职责是：

- 接收 rollout 返回的轨迹
- 用当前策略重算 `new_log_probs`
- 执行 GRPO / PPO 更新

它不负责：

- 采样
- VAE decode
- VideoAlign 奖励

### 5.2 rollout_ref 侧

`role="rollout_ref"` 的 worker 应负责：

- 加载 rollout 模型副本
- 加载 VAE
- 按配置决定是否初始化 VideoAlign inferencer
- 在 rollout 组内执行 diffusion 采样与奖励打分

它的职责是：

- 消费 dataloader 给出的 `encoder_hidden_states / encoder_attention_mask / caption`
- 生成完整 rollout 轨迹
- 解码视频
- 打 VQ / MQ 奖励
- 把训练样本回传给 actor 组

### 5.3 两边都走 Case3 专用 role-aware 分支

和 Case4 一样，Case3 也不应该继续用“通用函数里顺带兼容一点 Dance”这种写法。

理想成功态下：

- `init_model()` 会先判断 `dance_case3_mode`
- 命中后直接进入 Case3 专用初始化
- 再根据 `role="actor"` 或 `role="rollout_ref"` 分别构建对应对象

这样才能保证 Case3 的职责拆分是稳定的，而不是在通用初始化里夹杂偶然兼容。

## 6. Case3 的 dataloader 协议

Case3 的输入协议应与 `disco_rl` 和 Case4 成功迁移保持一致，继续使用 latent 数据，不回退到通用 diffusion prompt 协议。

理想中的 dataloader 返回值固定为：

```text
(encoder_hidden_states, encoder_attention_mask, caption)
```

driver 每步拿到 batch 后，应组装为：

```python
DataProto.from_single_dict(
    {
        "encoder_hidden_states": encoder_hidden_states,
        "encoder_attention_mask": encoder_attention_mask,
    },
    meta_info={"caption": caption},
)
```

这意味着 Case3 和 Case4 的数据协议是一致的，区别只在执行拓扑：

- Case4：同组 worker 接着 rollout + update
- Case3：rollout/ref 组先采样，再把轨迹送给 actor 组更新

## 7. `fit_dance_case3_dis()` 的目标主循环

Case3 目标态下，`fit_dis()` 顶部应先检查是否命中 `dance_case3_mode`。  
如果命中，就直接硬切到：

```text
fit_dis()
  -> fit_dance_case3_dis()
```

`fit_dance_case3_dis()` 每个训练 step 的理想顺序应固定为：

1. 从 latent dataloader 取一批 `(encoder_hidden_states, encoder_attention_mask, caption)`
2. 组装 `DataProto.from_single_dict(..., meta_info={"caption": caption})`
3. 在 rollout 前执行一次 actor -> rollout_ref 权重同步
4. `rollout_ref_wg.generate_sequences(new_batch)`
5. `actor_wg.update_actor(rollout_batch)`
6. 记录 actor metrics，进入下一步

这条链路是 Case3 最核心的定义。  
只要主循环里重新出现通用 diffusion `gen_batch`、`compute_reward`、`compute_advantage_diffusion`，就说明它已经偏出目标态了。

## 8. rollout 前的跨组权重同步

Case3 比 Case4 多出来的关键一步，就是 rollout 前的跨组参数对齐。

原因很简单：

- rollout_ref 侧使用的是 rollout 模型副本
- actor 侧持有的是最新可训练策略
- 如果不同步，rollout 采样和 actor 更新就会基于不同版本的策略

因此理想中的 `fit_dance_case3_dis()` 必须在每个 rollout 前执行：

```text
actor -> rollout_ref 权重同步
```

目标态里这套同步应满足两点：

1. 继续复用当前仓库已经具备的 disaggregate 同步调度能力。
2. 但同步对象必须改成 Dance Case3 的真实模块，而不是继续假设 rollout 端一定存在 `pipeline.transformer`。

换句话说，Case3 的成功迁移不是“有没有同步”，而是“同步 helper 终于同步到了对的模块”。

## 9. `generate_sequences()` 在 Case3 中负责什么

Case3 的 rollout/ref 侧 `generate_sequences()` 与 Case4 保持相同的数据学意义：

1. 读取：
   - `encoder_hidden_states`
   - `encoder_attention_mask`
   - `caption`
2. 在 `use_group=true` 时按 `num_generations` 展开条件与 caption
3. 构造 `sigma_schedule`
4. 用 diffusion transformer 逐 timestep rollout
5. 通过 `flux_step(...)` 记录旧策略 `log_probs`
6. 用 VAE decode 视频
7. 用 VideoAlign 计算 `VQ / MQ`
8. 返回轨迹级 `DataProto`

理想返回字段应与 `disco_rl` 基线保持一致：

- `timesteps`
- `latents`
- `next_latents`
- `log_probs`
- `vq_rewards`
- `mq_rewards`
- `encoder_hidden_states`
- `encoder_attention_mask`
- `meta_info["sigma_schedule"]`

这部分和 Case4 的目标是一致的，Case3 不应另起一套 rollout 协议。

## 10. `update_actor()` 在 Case3 中负责什么

Case3 的 actor 侧 `update_actor()` 也应沿用与 Case4 相同的 Dance 闭环更新思想，只是数据来源改成 rollout_ref 组。

主要步骤仍是：

1. 恢复 `sigma_schedule`
2. 按 `num_generations` 分组标准化 `vq_advantages / mq_advantages`
3. 做 best-of-n 筛选
4. 做 timestep shuffle 与子采样
5. 用当前 actor 的 `self.transformer` 重算 `new_log_probs`
6. 用 PPO clip loss 组合 `vq_loss / mq_loss`
7. `backward -> clip_grad -> optimizer.step -> lr_scheduler.step`

所以 Case3 和 Case4 的关键差异不是 actor 更新数学形式，而是：

- Case4：rollout 和 update 在同组 worker 内闭环
- Case3：rollout 轨迹由 rollout_ref 组产出，再交给 actor 组更新

## 11. Case3 与 Case4 的本质差异

如果把两者放在一起看，理想成功态下它们的关系应该非常清晰：

### 11.1 相同点

- 都是 DanceGRPO 的专用硬分支
- 都使用 latent dataloader 三元组协议
- 都在 worker 内完成奖励与 advantage 闭环
- 都不走通用 diffusion reward / advantage 路径

### 11.2 不同点

- Case4 是 colocated 单组 worker
- Case3 是 disaggregate 双组 worker
- Case4 的闭环在一个 worker group 内完成
- Case3 的闭环需要 actor 和 rollout_ref 两组协作
- Case3 比 Case4 多一个 rollout 前同步步骤

因此，Case3 的迁移本质不是再写一遍 Case4，而是把 Case4 已经验证成功的“专用硬分叉 + worker 内闭环”思路扩展到 disaggregate 双 WorkerGroup 场景。

## 12. Case3 目标态里明确不走的路径

为了避免后续实施时再被当前通用 diffusion 逻辑带偏，Case3 的目标态必须明确这些路径不属于主链：

- 不走 `_make_batch_data_dis()` 当前的 `prompt_embeds / negative_prompt_embeds` 协议
- 不走 `compute_reward(...)`
- 不走 `compute_advantage_diffusion(...)`
- 不走 critic worker
- 不走 ref worker
- 不走 reward model worker
- 不走通用 diffusion rollout 路径

这些路径都属于当前框架的通用 diffusion/PPO 逻辑；Case3 迁移成功的标志，就是它们不再出现在主执行链里。

## 13. 一句话总结

Case3 的成功迁移，本质上就是把 Case4 已经跑通的“专用硬分叉、最小污染主路径、失败时明确 fail-fast”的思路，扩展到 disaggregate 双 WorkerGroup 协作场景：同样的数据协议，同样的 worker 内 GRPO 闭环，但在 rollout 前多出一层可靠的 actor -> rollout_ref 跨组权重同步。
