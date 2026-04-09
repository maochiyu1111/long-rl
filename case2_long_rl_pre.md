# Case2（long-rl，目标态预演）：理想中的迁移成功执行流程

本文不是描述当前仓库已经实现的真实行为，而是按“Case2 迁移已经完成且整体风格与 Case3/Case4 成功经验一致”的目标态来写，用来回答：如果 `long-rl` 真正把 `disco_rl` 的 Case2 迁移干净了，执行流程应该长什么样。

目标主链路可以先浓缩成一句话：

```text
case2_dance.sh
  -> python -m verl.trainer.main_ppo
  -> TaskRunner.run()
  -> RayPPOTrainer._create_dance_latent_dataloader("dance_case2_mode")
  -> RayPPOTrainer.init_workers_dis()
  -> actor/rollout_ref setup_dist + init_model
  -> RayPPOTrainer.fit_dance_dual_rollout_dis_async(schedule="plain_async")
  -> _make_dance_dual_rollout_prompt_batches(schedule="plain_async")
  -> actor_wg.generate_sequences_dance_async(...) / rollout_ref_wg.generate_sequences_dance_async(...)
  -> ray.wait + DataProtoFuture.get()
  -> driver 侧 dual-adv 标准化 + best-of-n + 拼批
  -> actor_wg.update_actor_dance_async(samples, step_weight)
```

## 1. 命中的是哪条目标分支

Case2 的成功迁移不应再复活旧的 `actor_rollout_ref.actor.disco` 语义，而应像 Case3/Case4 一样使用一个明确的新模式位。

固定命中条件应为：

- `trainer.diffusion=true`
- `trainer.disaggregate=true`
- `trainer.pipelined_micro_batch=false`
- `algorithm.adv_estimator=grpo`
- `actor_rollout_ref.actor.dance_case2_mode=true`
- 不再使用 `actor_rollout_ref.actor.disco`

这意味着：

- Case2 仍然是 diffusion 训练
- 仍然是 disaggregate 双 WorkerGroup 拓扑
- 但 trainer 入口不再借旧名 `fit_disco_pipelined()` 承载语义
- 而是进入一条共用 async dual-rollout 骨架里的 `schedule="plain_async"` 分支

这样做的原因很直接：

- 当前 `long-rl` 已经把 `disco` 判为 obsolete
- `disco_rl` 里的异步双源 rollout 主链本来就复用了同一个旧入口
- 如果迁移时继续依赖 `disco`，只会把已经清理掉的旧分支重新带回来

## 2. 入口：`case2_dance.sh`

理想成功态下，Case2 应有自己的专用入口：

```bash
bash case2_dance.sh
```

脚本职责应与 `case3_dance.sh`、`case4_dance.sh` 保持同一哲学，但适配 Case2 的异步调度：

1. 设置基本环境变量。
2. 指向独立配置文件：
   - `examples/diffusion/config_video_diffusion_case2_dance.yaml`
3. 校验与异步 disaggregate 相关的关键参数：
   - `trainer.nnodes`
   - `trainer.n_gpus_per_node`
   - `trainer.disaggregate_actor_n_gpus_per_node`
   - `trainer.disaggregate_rollout_ref_n_gpus_per_node`
   - `data.train_batch_size`
   - `data.gen_batch_size`
   - `actor_rollout_ref.rollout.num_generations`
   - `actor_rollout_ref.rollout.bestofn`
   - `actor_rollout_ref.actor.gradient_accumulation_steps`
   - `bestofn <= actor_wg.world_size`
   - prompt 拆批与 world size 的整除关系
   - driver 拼批后能够稳定触发 actor update 的最小 batch 约束
4. 通过 Hydra override 启动 Case2 专用 trainer 分支。

和 Case3/Case4 一样，Case2 不应复用通用 diffusion 配置文件做“就地改参”，而应该拥有自己的独立 YAML 和 shell 入口，这样看到脚本名就能知道是在跑异步 Dance Case2，而不是旧的 `disco` 残留。

## 3. `main_ppo.py` 入口层的目标行为

Case2 成功迁移后，`TaskRunner.run()` 应像处理 Case3/Case4 那样，在 driver 侧先识别专用模式，再跳过会把流程提前带回通用 PPO/diffusion 路径的初始化。

理想状态下，`dance_case2_mode=true` 时应做这些事：

1. `_build_tokenizer_and_processor(config)` 直接返回 `(None, None)`。
2. 不初始化通用 reward manager：
   - `reward_fn = None`
   - `val_reward_fn = None`
3. 不初始化通用 RL dataset / sampler / collate_fn。
4. 直接把 “Case2 latent dataloader + disaggregate async trainer” 所需的最小对象传给 `RayPPOTrainer`。

原因与 Case3/Case4 一致：

- Case2 的输入协议不是通用 diffusion 的 `prompt_embeds / negative_prompt_embeds`
- 奖励不是 trainer 侧 `compute_reward(...)`
- advantage 也不是 `compute_advantage_diffusion(...)`
- 它的主链是 “worker 侧 rollout/reward + driver 侧 dual-adv/best-of-n + actor 异步更新”

只要入口层还强制初始化通用 tokenizer、reward manager、RL dataset，Case2 就会在专用链真正开始前先被旧逻辑带偏。

## 4. `RayPPOTrainer` 的目标初始化改造

Case2 迁移首先需要在 trainer 层补齐一组可复用的识别与 fail-fast 逻辑，而不是再给 Case2 单独复制一套专名 helper：

- `_is_dance_mode_enabled(mode_name)`
- `_dance_dual_rollout_mismatch_reasons(mode_name, expect_pipelined_micro_batch)`
- `_is_dance_dual_rollout_mode(mode_name, expect_pipelined_micro_batch)`
- `_make_dance_dual_rollout_prompt_batches(schedule)`
- `fit_dance_dual_rollout_dis_async(schedule)`

同时，`_validate_config()` 的语义也应改成：

- 继续拒绝旧的 `actor_rollout_ref.actor.disco`
- 但错误信息要明确提示：如果你要跑迁移后的 Case2，请改用 `dance_case2_mode`

如果实现上为了 call site 可读性，还想保留：

- `_is_dance_case2_enabled()`

那它也应只是：

```text
_is_dance_case2_enabled()
  -> _is_dance_mode_enabled("dance_case2_mode")
```

而不应再引入一整套 Case2 专属的 mode / mismatch / fit / dataloader 实现。

也就是说，Case2 的迁移目标不是“重新支持 disco”，而是“把 disco_rl 里的 Case2 训练语义，迁移成 long-rl 体系下的一条 async dual-rollout 基线”。

## 5. Case2 的 dataloader 协议

Case2 的数据协议应与 Case3/Case4 保持一致，继续使用 latent dataloader，不回退到通用 diffusion prompt 协议。

理想中的 dataloader 返回值固定为：

```text
(encoder_hidden_states, encoder_attention_mask, caption)
```

因此这里不必再新增 `_create_dance_case2_dataloader()`，直接复用现有泛化入口即可：

```text
_create_dance_latent_dataloader("dance_case2_mode")
```

这样 Case2、Case3、Case4 的输入层就能统一为同一份 latent 数据协议，差异只保留在训练调度层：

- Case4：colocated，同步 rollout + update
- Case3：disaggregate，同步 rollout_ref -> actor
- Case2：disaggregate，异步 actor/rollout_ref 双源 rollout -> driver 聚合 -> actor 异步 update

## 6. `init_workers_dis()` 仍然复用，但 worker 语义要升级

Case2 不需要发明第三种拓扑，仍应复用当前仓库已经存在的：

```text
RayPPOTrainer.init_workers_dis()
  -> actor_wg
  -> rollout_ref_wg
  -> actor_setup_dist / rollout_ref_setup_dist
  -> init_model()
```

`setup_dist()` 仍负责建立：

- default process group
- `actor_pg`
- `rollout_ref_pg`
- actor 与 rollout/ref 之间的跨组同步基础能力

但 Case2 相比 Case3 多一个要求：

- actor 组自己不仅要更新，还要承担一部分异步 rollout

因此 Case2 的关键不是拓扑变了，而是 actor worker 的能力边界变了。

## 7. Worker 初始化职责拆分

### 7.1 `rollout_ref` 侧

`role="rollout_ref"` 的 worker 职责与 Case3 基本一致，应负责：

- 加载 rollout 模型副本
- 加载 VAE
- 按配置决定是否初始化 VideoAlign inferencer
- 只做 rollout / decode / reward

也就是说，rollout_ref 仍然是“纯 rollout 侧 worker”。

### 7.2 `actor` 侧

Case2 与 Case3 最大的差异就在这里。

`role="actor"` 的 worker 不再只是“纯更新器”，而应变成“可更新 + 可异步 rollout”的混合 worker。它至少要具备：

- 可训练 actor `self.transformer`
- optimizer / lr scheduler
- 供 actor 路异步 rollout 使用的解码与奖励能力：
  - `self.vae`
  - 可选 `self.inferencer`

实现上，目标态应优先尊重 `disco_rl` 的原始职责设计：

- 如果源 Case2 依赖 actor 侧同时持有训练用 `self.transformer` 与 rollout 用 `self.rollout`，第一版迁移就应先按这一语义对齐
- 只有在完成语义对齐之后，才讨论是否进一步收敛成 `long-rl` 当前更统一的模型组织方式

也就是说，这里的优先级应当是：

1. 先把源 Case2 的 actor 路异步 rollout 能力迁干净。
2. 再评估是否有必要把 actor 侧 rollout 实现重构到更接近当前 Case4 的形态。

这样可以避免文档从“迁移规划”滑向“顺手重设计”。

## 8. Case2 需要专用的 prompt 拆批逻辑

Case2 不能像 Case3/Case4 那样每个 step 直接把 dataloader 返回的一个 batch 送去 rollout；它需要先把一个 latent batch 拆成多个异步提交单元。

因此 trainer 层更适合新增可复用 helper，例如：

```text
_make_dance_dual_rollout_prompt_batches(schedule)
```

其目标语义应与 `disco_rl` 里的 `_make_batch_prompts()` 对齐：

1. 从 latent dataloader 取出一批 `(encoder_hidden_states, encoder_attention_mask, caption)`。
2. 组装为多个 `DataProto` prompt batch，而不是一个大 `DataProto`。
3. 在 `num_generations` 与 rollout world size 可以整齐对齐时，保留“单样本 -> 一组 seeds -> 一次 group rollout”的语义。
4. 在不能整齐对齐时，退回到按 world size 切 chunk 的语义。
5. 每个 prompt batch 额外附带异步调度需要的元信息，例如：
   - `caption`
   - `use_seed`
   - `global_step`
   - `round`
   - `dispatch_role`
其中：

- Case2 只消费异步调度所需的公共字段
- “怎么拆 batch、什么时候 use_seed=true、什么时候按 world size 切 chunk” 这层不应重复实现

这里更稳妥的原则是：

- 只要不破坏当前 `long-rl` 的 `DataProto` 约定，优先对齐源 Case2 实际依赖的数据承载方式
- 如果源实现明确依赖 `non_tensor_batch["caption"]`，第一版迁移就不应为了风格统一而强行改协议

协议收敛可以留到迁移跑通之后再做，而不应在规划阶段提前扩大改动面。

## 9. `fit_dance_dual_rollout_dis_async(schedule="plain_async")` 的目标主循环

Case2 的训练入口不应再叫 `fit_disco_pipelined()`，但也没必要再复制一份只服务 Case2 的大循环。更合理的目标态是先落一条可复用的 trainer 主骨架，例如：

```text
fit_dis()
  -> if dance_case2_mode:
       init_workers_dis()
       fit_dance_dual_rollout_dis_async(schedule="plain_async")
```

`fit_dance_dual_rollout_dis_async(schedule="plain_async")` 每个 step 的理想顺序应固定为：

1. 调 `_make_dance_dual_rollout_prompt_batches(schedule="plain_async")` 生成一组 prompt batches。
2. 在当前 global step 的异步 rollout 提交前执行一次 actor -> rollout_ref 权重同步。
3. 将前一部分 prompt batches 提交给 `actor_wg.generate_sequences_dance_async(...)`。
4. 将剩余 prompt batches 提交给 `rollout_ref_wg.generate_sequences_dance_async(...)`。
5. 用 `ray.wait` 按完成顺序回收异步 rollout 结果。
6. 在 driver 侧对每个 rollout batch 做：
   - `vq/mq` group-wise 标准化
   - best-of-n 筛选
   - 拼批 / 累积
7. 满足累计条件后，调用 `actor_wg.update_actor_dance_async(samples, step_weight)`。
8. 等所有 update futures 完成，再进入下一 global step。

这里要特别强调两点：

- Case2 的核心是“异步双源 rollout + driver 聚合 + actor 异步更新”
- 而不是把 Case3 的同步主循环简单改成 `blocking=False`
- 也就是说，Case2 更适合作为 async dual-rollout 骨架的第一落地点，而不是再 fork 一条近似循环

## 10. actor / rollout_ref 的异步分工

理想成功态下，Case2 会形成一条非常明确的异步分工：

### 10.1 actor 路异步 rollout

actor 路负责：

- 消耗一部分 prompt batches
- 用 actor 当前策略即时 rollout
- 尽早产出可进入 actor 更新的样本

它的意义不是“角色更纯”，而是“把一部分 rollout 放到 actor 组本地消化”，从而保留 `disco_rl` Case2 的异步吞吐特征。

### 10.2 rollout_ref 路异步 rollout

rollout_ref 路负责：

- 消耗剩余 prompt batches
- 使用跨组同步后的策略副本做 rollout
- decode 视频并打 VideoAlign 奖励

它仍然是更偏 rollout 的一侧。

### 10.3 driver 侧聚合

driver 负责：

- 统一接收 actor 路和 rollout_ref 路返回的轨迹
- 做 dual-adv 标准化
- 做 best-of-n
- 做拼批与 gradient accumulation 调度

也就是说，Case2 的“中枢”在 driver，而不是像 Case3/Case4 那样更多把优势计算收进 worker 内部。

## 11. `generate_sequences_dance_async()` 需要什么语义

Case2 更适合新增一个可复用的异步 rollout worker 接口，例如：

```text
generate_sequences_dance_async(prompts: DataProto) -> DataProtoFuture
```

它应满足这些要求：

1. `Dispatch.DP_COMPUTE_PROTO`
2. `blocking=False`
3. 允许在 `role="actor"` 和 `role="rollout_ref"` 上都被调用
4. rollout 数学与返回字段仍与 Case3/Case4 的 Dance rollout 一致：
   - `timesteps`
   - `latents`
   - `next_latents`
   - `log_probs`
   - `vq_rewards`
   - `mq_rewards`
   - `encoder_hidden_states`
   - `encoder_attention_mask`
   - `meta_info["sigma_schedule"]`

换句话说，Case2 的异步化不应另起一套 rollout 协议；它变化的是调度方式，不是轨迹数据结构。

## 12. dual-adv 与 best-of-n 在 Case2 中仍放在 driver 侧

这是 Case2 与 Case3/Case4 的又一个本质差异。

Case3/Case4 当前更适合把：

- advantage 标准化
- best-of-n
- PPO loss 组织

收进 `_update_actor_dance()`。

但 Case2 若要对齐 `disco_rl` 的主语义，应继续保持：

- rollout worker 只负责产出轨迹与 reward
- driver 侧负责 dual-adv 标准化与 best-of-n
- actor worker 只负责消费“已经筛好”的样本做更新

因此，如果目标是最大程度对齐 `disco_rl` Case2 的原始训练职责，那么更稳妥的基线方案是新增一个可复用的：

```text
update_actor_dance_async(samples, step_weight)
```

而不是直接复用当前 Case3/Case4 的 `_update_actor_dance()` 实现。因为后者默认把 advantage/best-of-n 也收在 worker 内，不符合 `disco_rl` Case2 的原始调度职责划分。

不过这里更准确的表述应是：

- “driver 侧做 dual-adv/best-of-n + actor 异步更新” 是最贴近源 Case2 语义的首选迁移路径
- 不是唯一允许的实现形式
- 如果后续证明可以在不改变外部训练语义的前提下复用现有 worker-side update，则那属于迁移完成后的第二阶段收敛，而不是第一版规划默认要做的事

## 13. gradient accumulation 的目标实现

Case2 的异步更新仍应保留源链路的一个关键点：

- driver 侧决定什么时候只是累计梯度
- 什么时候真正 `optimizer.step()`

因此 `update_actor_dance_async(samples, step_weight)` 的输入里应明确带上类似：

- `step_weight=false`
- `step_weight=true`

这样的元信息，让 actor worker 能区分：

- 这次只 `backward`
- 还是要执行 `clip_grad -> optimizer.step -> lr_scheduler.step -> zero_grad`

但 `disco_rl` 里那套“根据 step time 自动调 actor_batch_num、并在测试分支直接 `SystemExit`”的逻辑不应原样迁入。

目标态更适合：

- 用固定配置项控制 actor 路 rollout 配额
- 用明确的 accumulation 规则控制何时 step
- 把计时报告作为观测，而不是训练控制分支

## 14. Case2 与 Case3/Case4 的本质差异

如果把三者放在一起看，理想成功态下它们的关系应非常清晰：

### 14.1 与 Case3 的关系

- 两者都是 disaggregate 双 WorkerGroup
- 两者都使用 latent dataloader 三元组协议
- 两者都需要 actor -> rollout_ref 同步能力
- 但 Case3 是同步 rollout_ref -> actor 链路
- Case2 是异步 actor/rollout_ref 双源 rollout -> driver 聚合 -> actor update

### 14.2 与 Case4 的关系

- 两者都可以复用当前 long-rl 已验证的 Dance worker 内 rollout 数学
- 两者都不应走通用 diffusion reward / advantage 逻辑
- 但 Case4 是 colocated 单组闭环
- Case2 是 disaggregate 异步调度闭环

所以 Case2 的迁移本质不是“再写一个 Case3”，也不是“把 Case4 改成 async”，而是把已经在 Case3/Case4 中验证过的数据协议和 rollout/update 核心，落进一条 async dual-rollout trainer 骨架里。

## 15. 需要落地修改的代码面

如果按这个目标态实施，核心改动面应集中在这些位置：

- `verl/workers/config/actor.py`
  - 增加 `dance_case2_mode`
- `verl/trainer/config/actor/actor.yaml`
  - 暴露 `dance_case2_mode: false`
- `verl/trainer/main_ppo.py`
  - 把 Case2 纳入 tokenizer / reward / dataset 的跳过逻辑
- `verl/trainer/ppo/ray_trainer.py`
  - 增加可复用的 mode 校验、prompt batching、异步 dual-rollout fit 主循环
  - Case2 作为 `schedule="plain_async"` 的第一落地点
- `verl/workers/fsdp_workers.py`
  - 增加 dual-rollout role-aware init 分支
  - 增加 actor/rollout_ref 共用的异步 rollout 接口
  - 增加可复用的异步 update 接口
- `examples/diffusion/config_video_diffusion_case2_dance.yaml`
  - 新增 Case2 专用配置
- `case2_dance.sh`
  - 新增 Case2 专用入口

## 16. 一句话总结

Case2 的成功迁移，本质上就是先把 `disco_rl` 里“异步双源 rollout + driver 聚合 + actor 异步更新”的训练语义，落成 `long-rl` 里的 async dual-rollout 骨架：沿用 Case3/Case4 已经成功的 latent 数据协议、专用 YAML/脚本、fail-fast 配置校验和 Dance rollout 数学，但不再恢复旧的 `disco` 开关与 `fit_disco_pipelined()` 历史包袱。
