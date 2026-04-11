# Case1（long-rl，目标态预演）：理想中的迁移成功执行流程

本文不是描述当前仓库已经实现的真实行为，而是按“Case1 迁移已经完成且整体风格与 Case2/Case3/Case4 成功经验一致”的目标态来写，用来回答：如果 `long-rl` 真正把 `disco_rl` 的 Case1 迁移干净了，执行流程应该长什么样。

目标主链路可以先浓缩成一句话：

```text
case1_dance.sh
  -> python -m verl.trainer.main_ppo
  -> TaskRunner.run()
  -> RayPPOTrainer._create_dance_latent_dataloader("dance_case1_mode")
  -> RayPPOTrainer.init_workers_dis()
  -> actor/rollout_ref setup_dist + init_model
  -> RayPPOTrainer.fit_dance_dual_rollout_dis_async(schedule="pipelined_micro_batch")
  -> _make_dance_dual_rollout_prompt_batches(schedule="pipelined_micro_batch")
  -> actor_wg.generate_sequences_dance_async(...) / rollout_ref_wg.generate_sequences_dance_async(...)
  -> ray.wait + driver 侧 dual-adv + best-of-n + micro-batch 累积
  -> actor_wg.update_actor_dance_async(samples，其中 meta_info["step_weight"] 控制是否真正 step)
```

## 1. 命中的是哪条目标分支

Case1 的成功迁移不应再借用旧的 `disco` 开关，也不应继续让 `pipelined_micro_batch=true` 隐式落到历史遗留的 `fit_disco_pipelined()`。

固定命中条件应为：

- `trainer.diffusion=true`
- `trainer.disaggregate=true`
- `trainer.pipelined_micro_batch=true`
- `algorithm.adv_estimator=grpo`
- `actor_rollout_ref.actor.dance_case1_mode=true`
- 不再使用 `actor_rollout_ref.actor.disco`

这意味着：

- Case1 仍然是 diffusion 训练
- 仍然是 disaggregate 双 WorkerGroup 拓扑
- 仍然保留 pipelined micro-batch 训练语义
- 但 trainer 不再借旧名 `fit_disco_pipelined()` 承载语义
- 而是进入一条 Case1/Case2 共用 async dual-rollout 骨架里的 `schedule="pipelined_micro_batch"` 分支

这样做的原因很直接：

- 当前 `long-rl` 已经把 `disco` 视为 obsolete
- `disco_rl` 里的 Case1 虽然复用了旧入口，但它的真实语义是“异步双源 rollout + pipeline 微批更新”
- 如果迁移时继续依赖 `disco` 或旧入口名，只会把已经清理掉的历史分支重新带回来

## 2. 入口：`case1_dance.sh`

理想成功态下，Case1 应有自己的专用入口：

```bash
bash case1_dance.sh
```

脚本职责应与 `case2_dance.sh`、`case3_dance.sh`、`case4_dance.sh` 保持同一哲学，但适配 Case1 的 pipelined 调度：

1. 设置基本环境变量。
2. 指向独立配置文件：
   - `examples/diffusion/config_video_diffusion_case1_dance.yaml`
3. 校验与 pipelined disaggregate 相关的关键参数：
   - `trainer.nnodes`
   - `trainer.n_gpus_per_node`
   - `trainer.disaggregate_actor_n_gpus_per_node`
   - `trainer.disaggregate_rollout_ref_n_gpus_per_node`
   - `data.train_batch_size`
   - `data.gen_batch_size`
   - `actor_rollout_ref.rollout.num_generations`
   - `actor_rollout_ref.rollout.bestofn`
   - `actor_rollout_ref.actor.gradient_accumulation_steps`
   - Case1 专用的 actor 路 prompt 配额或 pipeline window 配置
4. 通过 Hydra override 启动 Case1 专用 trainer 分支。

和 Case2/Case3/Case4 一样，Case1 不应复用通用 diffusion 配置文件做“就地改参”，而应该有自己的独立 YAML 和 shell 入口。这样看到脚本名就能知道是在跑 pipelined Dance Case1，而不是旧 `fit_disco_pipelined()` 的残留。

## 3. `main_ppo.py` 入口层的目标行为

Case1 成功迁移后，`TaskRunner.run()` 应像处理 Case3/Case4 那样，先在 driver 侧识别专用模式，再跳过会把流程提前带回通用 PPO/diffusion 路径的初始化。

理想状态下，`dance_case1_mode=true` 时应做这些事：

1. `_build_tokenizer_and_processor(config)` 直接返回 `(None, None)`。
2. 不初始化通用 reward manager：
   - `reward_fn = None`
   - `val_reward_fn = None`
3. 不初始化通用 RL dataset / sampler / collate_fn。
4. 直接把 “Case1 latent dataloader + disaggregate pipelined trainer” 所需的最小对象传给 `RayPPOTrainer`。

原因与 Case2/Case3/Case4 一致：

- Case1 的输入协议不是通用 diffusion 的 `prompt_embeds / negative_prompt_embeds`
- 奖励不是 trainer 侧 `compute_reward(...)`
- advantage 也不是通用 diffusion trainer 那套 `compute_advantage_diffusion(...)`
- 它的主链是 “worker 侧 rollout/reward + driver 侧 dual-adv/best-of-n + actor 侧 pipelined update”

只要入口层还强制初始化通用 tokenizer、reward manager、RL dataset，Case1 就会在专用链真正开始前先被旧逻辑带偏。

## 4. `RayPPOTrainer` 的目标初始化改造

如果假设 **Case2 已经先迁移完成**，那么 Case1 在 trainer 层就不应再补一整套平行的 Case1 专名 helper。更合理的做法是先复用 Case2 已经落好的 dual-rollout 骨架，再只为 Case1 补最薄的一层 mode wrapper：

- `_dance_dual_rollout_mismatch_reasons(mode_name, expect_pipelined_micro_batch)`
- `_is_dance_dual_rollout_mode(mode_name, expect_pipelined_micro_batch)`
- `_make_dance_dual_rollout_prompt_batches(schedule)`
- `fit_dance_dual_rollout_dis_async(schedule)`

同时，`_validate_config()` 的语义也应改成：

- 继续拒绝旧的 `actor_rollout_ref.actor.disco`
- 如果用户想跑迁移后的 Case1，要明确提示改用 `dance_case1_mode`
- 如果 `dance_case1_mode=true` 但 `trainer.pipelined_micro_batch!=true`，要直接 fail-fast

如果 call site 为了可读性，仍然保留：

- `_is_dance_case1_enabled()`
- `_dance_case1_mismatch_reasons()`
- `_is_dance_case1_mode()`

那么它们也应只是对 Case2 已有通用 helper 的薄封装，例如：

```text
_is_dance_case1_enabled()
  -> bool(self.config.actor_rollout_ref.actor.get("dance_case1_mode", False))

_dance_case1_mismatch_reasons()
  -> _dance_dual_rollout_mismatch_reasons("dance_case1_mode", expect_pipelined_micro_batch=True)

_is_dance_case1_mode()
  -> _is_dance_dual_rollout_mode("dance_case1_mode", expect_pipelined_micro_batch=True)
```

而不应再扩展出一整条独立的 `_dance_case1_*` 复制链，更不需要为了 Case1 先把 Case2 代码重构成一套更抽象的新框架。

也就是说，Case1 的迁移目标不是“重新支持 pipelined disco”，而是“把 disco_rl 里的 Case1 训练语义，迁移成 long-rl 体系下对共用 async dual-rollout 骨架的一种 pipelined 扩展”。

但这里要明确一个实施原则：

- 第一阶段目标是**行为对齐**，优先保证主链语义与 `disco_rl` Case1 一致
- 第二阶段才考虑收敛到更干净的 long-rl 统一实现

也就是说，凡是会改变源 Case1 训练语义的“顺手优化”，都不应混进第一版迁移目标里。

## 5. Case1 的 dataloader 协议

Case1 的数据协议应与 Case2/Case3/Case4 保持一致，继续使用 latent dataloader，不回退到通用 diffusion prompt 协议。

理想中的 dataloader 返回值固定为：

```text
(encoder_hidden_states, encoder_attention_mask, caption)
```

因此这里不应再新增 `_create_dance_case1_dataloader()`，而是直接复用：

```text
_create_dance_latent_dataloader("dance_case1_mode")
```

这样 Case1、Case2、Case3、Case4 的输入层就能统一为同一份 latent 数据协议，差异只保留在训练调度层：

- Case4：colocated，同步 rollout + update
- Case3：disaggregate，同步 rollout_ref -> actor
- Case2：disaggregate，异步 actor/rollout_ref 双源 rollout -> driver 聚合 -> actor 异步 update
- Case1：disaggregate，异步 actor/rollout_ref 双源 rollout -> driver 聚合 -> actor pipelined micro-batch update

## 6. `init_workers_dis()` 仍然复用，但 actor 语义要升级

Case1 不需要发明第三种拓扑，仍应复用当前仓库已经存在的：

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

但 Case1 相比 Case3 多一个明确要求：

- actor 组不只是更新器，还必须承担一部分 rollout

这也是迁移时最需要显式修正的一点：`disco_rl` 现有实现里，`disco=false` 的 actor 初始化不会构建 rollout 所需对象，但 trainer 却仍然把前几个 prompt 提交给 `actor_wg.generate_sequences_asyn_dance(...)`。  
Case1 的目标态不能保留这种“调用路径成立、对象能力却不闭环”的隐患。

## 7. Worker 初始化职责拆分

### 7.1 `rollout_ref` 侧

`role="rollout_ref"` 的 worker 职责与 Case2/Case3 基本一致，应负责：

- 加载 rollout 模型副本
- 加载 VAE
- 按配置决定是否初始化 VideoAlign inferencer
- 只做 rollout / decode / reward

也就是说，rollout_ref 仍然是“偏纯 rollout 侧 worker”。

### 7.2 `actor` 侧

Case1 与 Case3 的最大差异就在这里。

`role="actor"` 的 worker 不再只是“纯更新器”，而应变成“可更新 + 可 rollout”的混合 worker。它至少要具备：

- 可训练 actor `self.transformer`
- optimizer / lr scheduler
- 供 actor 路 rollout 使用的解码与奖励能力：
  - `self.vae`
  - 可选 `self.inferencer`

实现上，这里应明确区分两个阶段：

### 7.2.1 第一阶段：优先做 parity 版迁移

如果 `disco_rl` Case1 的 actor 路异步 rollout 语义依赖：

- actor 侧持有独立 rollout 用模型对象
- rollout 与 update 在对象边界上彼此独立

那么第一版迁移应优先保留这种能力边界，而不是一开始就把它收敛成 Case4 风格的共享实现。

换句话说，第一阶段的判断标准应是：

- 先保证 Case1 的 actor 路 rollout 真能成立
- 先保证与源链路的职责边界尽量一致
- 再谈是否进一步统一到 long-rl 更简洁的实现

### 7.2.2 第二阶段：再评估是否复用 `self.transformer`

在 parity 版跑通之后，才更适合评估是否沿用当前 `long-rl` 在 Case4 已验证过的思路：

- actor 路 rollout 直接复用 `self.transformer`
- 不再保留 actor 侧独立 rollout 副本

这样做的好处是：

- 避免 actor 侧维护一份额外 rollout replica 的同步问题
- 与当前 `_generate_sequences_dance()` 的语义更一致
- 能最大化复用 Case4 已经跑通的 “同一 transformer 既 rollout 又 update” 经验

但这一点应被明确标记为**第二阶段优化方向**，而不是第一版迁移的默认前提。否则最终得到的更像是“Case1 重设计”，而不是“Case1 迁移”。

### 7.3 两边都走 Case1/Case2 共用 role-aware 分支

理想成功态下：

- `init_model()` 会先判断 `dance_case1_mode`
- 命中后直接进入与 Case2 共用的 dual-rollout 初始化
- 再根据 `role="actor"` 或 `role="rollout_ref"` 分别构建对应对象

这样 Case1 的职责拆分才是稳定的，而不是落在“通用初始化里碰巧兼容一点异步路径”。  
如果 Case2 已经先迁完，那么这里更不应该再复制一份 Case1 专属 init 主体。

## 8. Case1 需要复用 Case2 的 prompt 拆批 helper，并只在调度元信息上补充字段

Case1 不能像 Case3/Case4 那样每个 step 直接把 dataloader 返回的一个 batch 送去 rollout；它需要先把一个 latent batch 拆成多个异步提交单元，并且这些单元还要带上 pipeline 调度所需的元信息。

因此 trainer 层更合理的是继续复用 Case2 已经落好的共用 helper：

```text
_make_dance_dual_rollout_prompt_batches(schedule)
```

其目标语义仍应与 `disco_rl` 里 Case1/Case2 共用的 `_make_batch_prompts()` 对齐：

1. 从 latent dataloader 取出一批 `(encoder_hidden_states, encoder_attention_mask, caption)`。
2. 组装为多个 `DataProto` prompt batch，而不是一个大 `DataProto`。
3. 在 `num_generations` 与 worker world size 可以整齐对齐时，保留“单样本 -> 一组 seeds -> 一次 group rollout”的语义。
4. 在不能整齐对齐时，退回到按 world size 切 chunk 的语义。
5. 每个 prompt batch 额外附带 pipelined 调度需要的元信息，例如：
   - `caption`
   - `use_seed`
   - `global_step`
   - `window_id`
   - `micro_batch_id`
   - `is_last_micro_batch`

这里的边界应写清楚：

- “如何从 latent batch 拆出多个 prompt batches” 这层与 Case2 共用
- “给每个 prompt batch 再补哪些 pipeline window 元信息” 这层是 Case1 专属
- 也就是说，Case1 应在共用 helper 上扩展元信息，而不是再复制一份 Case2 已有的拆批主体

这里还有一个比 Case2 更明确的要求：

- prompt batch 的切分规则不能只对 `rollout_ref_wg` 成立
- 还必须同时兼容 `actor_wg` 的 world size 与 pipeline window 切分

因为 Case1 的 prompt batches 会被同时发往 actor 与 rollout_ref 两组 worker。

## 9. `fit_dance_dual_rollout_dis_async(schedule="pipelined_micro_batch")` 的目标主循环

Case1 的训练入口不应再叫 `fit_disco_pipelined()`，但在假设 Case2 已经迁完的前提下，也不应再复制一份近似的 Case1 大循环。更合理的目标态是直接在 Case2 的共用 async dual-rollout trainer 上补一种调度策略：

```text
fit_dis()
  -> if dance_case1_mode:
       init_workers_dis()
       fit_dance_dual_rollout_dis_async(schedule="pipelined_micro_batch")
```

`fit_dance_dual_rollout_dis_async(schedule="pipelined_micro_batch")` 每个 step 的理想顺序应固定为：

1. 调 `_make_dance_dual_rollout_prompt_batches(schedule="pipelined_micro_batch")` 生成一组 prompt batches。
2. 如果沿用 Case2/现有 trainer 已有的跨组同步点，就在当前 pipeline window 开始前执行一次 actor -> rollout_ref 权重同步；但第一版 parity 不应为了补一套全新同步机制而扩大改动面。
3. 将前一部分 prompt batches 提交给 `actor_wg.generate_sequences_dance_async(...)`。
4. 将剩余 prompt batches 提交给 `rollout_ref_wg.generate_sequences_dance_async(...)`。
5. 用 `ray.wait` 按完成顺序回收异步 rollout 结果。
6. 在 driver 侧对每个 rollout batch 做：
   - `vq/mq` group-wise 标准化
   - best-of-n 筛选
   - 拼批 / 累积
7. 对于非窗口末尾的微批，先写入 `samples.meta_info["step_weight"] = False`，再调用 `actor_wg.update_actor_dance_async(samples)`。
8. 对于窗口末尾的微批，先写入 `samples.meta_info["step_weight"] = True`，再调用 `actor_wg.update_actor_dance_async(samples)`。
9. 等当前窗口内所有 update futures 完成，再进入下一 global step。

这里要特别强调三点：

- Case1 的核心是“异步双源 rollout + driver 聚合 + pipelined 微批更新”
- 它不是把 Case3 的同步主循环简单改成 `blocking=False`
- 它也不是重新 fork 一条 Case2 的复制版大循环，而是要把 pipeline window 和 delayed optimizer step 设计进 Case2 已有的共用 async dual-rollout 骨架

另外，主循环里还应保留与当前 Case3/Case4 一致的一层基础训练语义：

- 在进入专用 `fit_dance_dual_rollout_dis_async(schedule="pipelined_micro_batch")` 前完成 `init_workers_dis()`
- 在训练真正开始前保留 checkpoint 恢复入口

这样 Case1 的目标态才不会丢掉 long-rl 现有 trainer 在恢复训练上的基本能力。

## 10. actor / rollout_ref / driver 的 pipelined 分工

理想成功态下，Case1 会形成一条非常明确的三方分工：

### 10.1 actor 路异步 rollout

actor 路负责：

- 消耗一部分 prompt batches
- 用 actor 当前策略即时 rollout
- 尽快产出可进入本轮更新窗口的样本

它的意义不是“角色更纯”，而是“把一部分 rollout 放到 actor 组本地消化”，从而保留 `disco_rl` Case1 的 pipeline 吞吐特征。

### 10.2 rollout_ref 路异步 rollout

rollout_ref 路负责：

- 消耗剩余 prompt batches
- 使用跨组同步后的策略副本做 rollout
- decode 视频并打 VideoAlign 奖励

它仍然是更偏 rollout 的一侧。

### 10.3 driver 侧聚合与窗口调度

driver 负责：

- 统一接收 actor 路和 rollout_ref 路返回的轨迹
- 做 dual-adv 标准化
- 做 best-of-n
- 做拼批与 pipeline window 调度
- 决定哪次 update 只是累计梯度，哪次 update 真的执行 optimizer step

也就是说，Case1 的“中枢”依然在 driver，而不是像 Case3/Case4 那样更多把优势计算收进 worker 内部。

## 11. `generate_sequences_dance_async()` 需要什么语义

Case1 不应再新增一个只服务自己的 rollout RPC，而应继续复用 Case2 已经有的异步 rollout worker 接口：

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

换句话说，Case1 的异步化不应另起一套 rollout 协议；它变化的是调度方式，不是轨迹数据结构。

这里不只是“如果后续希望减少分叉才可以共用”，而是迁移策略本身就应默认共用；区别放在 trainer 侧的 pipeline 调度，而不是放在 rollout 数学本身。

## 12. dual-adv 与 best-of-n 在 Case1 中仍放在 driver 侧

这是 Case1 与 Case3/Case4 的又一个本质差异。

Case3/Case4 当前更适合把：

- advantage 标准化
- best-of-n
- PPO loss 组织

收进 `_update_actor_dance()`。

但 Case1 若要对齐 `disco_rl` 的主语义，应继续保持：

- rollout worker 只负责产出轨迹与 reward
- driver 侧负责 dual-adv 标准化与 best-of-n
- actor worker 只负责消费“已经筛好”的样本做更新

因此，Case1 更适合直接复用 Case2 已有的异步 update 入口：

```text
update_actor_dance_async(samples)
```

同时继续通过 `samples.meta_info["step_weight"]` 表达 delayed step。worker 内部依然可以复用 `_update_actor_dance()` 作为真正的更新核心，但调用契约应保持 Case2 这套分工：driver 先整理好 advantage / best-of-n / step_weight，再把样本交给 worker。这样就不会把“接口复用”误写成“必须另起一套 worker 更新实现”。  
也就是说，Case1 与 Case2 应共用同一套 async update RPC，只在 `step_weight` 和窗口调度上表现出不同节奏。

## 13. `step_weight` 的迁移原则与 gradient accumulation 的后续演进

Case1 的 pipelined 更新需要保留源链路里最重要的一个调度信号：

- driver 侧决定什么时候只是累计梯度
- 什么时候真正 `optimizer.step()`

因此传给 `update_actor_dance_async(samples)` 的 `samples.meta_info` 里应明确带上：

- `step_weight=false`
- `step_weight=true`

这样的元信息，让 actor worker 能区分：

- 这次只 `backward`
- 还是要执行 `clip_grad -> optimizer.step -> lr_scheduler.step -> zero_grad`

但这里同样要区分“迁移基线”和“后续优化”。

### 13.1 第一阶段：先忠实迁移 `step_weight` 协议

第一版迁移里，更稳妥的目标应是：

- 保留 `disco_rl` Case1 现有的 `step_weight` 调度接口
- 保留它在 trainer 侧表达“哪些 micro-batch 负责 step”的方式
- 先让 Case1 在 long-rl 中以尽量接近源实现的训练行为跑通

即使源实现里存在这样的已知问题：

- `update_actor_asyn()` 开头会先 `optimizer.zero_grad()`
- 这样跨多次异步调用并不会真正累计梯度

这个问题也不应在“Case1 迁移完成”的定义里被默认顺手修掉。否则迁移产物的训练行为就不再是严格意义上的源 Case1 语义。

### 13.2 第二阶段：再把假 accumulation 修成真 accumulation

在 parity 版迁移稳定后，可以再单独立一个优化目标：

- 让 `step_weight` 从“只是一种 step 调度信号”
- 演进成“真正有效的跨调用 gradient accumulation 协议”

到了这个阶段，才适合明确修改：

- `zero_grad` 的调用时机
- 多次异步 `backward` 之间的梯度保留方式
- `optimizer.step()` 与 `lr_scheduler.step()` 的窗口边界

这样改动的价值是很高的，但它应被视为**Case1 迁移后的第二阶段优化**，而不是第一版预案里的默认行为。

## 14. Case1 与 Case2/Case3/Case4 的本质差异

如果把四者放在一起看，理想成功态下它们的关系应非常清晰：

### 14.1 与 Case2 的关系

- 两者都是 disaggregate 双 WorkerGroup
- 两者都使用 latent dataloader 三元组协议
- 两者都需要 actor -> rollout_ref 同步能力
- 两者都走 actor/rollout_ref 双源异步 rollout + driver 聚合
- 两者都应复用同一条 async dual-rollout trainer / rollout RPC / update RPC 骨架
- 但 Case2 是 `schedule="plain_async"`
- Case1 还额外要求 `schedule="pipelined_micro_batch"` 与 `step_weight` 控制的 delayed step

### 14.2 与 Case3 的关系

- 两者都是 disaggregate 双 WorkerGroup
- 两者都使用 latent dataloader 三元组协议
- 两者都需要 actor -> rollout_ref 同步能力
- 但 Case3 是同步 rollout_ref -> actor 链路
- Case1 是异步 actor/rollout_ref 双源 rollout -> driver 聚合 -> actor pipelined update

### 14.3 与 Case4 的关系

- 两者都可以复用当前 long-rl 已验证的 Dance worker 内 rollout 数学
- 两者都不应走通用 diffusion reward / advantage 逻辑
- 但 Case4 是 colocated 单组闭环
- Case1 是 disaggregate 异步 pipeline 闭环

所以 Case1 的迁移本质不是“再写一个 Case3”，也不是“把 Case2 再复制一份”，而是把已经在 Case2/Case3/Case4 中验证过的数据协议和 rollout/update 核心，放进 **Case2 已经落好的 async dual-rollout trainer 骨架** 里，再补上 pipelined trainer 调度策略。

## 15. 需要落地修改的代码面

如果按这个目标态实施，核心改动面应集中在这些位置：

- `verl/workers/config/actor.py`
  - 增加 `dance_case1_mode`
- `verl/trainer/config/actor/actor.yaml`
  - 暴露 `dance_case1_mode: false`
- `verl/trainer/main_ppo.py`
  - 把 Case1 纳入 tokenizer / reward / dataset 的跳过逻辑
- `verl/trainer/ppo/ray_trainer.py`
  - 复用 Case2 已有的 mode 校验、latent dataloader、prompt batching、async dual-rollout fit 骨架
  - 只为 Case1 补 `schedule="pipelined_micro_batch"` 所需的窗口调度和元信息
- `verl/workers/fsdp_workers.py`
  - 复用 Case2 已有的 dual-rollout role-aware init 分支
  - 继续复用 Case2 已有的 actor/rollout_ref 共用异步 rollout 接口
  - 继续复用 Case2 已有的 async update 接口，只让 `step_weight` 在 Case1 中承担 pipelined 调度语义
- 第二阶段优化项
  - 评估是否把 actor 路 rollout 收敛到共享 `self.transformer`
  - 评估是否把 `step_weight` 修成真正的 gradient accumulation 语义
- `examples/diffusion/config_video_diffusion_case1_dance.yaml`
  - 新增 Case1 专用配置
- `case1_dance.sh`
  - 新增 Case1 专用入口
- `tests/trainer/ppo/`
  - 补一组 Case1 CPU 单测，至少覆盖：
    - dataloader 三元组协议
    - pipelined 主循环命中
    - `step_weight` 的累计/step 语义
    - actor/rollout_ref 双源异步 rollout 的接口契约

## 16. 一句话总结

Case1 的成功迁移，本质上就是把 `disco_rl` 里“异步双源 rollout + driver 聚合 + pipelined micro-batch 更新”的训练语义，接到 Case2 已经迁好的共用 async dual-rollout 骨架上：第一阶段先尽量忠实迁移源 Case1 的调度协议，复用 Case2/Case3/Case4 已经成功的 latent 数据协议、共用 rollout/update RPC、专用 YAML/脚本和 fail-fast 配置校验；第二阶段再评估是否把 actor 路 rollout 与 `step_weight` 累计语义进一步收敛成更统一、更干净的 long-rl 实现。
