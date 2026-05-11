# Flux 模型迁移指导文档（仅 dance 逻辑）

本文档用于把 DIscoRL 仓库 `flux is aligned` commit 中的 **Flux + DanceGRPO** 支持迁移到 long-rl。第一阶段只迁移 **dance variant**，不迁移、不保留、不兜底 `flow` 相关逻辑。

核心目标：在 long-rl 中新增一条独立的 `flux dance` 训练栈，沿用现有 `case1 / case2 / case3 / case4` 拆分方式，并保证现有 Hunyuan dance 栈不受影响。

---

## 1. 范围收紧

### 1.1 只支持 dance

第一阶段只迁移以下 DIscoRL 文件 / 配置：

| 类型 | DIscoRL 来源 | long-rl 目标 |
|---|---|---|
| trainer | `verl/trainer/ray_trainer_dance.py` | `verl/trainer/ray_trainer_flux_dance.py` |
| worker | `verl/workers/fsdp_workers_dance.py` | `verl/workers/fsdp_workers_flux_dance.py` |
| entry | `verl/trainer/main.py` 中 dance 分支 | `verl/trainer/main_flux.py`，只保留 dance |
| config | `examples/config_flux_dance.yaml` / `examples/config_flux_final.yaml` / `examples/config_flux.yaml` 的 dance 用法 | `examples/diffusion/config_video_diffusion_case{1..4}_flux.yaml` |
| sh | `examples/new_supports/flux_grpo_nodes.sh` / `flux_grpo_nodes2.sh` | `case{1..4}_flux.sh` |

以下内容本次不迁移：

- `verl/trainer/ray_trainer_flow.py`
- `verl/workers/fsdp_workers_flow.py`
- `verl/trainer/main_flow.py`
- `examples/config_flux_flow.yaml`
- `examples/new_supports/flux_grpo_nodes_flow.sh`
- 任何 flow 相关分支；同时本次迁移**完全不引入** `trainer.grpo_variant` 字段（long-rl 不使用这个 DIscoRL 用来在 dance / flow 之间分流的标志）。

### 1.2 entry 约束

`main_flux.py` 应写成 Flux 专用、Dance 专用入口，**沿用 long-rl `verl/trainer/main_ppo.py` 的框架风格**：Hydra 装饰器 + `@ray.remote` TaskRunner + Hydra `--config-path / --config-name` 加载 YAML，不采用 DIscoRL `main.py` 的 `OmegaConf.from_cli() + config=xxx.yaml` 风格。

搬运 DIscoRL `main.py` 的 dance 分支时，需要做**最小必要的架构适配**（见 §5 第 6 条）：

1. 把 OmegaConf CLI 解析替换为 `@hydra.main(config_path=..., config_name=...)`；
2. 删除所有 `trainer.grpo_variant` 相关分支与校验，不再引入该字段；
3. 删除 flow 分支与 flow 相关 import；
4. 对外只导出 Flux+Dance 这一条路径。

推荐形态（类名保持与 DIscoRL 一致，不加 `Flux` 前缀）：

```python
import hydra
import ray
from verl.trainer.ray_trainer_flux_dance import RayPPOTrainerDance, ResourcePoolManager, Role
from verl.workers.fsdp_workers_flux_dance import FSDPWorkerDance

trainer_cls = RayPPOTrainerDance
worker_cls = FSDPWorkerDance

@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_flux(config)
```

`run_flux` 内部组织 `role_worker_mapping` / `resource_pool_spec` 的逻辑直接搬运 DIscoRL `main.py` 的 dance 分支即可。

---

## 2. long-rl 现状

long-rl 当前已经有 Hunyuan dance 的 4 case 拆分：

| case | `trainer.disaggregate` | `trainer.pipelined_micro_batch` |
|---|---|---|
| case1 | true | true |
| case2 | true | false |
| case3 | true | false |
| case4 | false | false |

相关文件：

- [config_video_diffusion_case1_dance.yaml](file:///Users/bytedance/codegfile/long-rl/examples/diffusion/config_video_diffusion_case1_dance.yaml) ... [config_video_diffusion_case4_dance.yaml](file:///Users/bytedance/codegfile/long-rl/examples/diffusion/config_video_diffusion_case4_dance.yaml)
- [case1_dance.sh](file:///Users/bytedance/codegfile/long-rl/case1_dance.sh) ... [case4_dance.sh](file:///Users/bytedance/codegfile/long-rl/case4_dance.sh)
- NPU 脚本已使用 `/home/qzy/models` 作为实际测试机模型与数据根目录，例如 [case4_dance_npu.sh](file:///Users/bytedance/codegfile/long-rl/case4_dance_npu.sh)

fastvideo 层已有 Flux 相关文件，本次迁移原则上不改：

| 文件 | 备注 |
|---|---|
| [latent_flux_rl_datasets.py](file:///Users/bytedance/codegfile/long-rl/fastvideo/dataset/latent_flux_rl_datasets.py) | Flux latent 数据集 |
| [pipeline_flux.py](file:///Users/bytedance/codegfile/long-rl/fastvideo/models/flux_hf/pipeline_flux.py) | Flux pipeline |
| [train_grpo_flux.py](file:///Users/bytedance/codegfile/long-rl/fastvideo/train_grpo_flux.py) | 旧式 Flux GRPO 脚本，可作为逻辑参考 |
| [communications_flux.py](file:///Users/bytedance/codegfile/long-rl/fastvideo/utils/communications_flux.py) | Flux 通信工具 |

---

## 3. 权重路径约定

`/home/qzy/models` 是实际 NPU 测试环境路径，不是当前开发机路径。迁移时不要把 DIscoRL 的 `/share/models/dancegrpo/...` 或 `/workspace/...` 直接带入 long-rl。

### 3.1 推荐环境变量

Flux 启动脚本统一显式定义这些路径：

```bash
MODEL_ROOT=/home/qzy/models
FLUX_MODEL_PATH=${MODEL_ROOT}/flux
HPSV2_CKPT_PATH=${MODEL_ROOT}/HPS_v2.1_compressed.pt
OPEN_CLIP_CKPT_PATH=${MODEL_ROOT}/open_clip_pytorch_model.bin
DATA_JSON_PATH=${MODEL_ROOT}/flux/rl_embeddings/videos2caption.json
```

如果实际目录名不是 `flux`，只改 `FLUX_MODEL_PATH` 和 `DATA_JSON_PATH`，其它脚本逻辑不要改。

### 3.2 YAML 路径替换

DIscoRL Flux YAML 中的路径需要替换为测试机路径：

| DIscoRL 字段 | DIscoRL 原值 | long-rl 目标值 |
|---|---|---|
| `data.data_json_path` | `/share/models/dancegrpo/flux/rl_embeddings/videos2caption.json` | `${DATA_JSON_PATH}` 或 `/home/qzy/models/flux/rl_embeddings/videos2caption.json` |
| `worker.actor.model.pretrained_model_name_or_path` | `/share/models/dancegrpo/flux` | `/home/qzy/models/flux` |
| `worker.actor.model.vae_model_path` | `/share/models/dancegrpo/flux` | `/home/qzy/models/flux` |
| `worker.ref.scheduler` | `/workspace/Long-RL1/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/scheduler` | `/home/qzy/models/Wan2.1-T2V-1.3B-Diffusers/scheduler` |

> 字段名按 DIscoRL 原始 YAML 列出；在 long-rl 新增 YAML 里对应落位到 long-rl namespace（见 §5.1）。

### 3.3 HPSv2 / OpenCLIP 路径

DIscoRL / fastvideo Flux 逻辑中 HPSv2 目前有硬编码路径：

```python
create_model_and_transforms(
    "ViT-H-14",
    "/share/models/dancegrpo/hps_ckpt/open_clip_pytorch_model.bin",
    ...
)
cp = "/share/models/dancegrpo/hps_ckpt/HPS_v2.1_compressed.pt"
```

迁移到 `fsdp_workers_flux_dance.py` 时应改成配置字段或环境变量，推荐：

```python
open_clip_ckpt_path = os.environ.get("OPEN_CLIP_CKPT_PATH", "/home/qzy/models/open_clip_pytorch_model.bin")
hpsv2_ckpt_path = os.environ.get("HPSV2_CKPT_PATH", "/home/qzy/models/HPS_v2.1_compressed.pt")
```

启动脚本中同步导出：

```bash
export OPEN_CLIP_CKPT_PATH=${OPEN_CLIP_CKPT_PATH:-/home/qzy/models/open_clip_pytorch_model.bin}
export HPSV2_CKPT_PATH=${HPSV2_CKPT_PATH:-/home/qzy/models/HPS_v2.1_compressed.pt}
```

---

## 4. 新增文件清单

### 4.1 训练栈

| 新增文件 | 来源 | 备注 |
|---|---|---|
| `verl/trainer/ray_trainer_flux_dance.py` | DIscoRL `verl/trainer/ray_trainer_dance.py` | 不引入 flow |
| `verl/workers/fsdp_workers_flux_dance.py` | DIscoRL `verl/workers/fsdp_workers_dance.py` | 不引入 flow |
| `verl/trainer/main_flux.py` | DIscoRL `verl/trainer/main.py` 的 dance 部分 | 沿用 long-rl `main_ppo.py` 风格，只 import dance trainer / worker |

> 注：不新增 `verl/trainer/config_flux.py`。将 DIscoRL 的 Flux/Dance schema **直接追加到 long-rl 现有的 `verl/trainer/config.py`（或 `verl/trainer/config/ppo_trainer.yaml` 对应结构）中**，以避免维护两套 config 加载链。具体范围见 §6 Step 1。

### 4.2 配置

**case 分发规则以 long-rl 为准，不是 DIscoRL**。long-rl 已经**显式废弃** DIscoRL 的 `worker.actor.disco` 字段——在 [verl/trainer/ppo/ray_trainer.py](file:///Users/bytedance/codegfile/long-rl/verl/trainer/ppo/ray_trainer.py#L939-L947) 的 `_validate_config` 里遇到 `actor_rollout_ref.actor.disco` 直接 `ValueError`。因此：

- **禁止在任何 Flux YAML / shell / worker 代码里引入 `disco` 字段**（包括不得新增、不得沿用 DIscoRL 的 `worker.actor.disco`）。
- long-rl 的 case 路径分发靠**硬标志** `actor_rollout_ref.actor.dance_case{1,2,3,4}_mode`（由 `main_ppo.py` 的 `_is_dance_caseN_enabled` 读取），配合 `trainer.disaggregate` + `trainer.pipelined_micro_batch` 两个开关决定行为。
- Flux 新增的 4 份 YAML **必须严格对齐**同号 `config_video_diffusion_case{1..4}_dance.yaml` 的开关组合：

| case | `dance_caseN_mode` | `trainer.disaggregate` | `trainer.pipelined_micro_batch` |
|---|---|---|---|
| case1 | `dance_case1_mode: true` | true | true |
| case2 | `dance_case2_mode: true` | true | false |
| case3 | `dance_case3_mode: true` | true | false |
| case4 | `dance_case4_mode: true` | false | false |

> case2 与 case3 的三开关**完全相同**，两者的差异不在开关层，而在 trainer 主循环选择与其它训练参数（`gradient_accumulation_steps`、`num_generations`、`bestofn`、`timestep_fraction` 等）上。Flux 新增 YAML 的其它参数以 `case{1..4}_dance.yaml` 同号文件为唯一模板，不要跨 case 混搭。

DIscoRL 现有的 4 份 Flux YAML（`config_flux.yaml` / `config_flux_dance.yaml` / `config_flux_final.yaml` / `config_flux2.yaml`）全部是 `disaggregate=true + pipelined_micro_batch=false + disco=true`，**只覆盖 case2/case3 的开关组合**（其中 `disco=true` 在 long-rl 必须删掉），**没有对应 case1 / case4 的现成模板**。

| 新增文件 | 开关组合 | 基座文件 | 覆写要点 |
|---|---|---|---|
| `examples/diffusion/config_video_diffusion_case1_flux.yaml` | `dance_case1_mode=true`, `disaggregate=true`, `pipelined_micro_batch=true` | 以 `config_video_diffusion_case1_dance.yaml` 为骨架（决定 case 分发 & 资源池字段），Flux 训练参数从 `config_flux_dance.yaml` 拿 | 把 Hunyuan 骨架里模型/数据路径、`extra.dance.*` 下 Flux 相关字段替换为 Flux 值；删除 DIscoRL 侧的 `disco` 字段 |
| `examples/diffusion/config_video_diffusion_case2_flux.yaml` | `dance_case2_mode=true`, `disaggregate=true`, `pipelined_micro_batch=false` | `config_video_diffusion_case2_dance.yaml` + `config_flux_dance.yaml` | 同上；删除 DIscoRL 侧的 `disco` 字段 |
| `examples/diffusion/config_video_diffusion_case3_flux.yaml` | `dance_case3_mode=true`, `disaggregate=true`, `pipelined_micro_batch=false`，final 调参 | `config_video_diffusion_case3_dance.yaml` + `config_flux_final.yaml` | final 调参参考：`gradient_accumulation_steps=8`、`timestep_fraction=1`、`num_generations=8`、`bestofn=8`；删除 DIscoRL 侧的 `disco` 字段 |
| `examples/diffusion/config_video_diffusion_case4_flux.yaml` | `dance_case4_mode=true`, `disaggregate=false`, `pipelined_micro_batch=false` | `config_video_diffusion_case4_dance.yaml` + `config_flux_dance.yaml` | 同上；删除 DIscoRL 侧的 `disco` 字段 |

即：**case 骨架（YAML 顶层开关 + `dance_caseN_mode`）沿用 long-rl 同号 Hunyuan dance YAML；Flux 特有的模型/数据/采样参数从 DIscoRL Flux YAML 中抽取后塞进 `actor_rollout_ref.actor.extra.dance.*` 与 `actor_rollout_ref.rollout.*`**，不得将 DIscoRL YAML 的顶层结构（`worker.actor.*` / `worker.rollout.*` / `worker.ref.*` / `worker.actor.disco`）整块照抄。

#### 4.2.1 `disco` 语义在 long-rl 的承接

long-rl 删除 `disco` 字段后，DIscoRL 原 `disco` / `pipelined_micro_batch` / `disaggregate` 三字段隐式控制的调度路径，被重构为**显式 case 入口 + 专用 fit 函数**，分发位于 [ray_trainer.py#L623-L656](file:///Users/bytedance/codegfile/long-rl/verl/trainer/ppo/ray_trainer.py#L623-L656) 和 [ray_trainer.py#L3479](file:///Users/bytedance/codegfile/long-rl/verl/trainer/ppo/ray_trainer.py#L3479)：

| DIscoRL 字段组合 | 原语义 | long-rl 承接 case | long-rl fit 入口 |
|---|---|---|---|
| `pipelined_micro_batch=True` | micro-batch 流水线 | case1 | `fit_dance_dual_rollout_dis_async(schedule="pipelined_micro_batch")` |
| `disco=True` | actor/rollout 异步调度 | case2 | `fit_dance_dual_rollout_dis_async(schedule="plain_async")` |
| `disco=True` + final 调参 | 分离式 final 路径 | case3 | `fit_dance_case3_dis()` |
| `disaggregate=False` | 单资源池同步 | case4 | `fit_dance_case4()` |

结论：Flux 文件只要正确设置 `dance_caseN_mode` 并复用 long-rl 的 `fit_dis` / `fit` 分发，**不必在 Flux 侧再搬运任何 `disco` 分支代码**，原 `disco=True` 的异步调度由 long-rl 的 `schedule="plain_async"` 路径直接承担。

### 4.3 启动脚本

| 新增脚本 | 对应 YAML | 模板 |
|---|---|---|
| `case1_flux.sh` | `config_video_diffusion_case1_flux.yaml` | 同号 `case1_dance.sh` / NPU 可参考 `case1_dance_nodes_npu.sh` |
| `case2_flux.sh` | `config_video_diffusion_case2_flux.yaml` | 同号 `case2_dance.sh` / NPU 可参考 `case2_dance_nodes_npu.sh` |
| `case3_flux.sh` | `config_video_diffusion_case3_flux.yaml` | 同号 `case3_dance.sh` / NPU 可参考 `case3_dance_npu.sh` |
| `case4_flux.sh` | `config_video_diffusion_case4_flux.yaml` | 同号 `case4_dance.sh` / NPU 可参考 `case4_dance_npu.sh` |

关键替换（沿用 long-rl Hydra 风格的 CLI，而非 DIscoRL 的 `config=` 风格）：

```bash
python3 -m verl.trainer.main_ppo -> python3 -m verl.trainer.main_flux
--config-path=/workspace/projects/long-rl/examples/diffusion   # 保持不变
--config-name=config_video_diffusion_caseN_dance -> --config-name=config_video_diffusion_caseN_flux
MODEL_PATH=/home/qzy/models/HunyuanVideo -> FLUX_MODEL_PATH=/home/qzy/models/flux
```

CLI 中的 `actor_rollout_ref.actor.extra.dance.*` 等 override key **保留 long-rl 既有的命名空间**，不要替换为 DIscoRL 的 `worker.actor.*`。所有 DIscoRL 字段与 long-rl 字段的对应关系见 §5.1 参数对照表。

---

## 5. 允许的最小改动

搬运 DIscoRL dance 文件时只允许以下改动：

1. import 重定向到新增 Flux 文件，例如 `ray_trainer_dance` -> `ray_trainer_flux_dance`。
2. entry 删除 flow 分支、删除 `trainer.grpo_variant` 相关字段和校验；**本次迁移不保留、不读取、不新增 `grpo_variant` 字段**。
3. 绝对路径替换为 `/home/qzy/models` 相关路径或环境变量。
4. HPSv2 / OpenCLIP 的硬编码路径改为 `HPSV2_CKPT_PATH` / `OPEN_CLIP_CKPT_PATH`。
5. **类名保持与 DIscoRL 一致，不加 `Flux` 前缀**（即 `RayPPOTrainerDance` / `FSDPWorkerDance` / `ResourcePoolManager` / `Role` 等原样保留），只改模块路径。
6. **允许 long-rl 与 DIscoRL 架构兼容层面的重构**（例如把 OmegaConf CLI 替换为 Hydra 装饰器、把 DIscoRL 的 `worker.actor.*` / `trainer.*` 配置访问改写为 long-rl 的 `actor_rollout_ref.actor.extra.dance.*` / `trainer.*` 访问路径、Ray runtime env 组装方式对齐 `verl.trainer.constants_ppo.get_ppo_ray_runtime_env`）。**算法逻辑、训练循环、FSDP/Rollout 细节一律不允许重构**。

架构兼容层的改写请严格按 §5.1 的参数对照表来做，不要做对照表之外的"顺手优化"。

### 5.1 DIscoRL ↔ long-rl 参数对照表

框架差异主要体现在 config namespace。**迁移统一沿用 long-rl 的 key 命名**（Hunyuan dance 的历史参数名），DIscoRL 侧的 key 在 `ray_trainer_flux_dance.py` / `fsdp_workers_flux_dance.py` 中对访问代码做改写即可。

| 语义 | DIscoRL key | long-rl key |
|---|---|---|
| 训练入口启动模块 | `verl.trainer.main` | `verl.trainer.main_flux` |
| Actor 优化 batch | `worker.actor.global_batch_size` | `actor_rollout_ref.actor.ppo_mini_batch_size` |
| Actor per-device micro batch（update） | `worker.actor.micro_batch_size_per_device_for_update` | `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` |
| Actor per-device micro batch（experience） | `worker.actor.micro_batch_size_per_device_for_experience` | `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu` |
| Actor 梯度累积 | `worker.actor.gradient_accumulation_steps` | `actor_rollout_ref.actor.gradient_accumulation_steps` |
| Actor timestep fraction | `worker.actor.timestep_fraction` | `actor_rollout_ref.actor.extra.dance.timestep_fraction` |
| Actor master weight | `worker.actor.master_weight_type` | `actor_rollout_ref.actor.extra.dance.master_weight_type` |
| Actor 预训练权重 | `worker.actor.model.pretrained_model_name_or_path` | `actor_rollout_ref.actor.extra.dance.pretrained_model_name_or_path` |
| Actor VAE 权重 | `worker.actor.model.vae_model_path` | `actor_rollout_ref.actor.extra.dance.vae_model_path` |
| Actor 模型类型 | `worker.actor.model.model_type` | `actor_rollout_ref.actor.extra.dance.model_type` |
| disco 开关（DIscoRL 用于异步训练分发） | `worker.actor.disco` | **废弃**，不可引入；long-rl 在 `ray_trainer.py` 会对该字段 fail-fast。相应语义由 `dance_caseN_mode` 硬标志承载 |
| Rollout 采样步数 | `worker.rollout.sampling_steps` / `num_steps` | `actor_rollout_ref.rollout.sampling_steps` / `num_steps` |
| Rollout 组数 / bestofn | `worker.rollout.num_generations` / `bestofn` | `actor_rollout_ref.rollout.num_generations` / `bestofn` |
| Rollout 分辨率 | `worker.rollout.height` / `width` / `num_frames` | `actor_rollout_ref.rollout.height` / `width` / `num_frames` |
| Ref scheduler 路径 | `worker.ref.scheduler` | `actor_rollout_ref.ref.scheduler`（或归入 `extra.dance`，与现有 Hunyuan 字段对齐） |
| Reward 配置 | `worker.reward.*` | long-rl 中由 `reward_function` 和 `reward_model.enable` 组合承载；对照 `case*_dance.sh` 搬运 |
| 总节点数 | `trainer.nnodes` | `trainer.nnodes` |
| 每节点 GPU | `trainer.n_gpus_per_node` | `trainer.n_gpus_per_node` |
| 分段节点（fast） | `trainer.nnodes_fast` | 用 `trainer.disaggregate_rollout_ref_n_gpus_per_node` 等价表达 |
| 分段节点（slow） | `trainer.nnodes_slow` | 用 `trainer.disaggregate_actor_n_gpus_per_node` 等价表达 |
| disaggregate 主开关 | `trainer.disaggregate` | `trainer.disaggregate` |
| pipelined micro batch | `trainer.pipelined_micro_batch` | `trainer.pipelined_micro_batch` |
| Dance case 分发（long-rl 独有） | — | `actor_rollout_ref.actor.dance_case{1,2,3,4}_mode`（同一时间仅允许一个为 true；由 [main_ppo.py](file:///Users/bytedance/codegfile/long-rl/verl/trainer/main_ppo.py#L43-L60) 的 `_is_dance_caseN_enabled` 读取） |
| Max train steps | `trainer.max_train_steps` | `trainer.max_train_steps` |
| Data JSON | `data.data_json_path` | `data.data_json_path` |

> 迁移前请先根据这张表把 `ray_trainer_flux_dance.py` / `fsdp_workers_flux_dance.py` 内部的 config 访问语句改写到 long-rl 的 namespace；改完之后再开始 YAML / 脚本搬运，避免两边字段名错位。

### 5.2 `disco` 字段的处置

DIscoRL 的 `ray_trainer_dance.py` / `fsdp_workers_dance.py` / `main.py` 里大量出现 `worker.actor.disco` 读取与分支：

- [ray_trainer_dance.py#L160](file:///Users/bytedance/codegfile/DIscoRL/verl/trainer/ray_trainer_dance.py#L160) `self.disco = config.worker.actor.disco`
- [ray_trainer_dance.py#L660](file:///Users/bytedance/codegfile/DIscoRL/verl/trainer/ray_trainer_dance.py#L660) `if self.disco:`
- [fsdp_workers_dance.py#L1250](file:///Users/bytedance/codegfile/DIscoRL/verl/workers/fsdp_workers_dance.py#L1250) `if self.config.worker.actor.disco == True:`
- [main.py#L92-L95](file:///Users/bytedance/codegfile/DIscoRL/verl/trainer/main.py#L92-L95) `disco` 与 `pipelined_micro_batch` 的互斥断言
- [main.py#L124-L130](file:///Users/bytedance/codegfile/DIscoRL/verl/trainer/main.py#L124-L130) 基于 `disco or pipelined_micro_batch` 决定走 `fit_disco_pipelined()` 还是 `fit_dis()`

这些位置在迁移到 Flux 文件时必须**逐一改写**（这属于 §5 第 6 条允许的"架构兼容重构"，不属于算法重构）：

1. 把 `config.worker.actor.disco` 的读取删除；改为读 `config.actor_rollout_ref.actor.dance_case1_mode` / `dance_case2_mode` / `dance_case3_mode` / `dance_case4_mode`。
2. 原 `disco=true` 语义在 long-rl 里归属 **case2**（`dance_case2_mode=true`，开关 `disaggregate=true + pipelined_micro_batch=false`）。
3. 原 `pipelined_micro_batch=true` 语义在 long-rl 里归属 **case1**（`dance_case1_mode=true`）。
4. 原 `disaggregate=false` 语义在 long-rl 里归属 **case4**（`dance_case4_mode=true`）。
5. 原 `disaggregate=true + pipelined_micro_batch=false + disco=false`（DIscoRL Flux YAML 里并未出现，但代码中存在该路径）在 long-rl 里归属 **case3**（`dance_case3_mode=true`）。
6. 原 `disco` 与 `pipelined_micro_batch` 的互斥断言改写为"同一时间 `dance_caseN_mode` 只能一个为 true"的 fail-fast 校验。

所有改写完成后，Flux 文件里不得残留任何 `disco` 字符串（可用 `rg "\bdisco\b" verl/trainer/ray_trainer_flux_dance.py verl/workers/fsdp_workers_flux_dance.py verl/trainer/main_flux.py` 验证应为空）。

---

## 6. 推荐迁移步骤

### Step 1：扩展 long-rl 现有 config schema

- **不新增 `config_flux.py`**。直接在 long-rl 现有的 `verl/trainer/config.py`（或对应 `verl/trainer/config/ppo_trainer.yaml` / `verl/trainer/config/actor/*.yaml` 等 Hydra 子 schema）中，追加 DIscoRL Flux/Dance 额外需要的字段。
- 追加范围以 §5.1 参数对照表为权威清单（例如 `disaggregate`、`pipelined_micro_batch`、`timestep_fraction`、`master_weight_type`、`pretrained_model_name_or_path`、`vae_model_path`、`model_type` 等，在 long-rl 里尚不存在或 schema 不兼容的字段），落位到 long-rl 既有命名空间下（`actor_rollout_ref.actor.extra.dance.*` / `trainer.*`）。**绝不追加 `disco` 字段**。
- 不要 fork 出第二套 config 加载链，保持 `main_flux.py` 和 `main_ppo.py` 读同一套 schema。

### Step 2：新增 dance trainer / worker

- 新增 `verl/trainer/ray_trainer_flux_dance.py`。
- 新增 `verl/workers/fsdp_workers_flux_dance.py`。
- 来源分别是 DIscoRL `ray_trainer_dance.py` 和 `fsdp_workers_dance.py`。
- 只做第 5 节允许的最小改动（含 §5 第 6 条的架构兼容重构）。
- **fastvideo import 校验**：搬完后 grep 文件内所有 `from fastvideo...`，确认 import 的是 long-rl 已有的 `fastvideo.dataset.latent_rl_datasets`、`fastvideo.utils.communications`、`fastvideo.utils.load`、`fastvideo.utils.fsdp_util`、`fastvideo.models.videoalign.inference` 等模块。**不应**出现 `latent_flux_rl_datasets` 或 `communications_flux`——这两个 Flux 专用文件只被旧版单脚本（`train_grpo_flux.py` 等）使用，dance 栈不经过它们。

### Step 3：新增 `main_flux.py`

- 沿用 long-rl `verl/trainer/main_ppo.py` 的 Hydra + Ray TaskRunner 框架风格。
- 只 import `RayPPOTrainerDance`、`FSDPWorkerDance`、`ResourcePoolManager`、`Role`。
- 不 import flow 文件。
- 不实现 flow 分支。
- 不读取、不校验、不新增 `trainer.grpo_variant` 字段。

### Step 4：新增 4 份 Flux YAML

- 严格按 §4.2 case 分发表生成，三个 YAML 字段必须**严格对齐同号** `config_video_diffusion_case{1..4}_dance.yaml`：`dance_caseN_mode` + `trainer.disaggregate` + `trainer.pipelined_micro_batch`。
- YAML 的 schema namespace 用 long-rl 风格（`actor_rollout_ref.actor.extra.dance.*` 等），不要照抄 DIscoRL 的 `worker.actor.*`。
- 从 DIscoRL `config_flux_dance.yaml` / `config_flux_final.yaml` 中抽取的 Flux 专用字段（模型、VAE、采样参数等）塞进 `extra.dance.*` / `rollout.*`。
- 替换模型、数据、HPSv2、OpenCLIP、scheduler 路径到 `/home/qzy/models`。
- 保持 `trainer.model_name: "flux"`；**不写入 `trainer.grpo_variant` 字段**；**不写入 `worker.actor.disco` 或 `actor_rollout_ref.actor.disco` 字段**（已被 long-rl fail-fast 拒绝）。

### Step 5：新增 4 份 Flux 启动脚本

- 以同号 Hunyuan dance 脚本为模板（`case{1..4}_dance.sh`）。
- CLI 入口换为 `python3 -m verl.trainer.main_flux`，`--config-name` 换为 `config_video_diffusion_caseN_flux`，`--config-path` 保持指向 `examples/diffusion`。
- NPU 测试脚本参考已有 `case*_dance*_npu.sh` 中的 `/home/qzy/project/long-rl`、`/home/qzy/models`、Ray runtime env 写法。
- 显式导出 `OPEN_CLIP_CKPT_PATH` 和 `HPSV2_CKPT_PATH`。
- `worker.ref.scheduler`（在 long-rl 对应字段）使用 `/home/qzy/models/Wan2.1-T2V-1.3B-Diffusers/scheduler`。

### Step 6：Smoke 测试

1. Hunyuan dance case1-4 回归：确认现有文件未修改，能进入 train loop。
2. Flux dance case1-4：每个至少跑 2 个 train step。
3. NPU 上额外记录 Flux attention processor 类名、首步显存、首步耗时。
4. 如果 Flux NPU attention 报错，再单独开 attention patch 任务；不要在本次迁移里预先改。

---

## 7. 硬性约束

1. 不修改现有 Hunyuan dance 文件：`case{1..4}_dance*.sh`、`config_video_diffusion_case{1..4}_dance*.yaml`、`verl/trainer/main_ppo.py`、`verl/workers/fsdp_workers.py`。
2. 不迁移 flow：不新增 `ray_trainer_flux_flow.py`、`fsdp_workers_flux_flow.py`、`main_flux_flow.py`。
3. 不改 fastvideo Flux / Hunyuan 模型层。
4. 不保留 DIscoRL `/share/models/...` 绝对路径；全部改为 `/home/qzy/models` 或环境变量。

---

## 8. 验收清单

- [ ] `verl/trainer/config.py`（或对应 Hydra schema 文件）已追加 Flux/Dance 所需字段，未新增 `config_flux.py`。
- [ ] 新增 `verl/trainer/ray_trainer_flux_dance.py`，内部 config 访问已按 §5.1 对照表改写到 long-rl namespace。
- [ ] 新增 `verl/workers/fsdp_workers_flux_dance.py`，同上；`from fastvideo...` import 不含 `latent_flux_rl_datasets` / `communications_flux`。
- [ ] 新增 `verl/trainer/main_flux.py`，Hydra + Ray TaskRunner 风格，只支持 dance，无 `grpo_variant` 相关字段。
- [ ] 类名保留 DIscoRL 原名（`RayPPOTrainerDance` / `FSDPWorkerDance` / `ResourcePoolManager` / `Role`），未加 `Flux` 前缀。
- [ ] 新增 `examples/diffusion/config_video_diffusion_case{1,2,3,4}_flux.yaml`，`dance_caseN_mode` + `trainer.disaggregate` + `trainer.pipelined_micro_batch` 严格对齐同号 Hunyuan dance YAML。
- [ ] 任何 Flux YAML / shell / 新增 py 代码中均**不含** `disco` / `worker.actor.disco` / `actor_rollout_ref.actor.disco` 字段（`rg "\bdisco\b"` 在新增文件上为空）。
- [ ] 新增 `case{1,2,3,4}_flux.sh`，CLI 入口为 `verl.trainer.main_flux`，沿用 Hydra `--config-path/--config-name` 风格。
- [ ] Flux YAML / shell 中模型路径使用 `/home/qzy/models/flux` 或 `FLUX_MODEL_PATH`。
- [ ] HPSv2 使用 `/home/qzy/models/HPS_v2.1_compressed.pt` 或 `HPSV2_CKPT_PATH`。
- [ ] OpenCLIP 使用 `/home/qzy/models/open_clip_pytorch_model.bin` 或 `OPEN_CLIP_CKPT_PATH`。
- [ ] Ref scheduler 路径使用 `/home/qzy/models/Wan2.1-T2V-1.3B-Diffusers/scheduler`。
- [ ] 代码与 YAML 中不存在新增 flow import / flow entry / flow config，也不存在 `grpo_variant` 字段。
- [ ] `git diff` 确认现有 Hunyuan dance 栈没有被修改。
