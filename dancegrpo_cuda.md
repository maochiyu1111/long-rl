# dancegrpo 分支 CUDA 依赖审计与 NPU 兼容方案

## 1. 审计结论（核心判断）

`disco_rl:sijie/dancegrpo` 的 CUDA 绑定是**系统性**的，不是少量 API 替换问题。  
绑定点覆盖了：

1. 设备与进程组初始化（`torch.cuda` + `backend="nccl"`）
2. AMP 与 dtype 路径（大量 `autocast("cuda")`）
3. 显存/同步 API（`torch.cuda.empty_cache/synchronize/current_device/...`）
4. 通信与并行状态（`nccl_info`、NCCL 环境变量）
5. Attention 算子链（`flash_attn` + ring/varlen CUDA 扩展）
6. 训练脚本与推理组件默认设备（`device="cuda"`、`.to("cuda")`）

因此 NPU 兼容应采用“分层改造”，不能只做字符串替换。

## 2. 建议的代码替换模式（示例）

```python
from verl.utils.device import get_device_name, get_torch_device, get_nccl_backend

device_name = get_device_name()          # "cuda" | "npu" | "cpu"
torch_device = get_torch_device()        # torch.cuda | torch.npu | ...
backend = get_nccl_backend()             # "nccl" | "hccl"

dist.init_process_group(backend=backend, init_method="env://")
torch_device.set_device(local_rank)
device = torch.device(device_name, local_rank)

with torch.autocast(device_type=device_name, dtype=torch.bfloat16):
    ...

torch_device.empty_cache()
torch_device.synchronize()
```

## 7. dancegrpo 对 fastvideo 的依赖面

## 7.1 依赖结论

`dancegrpo` 在 `verl` 主训练链路上对 `fastvideo` 是**强依赖**，不是可有可无：

1. `verl/trainer/main.py` 直接用 `fastvideo.dataset.latent_rl_datasets.LatentDataset` 作为训练数据入口。
2. `verl/workers/fsdp_workers.py` 直接用 `fastvideo.utils.load` 和 `fastvideo.utils.fsdp_util` 做模型构建/FSDP封装。
3. 开启 `use_videoalign` 时，会动态依赖 `fastvideo.models.videoalign.inference.VideoVLMRewardInference`。

## 7.2 `verl` 主链路直接引用的 fastvideo 文件

1. `fastvideo/dataset/latent_rl_datasets.py`
2. `fastvideo/utils/load.py`
3. `fastvideo/utils/fsdp_util.py`
4. `fastvideo/models/videoalign/inference.py`（仅 `use_videoalign=true` 时）
5. `fastvideo/utils/communications.py`（在 `main.py` 有导入但当前未实际调用）

## 7.3 由直接依赖触发的核心传递依赖（训练主链路）

来自 `fastvideo/utils/load.py` 与 `fastvideo/utils/fsdp_util.py` 的核心依赖主要包括：

1. `fastvideo/models/hunyuan/modules/models.py`
2. `fastvideo/models/hunyuan/text_encoder/*`
3. `fastvideo/models/hunyuan/vae/autoencoder_kl_causal_3d.py`
4. `fastvideo/models/hunyuan_hf/modeling_hunyuan.py`
5. `fastvideo/models/mochi_hf/modeling_mochi.py`
6. `fastvideo/models/flash_attn_no_pad.py`
7. `fastvideo/utils/communications.py`
8. `fastvideo/utils/parallel_states.py`
9. `fastvideo/utils/logging_.py`

这部分是 dancegrpo 训练时真正“会走到”的 fastvideo 代码面。

## 7.4 VideoAlign 奖励链路的传递依赖（可选）

当 `use_videoalign=true` 时，还会额外引入：

1. `fastvideo/models/videoalign/data.py`
2. `fastvideo/models/videoalign/utils.py`
3. `fastvideo/models/videoalign/train_reward.py`
4. `fastvideo/models/videoalign/trainer.py`
5. `fastvideo/models/videoalign/prompt_template.py`
6. `fastvideo/models/videoalign/vision_process.py`

这条链路依赖重、CUDA绑定也更明显，建议在 NPU 迁移中作为“第二阶段可选能力”接入。

## 7.5 哪些 fastvideo 文件不是 `verl/main.py + fsdp_workers.py` 必需

以下大类在 dancegrpo 的 `verl` 主入口中不是必经路径：

1. `fastvideo/train_grpo_*.py`（独立训练脚本）
2. `fastvideo/data_preprocess/*.py`（离线预处理脚本）
3. `fastvideo/sample/*.py`（采样演示脚本）
4. `fastvideo/config_sd/*.py`（训练脚本配置）

含义是：迁移到 long-rl 时，不必把整个 fastvideo 全量并入“必需运行时依赖”，可优先保留主链路最小子集。

---

总体上，`dancegrpo` 的 CUDA 依赖是“框架层 + 算子层”的双重绑定。  
NPU 迁移应以“设备/后端抽象统一 + 高风险 CUDA 内核先降级 fallback”为主线推进，这样风险最可控，且与当前 `long-rl` 的 NPU 架构最一致。
