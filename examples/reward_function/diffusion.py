# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List
import torch
import piq

# ------------------------------------------------------------
_device = "cuda" if torch.cuda.is_available() else "cpu"

def _to_01(x: torch.Tensor) -> torch.Tensor:
    """
    归一化到 [0,1]，形状支持:
      (C,H,W)  or  (B,C,H,W)
    """
    if x.ndim == 3:                       # (C,H,W) → (1,C,H,W)
        x = x.unsqueeze(0)

    if x.shape[1] == 1:                   # 灰度 → 3 通道
        x = x.repeat(1, 3, 1, 1)

    x = x.float()
    if x.max() > 1.1 or x.min() < -0.1:   # 0‑255 或 −1~1
        x = (x + 1) / 2 if x.min() < 0 else x / 255.
    return x.clamp(0, 1)

def _brisque_batch(batch: torch.Tensor) -> float:
    """batch: (B,3,H,W) → 平均 BRISQUE 原始分 (越低越好)"""
    batch = _to_01(batch).to(_device)
    with torch.no_grad():
        return piq.brisque(batch, data_range=1.0).mean().item()

def _normalize(raw: float, lo: float = 0.0, hi: float = 100.0) -> float:
    """0‑100 → 0‑1，翻转方向：值越高代表画质越好"""
    return max(0.0, min(1.0, 1.0 - (raw - lo) / (hi - lo)))
# ------------------------------------------------------------

def compute_score(
    reward_inputs: List[Dict[str, Any]],
    format_weight: float = 0.0,
) -> List[Dict[str, float]]:
    """
    reward_inputs:
        [{"images": Tensor(3,H,W)},                 # 单张图片
         {"videos": Tensor(F,3,H,W)}, ...]         # 单段视频 (F,3,H,W)
    返回:
        [{"overall": v, "format": 0.0, "accuracy": v}, ...]
        accuracy / overall 已归一化到 0‑1，越高越好
    """
    results: List[Dict[str, float]] = []

    for itm in reward_inputs:
        # ---------- 单图 ----------
        if "images" in itm and isinstance(itm["images"], torch.Tensor):
            # print("*** images ***", itm["images"].shape)
            img = itm["images"]                        # (3,H,W)
            raw = _brisque_batch(img.unsqueeze(0))

        # ---------- 单段视频 ----------
        elif "videos" in itm and isinstance(itm["videos"], torch.Tensor):
            # print("*** videos ***", itm["videos"].shape)
            vid = itm["videos"]                        # (F,3,H,W)
            F, C, H, W = vid.shape
            n_sample = max(1, F // 8)                  # 随机 F/8 帧
            idx = torch.randperm(F, device=vid.device)[:n_sample]
            sampled = vid.index_select(0, idx)         # (n,3,H,W)
            raw = _brisque_batch(sampled)

        else:
            raise ValueError('reward_input 必须含 "images" 或 "videos" 张量')

        acc = _normalize(raw)          # 0‑1, 越高越好
        fmt = 0.0                      # 若需排版分，可替换
        overall = (1 - format_weight) * acc + format_weight * fmt

        results.append({
            "overall": overall,
            "format": fmt,
            "accuracy": acc,
        })

    return results
