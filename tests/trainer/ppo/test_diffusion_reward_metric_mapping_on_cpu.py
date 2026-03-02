# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

import numpy as np
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def _build_trainer(diffusion_algo="dancegrpo"):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.diffusion = True
    trainer.diffusion_algo = diffusion_algo
    trainer.config = OmegaConf.create(
        {
            "algorithm": {
                "dual_reward_missing_strategy": "error",
                "dual_reward_fill_value": 0.0,
            }
        }
    )
    return trainer


def _build_diffusion_batch(batch_size=3):
    return DataProto.from_dict(
        tensors={"token_level_scores": torch.zeros((batch_size, 1), dtype=torch.float32)},
    )


def test_fit_maps_vq_mq_metrics_into_batch_rewards():
    trainer = _build_trainer("dancegrpo")
    batch = _build_diffusion_batch(batch_size=3)
    reward_metrics = {
        "VQ": [1.0, 2.0, 3.0],
        "MQ": [4.0, 5.0, 6.0],
    }

    injected = trainer._inject_dual_rewards_from_sources(batch, reward_metrics)

    assert injected is True
    assert torch.allclose(batch.batch["vq_rewards"], torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))
    assert torch.allclose(batch.batch["mq_rewards"], torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32))
    assert torch.equal(batch.batch["dual_reward_valid_mask"], torch.tensor([True, True, True]))


def test_fit_dis_maps_vq_mq_metrics_numpy_values_into_batch_rewards():
    trainer = _build_trainer("dancegrpo")
    batch = _build_diffusion_batch(batch_size=2)
    reward_metrics = {
        "VQ": np.array([7.0, 8.0], dtype=np.float32),
        "MQ": np.array([9.0, 10.0], dtype=np.float32),
    }

    injected = trainer._inject_dual_rewards_from_sources(batch, reward_metrics)

    assert injected is True
    assert torch.allclose(batch.batch["vq_rewards"], torch.tensor([7.0, 8.0], dtype=torch.float32))
    assert torch.allclose(batch.batch["mq_rewards"], torch.tensor([9.0, 10.0], dtype=torch.float32))
    assert torch.equal(batch.batch["dual_reward_valid_mask"], torch.tensor([True, True]))


def test_flow_grpo_path_remains_unchanged_for_metric_injection():
    trainer = _build_trainer("flow_grpo")
    batch = _build_diffusion_batch(batch_size=2)
    reward_metrics = {"VQ": [1.0, 2.0], "MQ": [3.0, 4.0]}

    injected = trainer._inject_dual_rewards_from_sources(batch, reward_metrics)

    assert injected is False
    assert "vq_rewards" not in batch.batch.keys()
    assert "mq_rewards" not in batch.batch.keys()
