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

import pytest
import torch

from verl import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator, compute_grpo_outcome_advantage
from verl.trainer.ppo.ray_trainer import compute_advantage_diffusion


def _build_diffusion_batch(token_level_rewards, vq_rewards=None, mq_rewards=None, meta_info=None):
    tensors = {"token_level_rewards": torch.tensor(token_level_rewards, dtype=torch.float32).reshape(-1, 1)}
    if vq_rewards is not None:
        tensors["vq_rewards"] = torch.tensor(vq_rewards, dtype=torch.float32)
    if mq_rewards is not None:
        tensors["mq_rewards"] = torch.tensor(mq_rewards, dtype=torch.float32)
    return DataProto.from_dict(tensors=tensors, meta_info=meta_info or {})


def _normalize_group(values: torch.Tensor, group_size: int) -> torch.Tensor:
    out = torch.empty_like(values)
    for start in range(0, values.numel(), group_size):
        group = values[start : start + group_size]
        out[start : start + group_size] = (group - group.mean()) / (group.std() + 1e-8)
    return out


def test_dual_advantage_group_normalization():
    batch = _build_diffusion_batch(
        token_level_rewards=[0.5, 1.5, 2.5, 3.5],
        vq_rewards=[1.0, 3.0, 2.0, 6.0],
        mq_rewards=[2.0, 4.0, 1.0, 5.0],
    )
    config = {
        "dual_reward_missing_strategy": "error",
        "dual_reward_fill_value": 0.0,
        "dual_adv_mode_default": "group",
    }

    out = compute_advantage_diffusion(
        batch,
        adv_estimator=AdvantageEstimator.GRPO,
        config=config,
        diffusion_algo="dancegrpo",
        num_generations=2,
    )

    expected_vq = _normalize_group(torch.tensor([1.0, 3.0, 2.0, 6.0]), group_size=2)
    expected_mq = _normalize_group(torch.tensor([2.0, 4.0, 1.0, 5.0]), group_size=2)
    assert torch.allclose(out.batch["vq_advantages"][:, 0], expected_vq, atol=1e-5)
    assert torch.allclose(out.batch["mq_advantages"][:, 0], expected_mq, atol=1e-5)


def test_dual_advantage_batch_normalization_requires_explicit_marker():
    batch = _build_diffusion_batch(
        token_level_rewards=[0.0, 0.0, 0.0, 0.0],
        vq_rewards=[0.0, 1.0, 2.0, 3.0],
        mq_rewards=[3.0, 2.0, 1.0, 0.0],
        meta_info={"dual_adv_mode": "batch", "dual_adv_single_group": True},
    )
    config = {
        "dual_reward_missing_strategy": "error",
        "dual_reward_fill_value": 0.0,
        "dual_adv_mode_default": "group",
    }

    out = compute_advantage_diffusion(
        batch,
        adv_estimator=AdvantageEstimator.GRPO,
        config=config,
        diffusion_algo="dancegrpo",
        num_generations=2,
    )

    expected_vq = (torch.tensor([0.0, 1.0, 2.0, 3.0]) - 1.5) / (torch.tensor([0.0, 1.0, 2.0, 3.0]).std() + 1e-8)
    expected_mq = (torch.tensor([3.0, 2.0, 1.0, 0.0]) - 1.5) / (torch.tensor([3.0, 2.0, 1.0, 0.0]).std() + 1e-8)
    assert torch.allclose(out.batch["vq_advantages"][:, 0], expected_vq, atol=1e-5)
    assert torch.allclose(out.batch["mq_advantages"][:, 0], expected_mq, atol=1e-5)


def test_skip_strategy_skips_invalid_samples_in_batch_mode():
    batch = _build_diffusion_batch(
        token_level_rewards=[0.0, 0.0, 0.0, 0.0],
        vq_rewards=[0.0, 1.0, float("nan"), 3.0],
        mq_rewards=[3.0, 2.0, 1.0, 0.0],
        meta_info={"dual_adv_mode": "batch", "dual_adv_single_group": True},
    )
    config = {
        "dual_reward_missing_strategy": "skip",
        "dual_reward_fill_value": 0.0,
        "dual_adv_mode_default": "group",
    }

    out = compute_advantage_diffusion(
        batch,
        adv_estimator=AdvantageEstimator.GRPO,
        config=config,
        diffusion_algo="dancegrpo",
        num_generations=2,
    )

    valid_mask = out.batch["dual_reward_valid_mask"].reshape(-1)
    effective_mask = out.batch["dual_reward_effective_mask"].reshape(-1)
    assert torch.equal(valid_mask, torch.tensor([True, True, False, True]))
    assert torch.equal(effective_mask, torch.tensor([True, True, False, True]))

    filtered_vq = torch.tensor([0.0, 1.0, 3.0], dtype=torch.float32).reshape(-1, 1)
    filtered_mq = torch.tensor([3.0, 2.0, 0.0], dtype=torch.float32).reshape(-1, 1)
    response_mask = torch.ones((3, 1), dtype=torch.bool)
    index = torch.zeros((3,), dtype=torch.long)
    expected_vq, _ = compute_grpo_outcome_advantage(filtered_vq, response_mask, index, epsilon=1e-8)
    expected_mq, _ = compute_grpo_outcome_advantage(filtered_mq, response_mask, index, epsilon=1e-8)

    full_expected_vq = torch.zeros((4, 1), dtype=torch.float32)
    full_expected_mq = torch.zeros((4, 1), dtype=torch.float32)
    full_expected_vq[effective_mask] = expected_vq
    full_expected_mq[effective_mask] = expected_mq

    assert torch.allclose(out.batch["vq_advantages"], full_expected_vq, atol=1e-5)
    assert torch.allclose(out.batch["mq_advantages"], full_expected_mq, atol=1e-5)


def test_single_reward_flow_grpo_path_is_unchanged():
    batch = _build_diffusion_batch(token_level_rewards=[1.0, 2.0, 3.0, 4.0])
    out = compute_advantage_diffusion(
        batch,
        adv_estimator=AdvantageEstimator.GRPO,
        diffusion_algo="flow_grpo",
        num_generations=2,
    )

    response_mask = torch.ones((4, 1), dtype=torch.bool)
    index = torch.zeros((4,), dtype=torch.long)
    expected_adv, expected_ret = compute_grpo_outcome_advantage(batch.batch["token_level_rewards"], response_mask, index)
    assert torch.allclose(out.batch["advantages"], expected_adv)
    assert torch.allclose(out.batch["returns"], expected_ret)
    assert "vq_advantages" not in out.batch.keys()
    assert "mq_advantages" not in out.batch.keys()


def test_group_mode_requires_batch_divisible_by_num_generations():
    batch = _build_diffusion_batch(
        token_level_rewards=[0.1, 0.2, 0.3],
        vq_rewards=[1.0, 2.0, 3.0],
        mq_rewards=[1.0, 2.0, 3.0],
    )
    config = {
        "dual_reward_missing_strategy": "error",
        "dual_reward_fill_value": 0.0,
        "dual_adv_mode_default": "group",
    }

    with pytest.raises(ValueError, match="divisible"):
        compute_advantage_diffusion(
            batch,
            adv_estimator=AdvantageEstimator.GRPO,
            config=config,
            diffusion_algo="dancegrpo",
            num_generations=2,
        )


def test_batch_mode_requires_explicit_single_group_marker():
    batch = _build_diffusion_batch(
        token_level_rewards=[0.1, 0.2, 0.3, 0.4],
        vq_rewards=[1.0, 2.0, 3.0, 4.0],
        mq_rewards=[4.0, 3.0, 2.0, 1.0],
        meta_info={"dual_adv_mode": "batch"},
    )
    config = {
        "dual_reward_missing_strategy": "error",
        "dual_reward_fill_value": 0.0,
        "dual_adv_mode_default": "group",
    }

    with pytest.raises(ValueError, match="single-group marker"):
        compute_advantage_diffusion(
            batch,
            adv_estimator=AdvantageEstimator.GRPO,
            config=config,
            diffusion_algo="dancegrpo",
            num_generations=2,
        )
