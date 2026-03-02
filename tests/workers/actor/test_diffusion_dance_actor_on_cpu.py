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

import torch
import torch.nn as nn
import unittest

from verl import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.config import FSDPActorConfig, OptimizerConfig


class _DummyDiffusionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))


def _build_actor(timestep_fraction: float = 0.5) -> DataParallelPPOActor:
    actor = object.__new__(DataParallelPPOActor)
    actor.config = FSDPActorConfig(
        strategy="fsdp2",
        ppo_mini_batch_size=4,
        ppo_micro_batch_size_per_gpu=4,
        ppo_epochs=1,
        grad_clip=1.0,
        use_dynamic_bsz=False,
        use_torch_compile=False,
        timestep_fraction=timestep_fraction,
        optim=OptimizerConfig(lr=1e-3),
    )
    actor.actor_module = _DummyDiffusionModule()
    actor.actor_optimizer = torch.optim.SGD(actor.actor_module.parameters(), lr=0.1)
    actor._optimizer_step = DataParallelPPOActor._optimizer_step.__get__(actor, DataParallelPPOActor)
    return actor


def _build_dance_batch(
    *,
    include_log_probs: bool,
    bestofn: int,
    num_generations: int,
    vq_coef: float = 1.0,
    mq_coef: float = 1.0,
    batch_size: int = 4,
    total_steps: int = 6,
) -> DataProto:
    log_probs = torch.randn(batch_size, total_steps, dtype=torch.float32) * 0.01
    tensors = {
        "latents": torch.randn(batch_size, total_steps, 1, 1, dtype=torch.float32),
        "next_latents": torch.randn(batch_size, total_steps, 1, 1, dtype=torch.float32),
        "timesteps": torch.arange(total_steps, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1),
        "prompt_embeds": torch.randn(batch_size, 4, 8, dtype=torch.float32),
        "negative_prompt_embeds": torch.randn(batch_size, 4, 8, dtype=torch.float32),
        "vq_advantages": torch.tensor([[-1.0], [-0.2], [0.3], [1.2]], dtype=torch.float32),
        "mq_advantages": torch.tensor([[1.0], [0.1], [-0.4], [-1.0]], dtype=torch.float32),
    }
    if include_log_probs:
        tensors["log_probs"] = log_probs
    else:
        tensors["old_log_probs"] = log_probs

    meta_info = {
        "diffusion_algo": "dancegrpo",
        "dance_bestofn": bestofn,
        "dance_num_generations": num_generations,
        "dance_vq_coef": vq_coef,
        "dance_mq_coef": mq_coef,
    }
    return DataProto.from_dict(tensors=tensors, meta_info=meta_info)


def _metric_scalar(metrics: dict, key: str) -> float:
    value = metrics[key]
    if isinstance(value, list):
        value = value[-1]
    return float(value)


class TestDiffusionDanceActorOnCPU(unittest.TestCase):
    def test_dance_bestofn_must_be_even(self):
        actor = _build_actor()
        batch = _build_dance_batch(include_log_probs=True, bestofn=3, num_generations=4)

        with self.assertRaisesRegex(ValueError, "even"):
            actor.update_policy_diffusion(batch)

    def test_dance_passthrough_path_and_timestep_fraction(self):
        actor = _build_actor(timestep_fraction=0.5)
        batch = _build_dance_batch(include_log_probs=True, bestofn=4, num_generations=4, total_steps=6)
        call_shapes = []

        def _fake_forward(model_inputs, temperature, step_idx=0):
            _ = temperature
            call_shapes.append((model_inputs["log_probs"].shape[0], step_idx))
            bias = actor.actor_module.logit_bias
            new_log_probs = model_inputs["log_probs"][:, step_idx] + bias
            prev_sample_mean = torch.zeros((new_log_probs.shape[0], 1), dtype=new_log_probs.dtype)
            return new_log_probs, prev_sample_mean

        actor._forward_micro_batch = _fake_forward
        metrics = actor.update_policy_diffusion(batch)

        self.assertEqual(len(call_shapes), 3)  # int(6 * 0.5)
        self.assertTrue(all(shape[0] == 4 for shape in call_shapes))
        self.assertIn("actor/dance/vq_loss", metrics)
        self.assertIn("actor/dance/mq_loss", metrics)
        self.assertIn("actor/dance/final_loss", metrics)
        self.assertIn("actor/dance/train_step_ratio", metrics)
        self.assertIn("actor/dance/bestofn_hit_rate", metrics)

        final_loss = _metric_scalar(metrics, "actor/dance/final_loss")
        step_ratio = _metric_scalar(metrics, "actor/dance/train_step_ratio")
        hit_rate = _metric_scalar(metrics, "actor/dance/bestofn_hit_rate")

        self.assertTrue(torch.isfinite(torch.tensor(final_loss)).item())
        self.assertAlmostEqual(step_ratio, 0.5, places=6)
        self.assertAlmostEqual(hit_rate, 1.0, places=6)

    def test_dance_old_log_prob_fallback_and_bestofn_selection(self):
        actor = _build_actor(timestep_fraction=0.5)
        batch = _build_dance_batch(include_log_probs=False, bestofn=2, num_generations=4, total_steps=6)
        call_shapes = []

        def _fake_forward(model_inputs, temperature, step_idx=0):
            _ = temperature
            call_shapes.append((model_inputs["log_probs"].shape[0], step_idx))
            bias = actor.actor_module.logit_bias
            new_log_probs = model_inputs["log_probs"][:, step_idx] + bias
            prev_sample_mean = torch.zeros((new_log_probs.shape[0], 1), dtype=new_log_probs.dtype)
            return new_log_probs, prev_sample_mean

        actor._forward_micro_batch = _fake_forward
        metrics = actor.update_policy_diffusion(batch)

        self.assertEqual(len(call_shapes), 3)  # int(6 * 0.5)
        self.assertTrue(all(shape[0] == 2 for shape in call_shapes))  # selected batch size == bestofn

        final_loss = _metric_scalar(metrics, "actor/dance/final_loss")
        hit_rate = _metric_scalar(metrics, "actor/dance/bestofn_hit_rate")
        self.assertTrue(torch.isfinite(torch.tensor(final_loss)).item())
        self.assertAlmostEqual(hit_rate, 0.5, places=6)

    def test_dance_bestofn_applies_before_micro_split(self):
        actor = _build_actor(timestep_fraction=0.5)
        actor.config.ppo_micro_batch_size_per_gpu = 2
        batch = _build_dance_batch(include_log_probs=False, bestofn=2, num_generations=4, total_steps=6)
        call_shapes = []

        def _fake_forward(model_inputs, temperature, step_idx=0):
            _ = temperature
            call_shapes.append((model_inputs["log_probs"].shape[0], step_idx))
            bias = actor.actor_module.logit_bias
            new_log_probs = model_inputs["log_probs"][:, step_idx] + bias
            prev_sample_mean = torch.zeros((new_log_probs.shape[0], 1), dtype=new_log_probs.dtype)
            return new_log_probs, prev_sample_mean

        actor._forward_micro_batch = _fake_forward
        metrics = actor.update_policy_diffusion(batch)

        # Best-of-N happens on mini-batch first: selected batch size is 2, then split by micro=2 => 1 micro batch.
        self.assertEqual(len(call_shapes), 3)  # int(6 * 0.5)
        self.assertTrue(all(shape[0] == 2 for shape in call_shapes))

        hit_rate = _metric_scalar(metrics, "actor/dance/bestofn_hit_rate")
        self.assertAlmostEqual(hit_rate, 0.5, places=6)
