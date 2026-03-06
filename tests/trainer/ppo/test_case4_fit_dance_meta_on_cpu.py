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
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def _build_trainer(diffusion_algo: str) -> RayPPOTrainer:
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.diffusion = True
    trainer.diffusion_algo = diffusion_algo
    return trainer


def _build_diffusion_gen_batch(with_seed: bool) -> DataProto:
    tensors = {
        "prompt_embeds": torch.zeros((2, 1, 4, 8), dtype=torch.float32),
        "negative_prompt_embeds": torch.zeros((2, 1, 4, 8), dtype=torch.float32),
    }
    if with_seed:
        tensors["seed"] = torch.tensor([11, 12], dtype=torch.long)
    return DataProto.from_dict(tensors=tensors, meta_info={"global_steps": 7})


def test_fit_dance_meta_sets_diffusion_algo_and_use_seed_true_when_seed_exists():
    trainer = _build_trainer("dancegrpo")
    gen_batch = _build_diffusion_gen_batch(with_seed=True)

    trainer._prepare_diffusion_gen_meta(gen_batch)

    assert gen_batch.meta_info["global_steps"] == 7
    assert gen_batch.meta_info["diffusion_algo"] == "dancegrpo"
    assert gen_batch.meta_info["use_seed"] is True


def test_fit_dance_meta_sets_use_seed_false_when_seed_not_exists():
    trainer = _build_trainer("dancegrpo")
    gen_batch = _build_diffusion_gen_batch(with_seed=False)

    trainer._prepare_diffusion_gen_meta(gen_batch)

    assert gen_batch.meta_info["global_steps"] == 7
    assert gen_batch.meta_info["diffusion_algo"] == "dancegrpo"
    assert gen_batch.meta_info["use_seed"] is False


def test_fit_flow_grpo_path_does_not_inject_use_seed_field():
    trainer = _build_trainer("flow_grpo")
    gen_batch = _build_diffusion_gen_batch(with_seed=True)

    trainer._prepare_diffusion_gen_meta(gen_batch)

    assert gen_batch.meta_info["global_steps"] == 7
    assert gen_batch.meta_info["diffusion_algo"] == "flow_grpo"
    assert "use_seed" not in gen_batch.meta_info


class _FakeActorRolloutWG:
    def __init__(self, num_generations: int, use_group: bool):
        self.num_generations = num_generations
        self.use_group = use_group

    def generate_sequences(self, gen_batch: DataProto) -> DataProto:
        batch_size = len(gen_batch)
        if gen_batch.meta_info.get("diffusion_algo") == "dancegrpo" and self.use_group:
            batch_size = batch_size * self.num_generations
        return DataProto.from_dict(
            tensors={
                "timesteps": torch.zeros((batch_size, 1), dtype=torch.long),
                "vq_rewards": torch.full((batch_size,), -1.0, dtype=torch.float32),
                "mq_rewards": torch.full((batch_size,), -1.0, dtype=torch.float32),
            },
            meta_info={"timing": {}},
        )


def _fake_reward_fn(batch: DataProto, return_dict: bool = False):
    reward = torch.zeros((len(batch), 1), dtype=torch.float32)
    if return_dict:
        reward_extra_info = {
            "VQ": [0.1 for _ in range(len(batch))],
            "MQ": [0.2 for _ in range(len(batch))],
        }
        return {"reward_tensor": reward, "reward_extra_info": reward_extra_info}
    return reward


def test_fit_case4_group_rollout_repeats_batch_before_advantage(monkeypatch):
    def _raise_with_batch_size(batch: DataProto, **kwargs):
        _ = kwargs
        raise RuntimeError(f"batch_size={len(batch)}")

    monkeypatch.setattr("verl.trainer.ppo.ray_trainer.compute_advantage_diffusion", _raise_with_batch_size)

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.diffusion = True
    trainer.diffusion_algo = "dancegrpo"
    trainer.disaggregate_actor_rollout = False
    trainer.async_rollout_mode = False
    trainer.use_rm = False
    trainer.use_critic = False
    trainer.val_reward_fn = None
    trainer.reward_fn = _fake_reward_fn
    trainer.actor_rollout_wg = _FakeActorRolloutWG(num_generations=2, use_group=True)
    trainer.total_training_steps = 1
    trainer.train_dataloader = [
        {
            "prompt_embeds": torch.zeros((3, 1, 4, 8), dtype=torch.float32),
            "negative_prompt_embeds": torch.zeros((3, 1, 4, 8), dtype=torch.float32),
            "pooled_prompt_embeds": torch.zeros((3, 1, 8), dtype=torch.float32),
            "negative_pooled_prompt_embeds": torch.zeros((3, 1, 8), dtype=torch.float32),
        }
    ]
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "profile_steps": None,
                "profile_continuous_steps": False,
                "total_epochs": 1,
                "balance_batch": False,
                "critic_warmup": 2,
                "test_freq": 0,
                "save_freq": 0,
                "esi_redundant_time": 0,
            },
            "reward_model": {"launch_reward_fn_async": False},
            "algorithm": {
                "adv_estimator": "grpo",
                "use_kl_in_reward": False,
                "gamma": 1.0,
                "lam": 1.0,
                "norm_adv_by_std_in_grpo": True,
            },
            "actor_rollout_ref": {
                "actor": {"use_kl_loss": False},
                "rollout": {
                    "n": 1,
                    "num_generations": 2,
                    "use_group": True,
                    "multi_turn": {"enable": False},
                    "bestofn": 2,
                    "vq_coef": 1.0,
                    "mq_coef": 1.0,
                },
            },
        }
    )

    trainer._load_checkpoint = lambda: None
    trainer._prof_start = lambda: None
    trainer._prof_step = lambda: None
    trainer._prof_stop = lambda: None
    trainer._start_profiling = lambda *_args, **_kwargs: None
    trainer._stop_profiling = lambda *_args, **_kwargs: None
    trainer._log_step_timing = lambda **_kwargs: None

    with pytest.raises(
        RuntimeError,
        match=r"batch_size=6",
    ):
        trainer.fit()


def test_fit_case4_fail_fast_reports_group_precondition_fields():
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.diffusion = True
    trainer.diffusion_algo = "dancegrpo"
    trainer.disaggregate_actor_rollout = False
    trainer.async_rollout_mode = False
    trainer.use_rm = False
    trainer.use_critic = False
    trainer.val_reward_fn = None
    trainer.reward_fn = _fake_reward_fn
    trainer.actor_rollout_wg = _FakeActorRolloutWG(num_generations=2, use_group=False)
    trainer.total_training_steps = 1
    trainer.train_dataloader = [
        {
            "prompt_embeds": torch.zeros((3, 1, 4, 8), dtype=torch.float32),
            "negative_prompt_embeds": torch.zeros((3, 1, 4, 8), dtype=torch.float32),
            "pooled_prompt_embeds": torch.zeros((3, 1, 8), dtype=torch.float32),
            "negative_pooled_prompt_embeds": torch.zeros((3, 1, 8), dtype=torch.float32),
        }
    ]
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "profile_steps": None,
                "profile_continuous_steps": False,
                "total_epochs": 1,
                "balance_batch": False,
                "critic_warmup": 2,
                "test_freq": 0,
                "save_freq": 0,
                "esi_redundant_time": 0,
            },
            "reward_model": {"launch_reward_fn_async": False},
            "algorithm": {
                "adv_estimator": "grpo",
                "use_kl_in_reward": False,
                "gamma": 1.0,
                "lam": 1.0,
                "norm_adv_by_std_in_grpo": True,
            },
            "actor_rollout_ref": {
                "actor": {"use_kl_loss": False},
                "rollout": {
                    "n": 1,
                    "num_generations": 2,
                    "use_group": False,
                    "multi_turn": {"enable": False},
                    "bestofn": 2,
                    "vq_coef": 1.0,
                    "mq_coef": 1.0,
                },
            },
        }
    )

    trainer._load_checkpoint = lambda: None
    trainer._prof_start = lambda: None
    trainer._prof_step = lambda: None
    trainer._prof_stop = lambda: None
    trainer._start_profiling = lambda *_args, **_kwargs: None
    trainer._stop_profiling = lambda *_args, **_kwargs: None
    trainer._log_step_timing = lambda **_kwargs: None

    with pytest.raises(
        ValueError,
        match=(
            r"len\(batch\)=3, num_generations=2, diffusion_algo=dancegrpo, use_group=False"
        ),
    ):
        trainer.fit()
