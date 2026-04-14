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


class _ImmediateFuture:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


class _RecordingCase1ActorWG:
    def __init__(self):
        self.world_size = 2
        self.generate_inputs = []
        self.update_inputs = []

    def generate_sequences_dance_async(self, gen_batch: DataProto):
        self.generate_inputs.append(gen_batch)
        batch_size = len(gen_batch)
        return _ImmediateFuture(
            DataProto.from_dict(
                tensors={
                    "timesteps": torch.zeros((batch_size, 1), dtype=torch.long),
                    "latents": torch.zeros((batch_size, 1, 1), dtype=torch.float32),
                    "next_latents": torch.zeros((batch_size, 1, 1), dtype=torch.float32),
                    "log_probs": torch.zeros((batch_size, 1), dtype=torch.float32),
                    "vq_rewards": torch.tensor([1.0, -1.0], dtype=torch.float32)[:batch_size],
                    "mq_rewards": torch.tensor([0.5, -0.5], dtype=torch.float32)[:batch_size],
                    "encoder_hidden_states": gen_batch.batch["encoder_hidden_states"].clone(),
                    "encoder_attention_mask": gen_batch.batch["encoder_attention_mask"].clone(),
                },
                meta_info={"sigma_schedule": [1.0, 0.0]},
            )
        )

    def update_actor_dance_async(self, batch: DataProto):
        self.update_inputs.append(batch)
        return _ImmediateFuture(DataProto(meta_info={"metrics": {"actor/loss": 0.125}}))


class _RecordingCase1RolloutRefWG:
    def __init__(self):
        self.world_size = 2
        self.generate_inputs = []

    def generate_sequences_dance_async(self, gen_batch: DataProto):
        self.generate_inputs.append(gen_batch)
        batch_size = len(gen_batch)
        return _ImmediateFuture(
            DataProto.from_dict(
                tensors={
                    "timesteps": torch.zeros((batch_size, 1), dtype=torch.long),
                    "latents": torch.zeros((batch_size, 1, 1), dtype=torch.float32),
                    "next_latents": torch.zeros((batch_size, 1, 1), dtype=torch.float32),
                    "log_probs": torch.zeros((batch_size, 1), dtype=torch.float32),
                    "vq_rewards": torch.tensor([0.6, -0.6], dtype=torch.float32)[:batch_size],
                    "mq_rewards": torch.tensor([0.3, -0.3], dtype=torch.float32)[:batch_size],
                    "encoder_hidden_states": gen_batch.batch["encoder_hidden_states"].clone(),
                    "encoder_attention_mask": gen_batch.batch["encoder_attention_mask"].clone(),
                },
                meta_info={"sigma_schedule": [1.0, 0.0]},
            )
        )


def test_fit_dis_dance_case1_uses_pipelined_dual_rollout_and_step_weight(monkeypatch):
    monkeypatch.setattr(
        "verl.trainer.ppo.ray_trainer.ray.remote",
        lambda fn: type("_Remote", (), {"remote": staticmethod(fn)})(),
    )
    monkeypatch.setattr(
        "verl.trainer.ppo.ray_trainer.ray.wait",
        lambda refs, num_returns=1: (refs[:num_returns], refs[num_returns:]),
    )
    monkeypatch.setattr("verl.trainer.ppo.ray_trainer.ray.get", lambda obj: obj)

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.diffusion = True
    trainer.diffusion_disaggregate = True
    trainer.total_training_steps = 1
    trainer.train_dataloader = [
        (
            torch.ones((4, 3, 4), dtype=torch.float32),
            torch.ones((4, 3), dtype=torch.long),
            ["caption-a", "caption-b", "caption-c", "caption-d"],
        )
    ]
    trainer.actor_wg = _RecordingCase1ActorWG()
    trainer.rollout_ref_wg = _RecordingCase1RolloutRefWG()
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "max_train_steps": 1,
                "pipelined_micro_batch": True,
            },
            "algorithm": {
                "adv_estimator": "grpo",
            },
            "actor_rollout_ref": {
                "actor": {
                    "dance_case1_mode": True,
                    "gradient_accumulation_steps": 2,
                    "extra": {
                        "dance": {
                            "actor_prompt_batch_count": 2,
                        }
                    },
                },
                "rollout": {
                    "num_generations": 2,
                    "bestofn": 2,
                    "vq_coef": 1.0,
                    "mq_coef": 1.0,
                },
            },
        }
    )

    calls = {"init_workers_dis": 0}

    def _fake_init_workers_dis():
        calls["init_workers_dis"] += 1

    trainer.init_workers_dis = _fake_init_workers_dis

    trainer.fit_dis()

    assert calls["init_workers_dis"] == 1
    assert len(trainer.actor_wg.generate_inputs) == 2
    assert len(trainer.rollout_ref_wg.generate_inputs) == 2
    first_prompt = trainer.actor_wg.generate_inputs[0]
    last_prompt = trainer.rollout_ref_wg.generate_inputs[-1]
    assert first_prompt.meta_info["window_id"] == 0
    assert first_prompt.meta_info["micro_batch_id"] == 0
    assert first_prompt.meta_info["is_last_micro_batch"] is False
    assert last_prompt.meta_info["is_last_micro_batch"] is True
    assert [batch.meta_info["step_weight"] for batch in trainer.actor_wg.update_inputs] == [False, True, False, True]
    assert all(batch.meta_info["use_precomputed_advantages"] is True for batch in trainer.actor_wg.update_inputs)
    assert all(batch.meta_info["skip_bestofn"] is True for batch in trainer.actor_wg.update_inputs)


def test_fit_dis_dance_case1_fail_fast_reports_pipelined_preconditions():
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.diffusion = True
    trainer.diffusion_disaggregate = True
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "pipelined_micro_batch": False,
            },
            "algorithm": {
                "adv_estimator": "grpo",
            },
            "actor_rollout_ref": {
                "actor": {
                    "dance_case1_mode": True,
                },
            },
        }
    )

    with pytest.raises(
        ValueError,
        match=r"trainer\.pipelined_micro_batch must be true",
    ):
        trainer.fit_dis()


def test_create_dataloader_dance_case1_skips_validation_construction(monkeypatch):
    created_dataloaders = []

    class _FakeLatentDataset:
        def __init__(self, json_path, num_latent_t, cfg_rate):
            self.json_path = json_path
            self.num_latent_t = num_latent_t
            self.cfg_rate = cfg_rate

        def __len__(self):
            return 4

    class _FakeStatefulDataLoader:
        def __init__(self, dataset, batch_size, **kwargs):
            created_dataloaders.append(
                {
                    "dataset": dataset,
                    "batch_size": batch_size,
                    "kwargs": kwargs,
                }
            )
            self.dataset = dataset
            self.batch_size = batch_size

        def __len__(self):
            return 2

    monkeypatch.setattr("fastvideo.dataset.latent_rl_datasets.LatentDataset", _FakeLatentDataset)
    monkeypatch.setattr(
        "fastvideo.dataset.latent_rl_datasets.latent_collate_function",
        lambda batch: batch,
    )
    monkeypatch.setattr("verl.trainer.ppo.ray_trainer.StatefulDataLoader", _FakeStatefulDataLoader)

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "data": {
                "data_json_path": "/tmp/dance_case1.json",
                "t": 28,
                "cfg": 0.0,
                "train_batch_size": 2,
                "gen_batch_size": 4,
                "dataloader_num_workers": 0,
            },
            "trainer": {
                "total_epochs": 1,
                "total_training_steps": None,
            },
            "actor_rollout_ref": {
                "actor": {
                    "dance_case1_mode": True,
                },
            },
        }
    )

    trainer._create_dataloader(train_dataset=None, val_dataset=None, collate_fn=None, train_sampler=None)

    assert isinstance(trainer.train_dataset, _FakeLatentDataset)
    assert trainer.train_dataset.json_path == "/tmp/dance_case1.json"
    assert trainer.val_dataset is None
    assert trainer.val_dataloader is None
    assert len(created_dataloaders) == 1
    assert created_dataloaders[0]["batch_size"] == 2
