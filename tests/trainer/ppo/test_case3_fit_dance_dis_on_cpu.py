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


class _RecordingRolloutRefWG:
    def __init__(self):
        self.generate_inputs = []

    def generate_sequences(self, gen_batch: DataProto) -> DataProto:
        self.generate_inputs.append(gen_batch)
        return DataProto.from_dict(
            tensors={"timesteps": torch.zeros((2, 1), dtype=torch.long)},
            meta_info={"sigma_schedule": [1.0, 0.0]},
        )


class _RecordingActorWG:
    def __init__(self):
        self.update_inputs = []

    def update_actor(self, batch: DataProto) -> DataProto:
        self.update_inputs.append(batch)
        return DataProto(meta_info={"metrics": {"actor/loss": 0.25}})


def test_fit_dis_dance_case3_uses_encoder_state_protocol_and_syncs_before_rollout(monkeypatch):
    def _legacy_path_should_not_run(*_args, **_kwargs):
        raise AssertionError("legacy diffusion fit_dis path should not run in dance_case3")

    monkeypatch.setattr("verl.trainer.ppo.ray_trainer.compute_advantage_diffusion", _legacy_path_should_not_run)

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.diffusion = True
    trainer.diffusion_disaggregate = True
    trainer.total_training_steps = 1
    trainer.train_dataloader = [
        (
            torch.ones((2, 3, 4), dtype=torch.float32),
            torch.ones((2, 3), dtype=torch.long),
            ["caption-a", "caption-b"],
        )
    ]
    trainer.rollout_ref_wg = _RecordingRolloutRefWG()
    trainer.actor_wg = _RecordingActorWG()
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "max_train_steps": 1,
                "pipelined_micro_batch": False,
            },
            "algorithm": {
                "adv_estimator": "grpo",
            },
            "actor_rollout_ref": {
                "actor": {
                    "dance_case3_mode": True,
                },
            },
        }
    )

    calls = {"init_workers_dis": 0, "sync": 0}

    def _fake_init_workers_dis():
        calls["init_workers_dis"] += 1

    def _fake_sync():
        calls["sync"] += 1

    trainer.init_workers_dis = _fake_init_workers_dis
    trainer._sync_diffusion_disaggregate_before_rollout = _fake_sync

    trainer.fit_dis()

    assert calls["init_workers_dis"] == 1
    assert calls["sync"] == 1
    assert len(trainer.rollout_ref_wg.generate_inputs) == 1
    generate_input = trainer.rollout_ref_wg.generate_inputs[0]
    assert set(generate_input.batch.keys()) == {"encoder_hidden_states", "encoder_attention_mask"}
    assert torch.equal(generate_input.batch["encoder_hidden_states"], torch.ones((2, 3, 4), dtype=torch.float32))
    assert torch.equal(generate_input.batch["encoder_attention_mask"], torch.ones((2, 3), dtype=torch.long))
    assert generate_input.meta_info["caption"] == ["caption-a", "caption-b"]
    assert len(trainer.actor_wg.update_inputs) == 1


def test_fit_dis_dance_case3_fail_fast_reports_disaggregate_preconditions():
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.diffusion = True
    trainer.diffusion_disaggregate = False
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
                    "dance_case3_mode": True,
                },
            },
        }
    )

    with pytest.raises(
        ValueError,
        match=r"trainer\.disaggregate must be true",
    ):
        trainer.fit_dis()


def test_create_dataloader_dance_case3_skips_validation_construction(monkeypatch):
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
                "data_json_path": "/tmp/dance.json",
                "t": 28,
                "cfg": 0.0,
                "gen_batch_size": 4,
                "dataloader_num_workers": 0,
            },
            "trainer": {
                "total_epochs": 1,
                "total_training_steps": None,
            },
            "actor_rollout_ref": {
                "actor": {
                    "dance_case3_mode": True,
                },
            },
        }
    )

    trainer._create_dataloader(train_dataset=None, val_dataset=None, collate_fn=None, train_sampler=None)

    assert isinstance(trainer.train_dataset, _FakeLatentDataset)
    assert trainer.train_dataset.json_path == "/tmp/dance.json"
    assert trainer.val_dataset is None
    assert trainer.val_dataloader is None
    assert len(created_dataloaders) == 1
