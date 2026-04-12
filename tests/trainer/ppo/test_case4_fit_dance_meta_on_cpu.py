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

import json
from pathlib import Path

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


class _RecordingDanceActorRolloutWG:
    def __init__(self):
        self.generate_inputs = []
        self.update_inputs = []

    def generate_sequences(self, gen_batch: DataProto) -> DataProto:
        self.generate_inputs.append(gen_batch)
        return DataProto.from_dict(
            tensors={"timesteps": torch.zeros((2, 1), dtype=torch.long)},
            meta_info={"sigma_schedule": [1.0, 0.0]},
        )

    def update_actor(self, batch: DataProto) -> DataProto:
        self.update_inputs.append(batch)
        return DataProto(meta_info={"metrics": {"actor/loss": 0.5}})


def test_fit_dance_case4_uses_encoder_state_protocol_and_skips_legacy_diffusion_flow(monkeypatch):
    def _legacy_path_should_not_run(*_args, **_kwargs):
        raise AssertionError("legacy diffusion advantage path should not run in dance_case4")

    monkeypatch.setattr("verl.trainer.ppo.ray_trainer.compute_advantage_diffusion", _legacy_path_should_not_run)

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.diffusion = True
    trainer.diffusion_disaggregate = False
    trainer.actor_rollout_wg = _RecordingDanceActorRolloutWG()
    trainer.total_training_steps = 1
    trainer.train_dataloader = [
        (
            torch.ones((2, 3, 4), dtype=torch.float32),
            torch.ones((2, 3), dtype=torch.long),
            ["caption-a", "caption-b"],
        )
    ]
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
                    "dance_case4_mode": True,
                },
            },
        }
    )

    trainer.fit()

    assert len(trainer.actor_rollout_wg.generate_inputs) == 1
    generate_input = trainer.actor_rollout_wg.generate_inputs[0]
    assert set(generate_input.batch.keys()) == {"encoder_hidden_states", "encoder_attention_mask"}
    assert torch.equal(generate_input.batch["encoder_hidden_states"], torch.ones((2, 3, 4), dtype=torch.float32))
    assert torch.equal(generate_input.batch["encoder_attention_mask"], torch.ones((2, 3), dtype=torch.long))
    assert generate_input.meta_info["caption"] == ["caption-a", "caption-b"]
    assert len(trainer.actor_rollout_wg.update_inputs) == 1


def test_create_dataloader_dance_case4_skips_validation_construction(monkeypatch):
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
                    "dance_case4_mode": True,
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


def test_record_dance_case1_step_timing_persists_async_update_timeline(tmp_path):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "project_name": "dance_case1_test",
                "experiment_name": "timing",
                "step_timing_report_dir": str(tmp_path),
            },
            "data": {
                "train_batch_size": 16,
                "gen_batch_size": 16,
            },
            "actor_rollout_ref": {
                "rollout": {
                    "sampling_steps": 28,
                    "num_generations": 4,
                    "bestofn": 4,
                },
            },
        }
    )

    trainer._init_dance_dual_rollout_step_timing(case_name="dance_case1", schedule="pipelined_micro_batch")
    trainer._record_dance_dual_rollout_step_timing(
        epoch=0,
        step=1,
        timing_raw={
            "step": 10.0,
            "prompt_prepare": 0.5,
            "rollout_collect": 8.0,
            "update_wait": 1.5,
        },
        actor_metrics={"actor/loss": 0.123},
        step_meta_info={
            "prompt_batch_count": 8,
            "actor_prompt_count": 1,
            "actor_world_size": 8,
            "rollout_rounds": 8,
            "update_count": 2,
            "rollout_collect_end_offset_s": 8.5,
            "update_events": [
                {
                    "update_id": 0,
                    "input_batch_size": 8,
                    "prepared_batch_count": 2,
                    "step_weight": False,
                    "submit_offset_s": 3.0,
                    "wait_start_offset_s": 8.6,
                    "finish_offset_s": 8.7,
                    "wait_duration_s": 0.1,
                },
                {
                    "update_id": 1,
                    "input_batch_size": 8,
                    "prepared_batch_count": 2,
                    "step_weight": True,
                    "submit_offset_s": 6.0,
                    "wait_start_offset_s": 9.0,
                    "finish_offset_s": 9.4,
                    "wait_duration_s": 0.4,
                },
            ],
        },
    )

    summary = json.loads((tmp_path / "step_timing_summary.json").read_text(encoding="utf-8"))
    record = summary["steps"][0]
    assert record["update_submit_count"] == 2
    assert record["update_input_samples_total"] == 16
    assert record["update_tail_after_rollout_s"] == pytest.approx(0.9)
    assert record["max_update_wait_duration_s"] == pytest.approx(0.4)
    assert record["update_events"][1]["step_weight"] is True
    assert summary["async_update"]["tail_after_rollout_s"]["mean_s"] == pytest.approx(0.9)


@pytest.mark.parametrize(
    ("config_path", "expected_master_weight_type"),
    [
        ("examples/diffusion/config_video_diffusion_case1_dance.yaml", "bf16"),
        ("examples/diffusion/config_video_diffusion_case2_dance.yaml", "bf16"),
        ("examples/diffusion/config_video_diffusion_case3_dance.yaml", "bf16"),
        ("examples/diffusion/config_video_diffusion_case3_dance_nodes.yaml", "bf16"),
        ("examples/diffusion/config_video_diffusion_case3_dance_npu.yaml", "bf16"),
        ("examples/diffusion/config_video_diffusion_case4_dance.yaml", "bf16"),
        ("examples/diffusion/config_video_diffusion_case4_dance_nodes.yaml", "bf16"),
        ("examples/diffusion/config_video_diffusion_case4_dance_npu.yaml", "bf16"),
        ("examples/diffusion/config_video_diffusion_case4_dance_nodes_npu.yaml", "bf16"),
    ],
)
def test_dance_configs_align_master_weight_precision(config_path, expected_master_weight_type):
    config = OmegaConf.load(Path(config_path))
    assert config.actor_rollout_ref.actor.extra.dance.master_weight_type == expected_master_weight_type
