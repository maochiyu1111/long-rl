# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import unittest
from typing import List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from verl.trainer.config import AlgoConfig, KLControlConfig
from verl.trainer.ppo.core_algos import (
    compute_gae_advantage_return,
    compute_grpo_outcome_advantage,
    get_adv_estimator_fn,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.config import omega_conf_to_dataclass


class TestAlgoConfig(unittest.TestCase):
    """Test the AlgoConfig dataclass and its integration with core algorithms."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a sample algorithm config as DictConfig (similar to what comes from YAML)
        self.config_dict = {
            "_target_": "verl.trainer.config.AlgoConfig",
            "gamma": 0.99,
            "lam": 0.95,
            "adv_estimator": "gae",
            "norm_adv_by_std_in_grpo": True,
            "use_kl_in_reward": True,
            "kl_penalty": "kl",
            "kl_ctrl": {
                "_target_": "verl.trainer.config.KLControlConfig",
                "type": "adaptive",
                "kl_coef": 0.002,
                "horizon": 5000,
                "target_kl": 0.05,
            },
            "use_pf_ppo": True,
            "pf_ppo": {"reweight_method": "max_min", "weight_pow": 3.0},
            "dual_reward_missing_strategy": "fill",
            "dual_reward_fill_value": -1.0,
            "dual_adv_mode_default": "group",
        }
        self.omega_config = OmegaConf.create(self.config_dict)

    def test_dataclass_creation_from_dict(self):
        """Test creating AlgoConfig from dictionary."""
        config = omega_conf_to_dataclass(self.config_dict)

        self.assertIsInstance(config, AlgoConfig)
        self.assertEqual(config.gamma, 0.99)
        self.assertEqual(config.lam, 0.95)
        self.assertEqual(config.adv_estimator, "gae")
        self.assertTrue(config.norm_adv_by_std_in_grpo)
        self.assertTrue(config.use_kl_in_reward)
        self.assertEqual(config.kl_penalty, "kl")
        self.assertTrue(config.use_pf_ppo)
        self.assertEqual(config.dual_reward_missing_strategy, "fill")
        self.assertEqual(config.dual_reward_fill_value, -1.0)
        self.assertEqual(config.dual_adv_mode_default, "group")

    def test_dataclass_creation_from_omega_config(self):
        """Test creating AlgoConfig from OmegaConf DictConfig."""
        config = omega_conf_to_dataclass(self.omega_config)

        self.assertIsInstance(config, AlgoConfig)
        self.assertEqual(config.gamma, 0.99)
        self.assertEqual(config.lam, 0.95)

    def test_nested_configs(self):
        """Test that nested configurations are properly converted."""
        config = omega_conf_to_dataclass(self.omega_config)

        # Test KL control config
        self.assertIsInstance(config.kl_ctrl, KLControlConfig)
        self.assertEqual(config.kl_ctrl.type, "adaptive")
        self.assertEqual(config.kl_ctrl.kl_coef, 0.002)
        self.assertEqual(config.kl_ctrl.horizon, 5000)
        self.assertEqual(config.kl_ctrl.target_kl, 0.05)

        # Test PF PPO config
        self.assertEqual(config.pf_ppo.get("reweight_method"), "max_min")
        self.assertEqual(config.pf_ppo.get("weight_pow"), 3.0)

    def test_default_values(self):
        """Test that default values are properly set."""
        minimal_config = {"gamma": 0.8}
        config = omega_conf_to_dataclass(minimal_config, AlgoConfig)

        self.assertEqual(config.gamma, 0.8)
        self.assertEqual(config.lam, 1.0)  # default value
        self.assertEqual(config.adv_estimator, "gae")  # default value
        self.assertTrue(config.norm_adv_by_std_in_grpo)  # default value
        self.assertFalse(config.use_kl_in_reward)  # default value
        self.assertEqual(config.kl_penalty, "kl")  # default value
        self.assertFalse(config.use_pf_ppo)  # default value
        self.assertEqual(config.dual_reward_missing_strategy, "error")
        self.assertEqual(config.dual_reward_fill_value, 0.0)
        self.assertEqual(config.dual_adv_mode_default, "group")

    def test_get_method_backward_compatibility(self):
        """Test the get method for backward compatibility."""
        config = omega_conf_to_dataclass(self.omega_config)

        # Test existing attribute
        self.assertEqual(config.get("gamma"), 0.99)
        self.assertEqual(config.get("gamma", 1.0), 0.99)

        # Test non-existing attribute
        self.assertIsNone(config.get("non_existing"))
        self.assertEqual(config.get("non_existing", "default"), "default")

    def test_post_init_nested_configs(self):
        """Test that __post_init__ properly initializes nested configs when None."""
        # Create config without nested configs
        minimal_config = AlgoConfig(gamma=0.9)

        # Check that nested configs are initialized
        self.assertIsNotNone(minimal_config.kl_ctrl)
        self.assertIsInstance(minimal_config.kl_ctrl, KLControlConfig)
        assert not minimal_config.pf_ppo

    def test_config_init_from_yaml(self):
        import os

        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
            cfg = compose(config_name="ppo_trainer")
        algo_config = omega_conf_to_dataclass(cfg.algorithm)
        from verl.trainer.config import AlgoConfig

        assert isinstance(algo_config, AlgoConfig)
        assert algo_config.dual_reward_missing_strategy == "error"
        assert algo_config.dual_reward_fill_value == 0.0
        assert algo_config.dual_adv_mode_default == "group"


class TestDanceGRPOConfigValidation(unittest.TestCase):
    """Test stage-A dancegrpo config gate in RayPPOTrainer._validate_config."""

    @staticmethod
    def _compose_cfg(diffusion_algo: str = "dancegrpo", extra_overrides: Optional[List[str]] = None):
        import os

        from hydra import compose, initialize_config_dir

        overrides = [
            f"trainer.diffusion_algo={diffusion_algo}",
            "trainer.nnodes=1",
            "trainer.n_gpus_per_node=1",
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
            "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
            "critic.enable=false",
            "reward_model.enable=false",
        ]
        if extra_overrides:
            overrides.extend(extra_overrides)
        with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
            return compose(config_name="ppo_trainer", overrides=overrides)

    @staticmethod
    def _run_validate(cfg):
        trainer = RayPPOTrainer.__new__(RayPPOTrainer)
        trainer.config = cfg
        trainer.diffusion = False
        trainer.diffusion_algo = cfg.trainer.diffusion_algo
        trainer.use_reference_policy = False
        trainer.use_critic = False
        RayPPOTrainer._validate_config(trainer)

    def test_dancegrpo_valid_config_passes(self):
        cfg = self._compose_cfg("dancegrpo")
        self._run_validate(cfg)

    def test_dancegrpo_bestofn_must_not_exceed_num_generations(self):
        cfg = self._compose_cfg("dancegrpo", ["actor_rollout_ref.rollout.bestofn=26"])
        with self.assertRaisesRegex(ValueError, "bestofn"):
            self._run_validate(cfg)

    def test_dancegrpo_bestofn_must_be_even(self):
        cfg = self._compose_cfg("dancegrpo", ["actor_rollout_ref.rollout.bestofn=7"])
        with self.assertRaisesRegex(ValueError, "must be even"):
            self._run_validate(cfg)

    def test_dancegrpo_timestep_fraction_range(self):
        cfg = self._compose_cfg("dancegrpo", ["actor_rollout_ref.actor.timestep_fraction=1.5"])
        with self.assertRaisesRegex(ValueError, "timestep_fraction"):
            self._run_validate(cfg)

    def test_dancegrpo_non_negative_coeff(self):
        cfg = self._compose_cfg("dancegrpo", ["actor_rollout_ref.rollout.vq_coef=-0.1"])
        with self.assertRaisesRegex(ValueError, "vq_coef"):
            self._run_validate(cfg)
        cfg = self._compose_cfg("dancegrpo", ["actor_rollout_ref.rollout.mq_coef=-0.2"])
        with self.assertRaisesRegex(ValueError, "mq_coef"):
            self._run_validate(cfg)

    def test_dual_reward_config_values_are_validated(self):
        cfg = self._compose_cfg("dancegrpo", ["algorithm.dual_reward_missing_strategy=unknown"])
        with self.assertRaisesRegex(ValueError, "dual_reward_missing_strategy"):
            self._run_validate(cfg)

        cfg = self._compose_cfg("dancegrpo", ["algorithm.dual_adv_mode_default=unknown"])
        with self.assertRaisesRegex(ValueError, "dual_adv_mode_default"):
            self._run_validate(cfg)

        cfg = self._compose_cfg(
            "dancegrpo",
            ["algorithm.dual_reward_missing_strategy=fill", "algorithm.dual_adv_mode_default=batch"],
        )
        self._run_validate(cfg)

    def test_flow_grpo_default_path_ignores_dance_constraints(self):
        cfg = self._compose_cfg(
            "flow_grpo",
            [
                "actor_rollout_ref.rollout.bestofn=7",
                "actor_rollout_ref.actor.timestep_fraction=2.0",
                "actor_rollout_ref.rollout.vq_coef=-1.0",
            ],
        )
        self._run_validate(cfg)

    def test_diffusion_algo_enum_validation(self):
        cfg = self._compose_cfg("unknown_algo")
        with self.assertRaisesRegex(ValueError, "diffusion_algo"):
            self._run_validate(cfg)


class TestAlgoCompute(unittest.TestCase):
    """Test the AlgoConfig dataclass and its integration with core algorithms."""

    def setUp(self):
        """Set up test fixtures."""
        self.algo_config = AlgoConfig(
            gamma=0.99,
            lam=0.95,
            adv_estimator="gae",
            norm_adv_by_std_in_grpo=True,
            use_kl_in_reward=True,
            kl_penalty="kl",
            kl_ctrl=KLControlConfig(type="adaptive", kl_coef=0.002, horizon=5000, target_kl=0.05),
            use_pf_ppo=True,
            pf_ppo={"reweight_method": "max_min", "weight_pow": 3.0},
        )

    def test_advantage_estimator_with_cfg(self):
        """Test integration with advantage estimators from core_algos."""
        config = self.algo_config

        # Test GAE advantage estimator
        adv_fn = get_adv_estimator_fn(config.adv_estimator)
        self.assertIsNotNone(adv_fn)

        # Test with actual GAE computation
        batch_size, seq_len = 2, 5
        token_level_rewards = torch.randn(batch_size, seq_len)
        values = torch.randn(batch_size, seq_len)
        response_mask = torch.ones(batch_size, seq_len)

        advantages, returns = compute_gae_advantage_return(
            token_level_rewards=token_level_rewards,
            values=values,
            response_mask=response_mask,
            gamma=config.gamma,
            lam=config.lam,
        )

        self.assertEqual(advantages.shape, (batch_size, seq_len))
        self.assertEqual(returns.shape, (batch_size, seq_len))

    def test_grpo_advantage_estimator_with_cfg(self):
        """Test integration with GRPO advantage estimator."""
        grpo_config = AlgoConfig(adv_estimator="grpo", norm_adv_by_std_in_grpo=True)

        # Test GRPO advantage computation
        batch_size, seq_len = 4, 3
        token_level_rewards = torch.tensor([[1.0, 0.5, 0.0], [2.0, 1.0, 0.0], [0.5, 0.2, 0.0], [1.5, 0.8, 0.0]])
        response_mask = torch.ones(batch_size, seq_len)
        index = np.array([0, 0, 1, 1])  # Two groups

        advantages, returns = compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            norm_adv_by_std_in_grpo=grpo_config.norm_adv_by_std_in_grpo,
        )

        self.assertEqual(advantages.shape, (batch_size, seq_len))
        self.assertEqual(returns.shape, (batch_size, seq_len))


if __name__ == "__main__":
    unittest.main()
