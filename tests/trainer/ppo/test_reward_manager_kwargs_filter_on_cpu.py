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

from omegaconf import OmegaConf

from verl.trainer.ppo.reward import load_reward_manager
from verl.workers.reward_manager.batch import BatchRewardManager
from verl.workers.reward_manager.naive import NaiveRewardManager


def test_load_reward_manager_filters_batch_only_kwargs_for_naive():
    config = OmegaConf.create(
        {
            "data": {"reward_fn_key": "data_source"},
            "reward_model": {
                "reward_manager": "naive",
                "reward_kwargs": {
                    "backend": "builtin",
                    "use_videoalign": False,
                    "videoalign_device": "cpu",
                },
            },
        }
    )

    manager = load_reward_manager(
        config,
        tokenizer=None,
        num_examine=0,
        **config.reward_model.get("reward_kwargs", {}),
    )

    assert isinstance(manager, NaiveRewardManager)


def test_load_reward_manager_keeps_batch_kwargs_for_batch_manager():
    config = OmegaConf.create(
        {
            "data": {"reward_fn_key": "data_source"},
            "reward_model": {
                "reward_manager": "batch",
                "reward_kwargs": {
                    "backend": "videoalign",
                    "use_videoalign": True,
                    "videoalign_device": "cpu",
                },
            },
        }
    )

    manager = load_reward_manager(
        config,
        tokenizer=None,
        num_examine=0,
        **config.reward_model.get("reward_kwargs", {}),
    )

    assert isinstance(manager, BatchRewardManager)
    assert manager.reward_backend == "videoalign"
    assert manager.videoalign_init_error == "missing_model_path"
