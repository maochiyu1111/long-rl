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
"""Flux + DanceGRPO entrypoint using the long-rl Hydra/Ray style."""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ray_trainer_flux_dance import RayPPOTrainerDance, ResourcePoolManager, Role
from verl.utils.device import is_cuda_available
from verl.workers.fsdp_workers import CriticWorker
from verl.workers.fsdp_workers_flux_dance import FSDPWorkerDance


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_flux(config)


def _is_dance_case_enabled(config, case_name: str) -> bool:
    actor_config = getattr(config.actor_rollout_ref, "actor", {})
    return bool(actor_config.get(case_name, False))


def _validate_single_case(config) -> None:
    enabled = [
        name
        for name in (
            "dance_case1_mode",
            "dance_case2_mode",
            "dance_case3_mode",
            "dance_case4_mode",
        )
        if _is_dance_case_enabled(config, name)
    ]
    if len(enabled) != 1:
        raise ValueError(f"Flux dance requires exactly one dance_case*_mode=true, got {enabled}")


def run_flux(config) -> None:
    _validate_single_case(config)

    if not ray.is_initialized():
        ray.init(
            runtime_env=get_ppo_ray_runtime_env(),
            num_cpus=config.ray_init.num_cpus,
        )

    if (
        is_cuda_available
        and config.trainer.get("profile_steps") is not None
        and len(config.trainer.get("profile_steps", [])) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available

        assert is_nvtx_available(), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
        nsight_options = OmegaConf.to_container(config.trainer.controller_nsight_options)
        runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = TaskRunner.remote()

    ray.get(runner.run.remote(config))

    timeline_json_file = config.ray_init.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        from pprint import pprint

        from verl.single_controller.ray import RayWorkerGroup

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        fit_disaggregate = bool(getattr(config.trainer, "disaggregate", False))
        global_pool_id = "global_pool"
        actor_pool_id = "actor_pool"
        rollout_ref_pool_id = "rollout_ref_pool"

        if fit_disaggregate:
            role_worker_mapping = {
                Role.Actor: ray.remote(FSDPWorkerDance),
                Role.RolloutRef: ray.remote(FSDPWorkerDance),
                Role.Critic: ray.remote(CriticWorker),
            }
            total_gpus_per_node = int(config.trainer.n_gpus_per_node)
            actor_gpus = config.trainer.get("disaggregate_actor_n_gpus_per_node", None)
            rollout_gpus = config.trainer.get("disaggregate_rollout_ref_n_gpus_per_node", None)
            if actor_gpus is None and rollout_gpus is None:
                actor_gpus = total_gpus_per_node // 2
                rollout_gpus = total_gpus_per_node - actor_gpus
            elif actor_gpus is None:
                rollout_gpus = int(rollout_gpus)
                actor_gpus = total_gpus_per_node - rollout_gpus
            elif rollout_gpus is None:
                actor_gpus = int(actor_gpus)
                rollout_gpus = total_gpus_per_node - actor_gpus
            else:
                actor_gpus = int(actor_gpus)
                rollout_gpus = int(rollout_gpus)
            if actor_gpus <= 0 or rollout_gpus <= 0:
                raise ValueError(f"Invalid Flux resource split: {actor_gpus=}, {rollout_gpus=}")
            resource_pool_spec = {
                actor_pool_id: [actor_gpus] * int(config.trainer.nnodes),
                rollout_ref_pool_id: [rollout_gpus] * int(config.trainer.nnodes),
            }
            mapping = {
                Role.Actor: actor_pool_id,
                Role.RolloutRef: rollout_ref_pool_id,
                Role.Critic: actor_pool_id,
            }
        else:
            role_worker_mapping = {
                Role.ActorRollout: ray.remote(FSDPWorkerDance),
                Role.Critic: ray.remote(CriticWorker),
            }
            resource_pool_spec = {
                global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
            }
            mapping = {
                Role.ActorRollout: global_pool_id,
                Role.Critic: global_pool_id,
            }

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
        trainer = RayPPOTrainerDance(
            config=config,
            tokenizer=None,
            processor=None,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=RayWorkerGroup,
            reward_fn=None,
            val_reward_fn=None,
            train_dataset=None,
            val_dataset=None,
            collate_fn=None,
            train_sampler=None,
        )
        if fit_disaggregate:
            trainer.fit_dis()
        else:
            trainer.init_workers()
            trainer.fit()


if __name__ == "__main__":
    main()
