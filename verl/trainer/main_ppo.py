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
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.experimental.dataset.sampler import AbstractSampler
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.device import is_cuda_available
from verl.utils.import_utils import load_extern_type


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """Main entry point for PPO training with Hydra configuration management.

    Args:
        config_dict: Hydra configuration dictionary containing training parameters.
    """
    run_ppo(config)


# Define a function to run the PPO-like training process
def run_ppo(config) -> None:
    """Initialize Ray cluster and run distributed PPO training process.

    Args:
        config: Training configuration object containing all necessary parameters
                for distributed PPO training including Ray initialization settings,
                model paths, and training hyperparameters.
    """
    # Check if Ray is not initialized
    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        # Set environment variables in the runtime environment to control tokenizer parallelism,
        # NCCL debug level, VLLM logging level, and allow runtime LoRA updating
        # `num_cpus` specifies the number of CPU cores Ray can use, obtained from the configuration
        ray.init(
            runtime_env=get_ppo_ray_runtime_env(),
            num_cpus=config.ray_init.num_cpus,
        )

    # Create a remote instance of the TaskRunner class, and
    # Execute the `run` method of the TaskRunner instance remotely and wait for it to complete
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

    # [Optional] get the path of the timeline trace file from the configuration, default to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_init.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    """Ray remote class for executing distributed PPO training tasks.

    This class encapsulates the main training logic and runs as a Ray remote actor
    to enable distributed execution across multiple nodes and GPUs.
    """

    def run(self, config):
        """Execute the main PPO training workflow.

        This method sets up the distributed training environment, initializes
        workers, datasets, and reward functions, then starts the training process.

        Args:
            config: Training configuration object containing all parameters needed
                   for setting up and running the PPO training process.
        """
        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        diffusion_enabled = bool(config.trainer.get("diffusion", False))
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code, diffusion=diffusion_enabled)
        # Used for multimodal LLM, could be None
        processor = hf_processor(
            local_path, trust_remote_code=trust_remote_code, use_fast=True, diffusion=diffusion_enabled
        )

        # Define worker classes based on the actor strategy.
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            assert config.critic.strategy in {"fsdp", "fsdp2"}
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        placement = os.getenv("TASK_PLACEMENT", "colocated")
        fit_disaggregate = bool(getattr(config.trainer, "disaggregate", False))

        from verl.workers.fsdp_workers import ActorRolloutRefWorker_encoder, ActorRolloutRefWorker_llm

        # Define the resource pool specification.
        global_pool_id = "global_pool"
        actor_pool_id = "actor_pool"
        rollout_ref_pool_id = "rollout_ref_pool"
        actor_rollout_encoder_id = "actor_rollout_encoder_pool"
        actor_rollout_llm_id = "actor_rollout_llm_pool"
        ref_encoder_id = "ref_encoder_pool"
        ref_llm_id = "ref_llm_pool"

        # Map roles to their corresponding remote worker classes.
        # Map roles to the resource pool.
        if placement == "colocated":
            if fit_disaggregate:
                # fit_dis() uses separate WorkerGroups (actor + rollout_ref). By default, split GPUs evenly.
                total_gpus_per_node = int(config.trainer.n_gpus_per_node)
                actor_gpus_per_node = config.trainer.get("disaggregate_actor_n_gpus_per_node", None)
                rollout_ref_gpus_per_node = config.trainer.get("disaggregate_rollout_ref_n_gpus_per_node", None)

                if actor_gpus_per_node is None and rollout_ref_gpus_per_node is None:
                    if total_gpus_per_node < 2:
                        raise ValueError(
                            "trainer.disaggregate=true requires at least 2 GPUs per node by default; "
                            "or explicitly set trainer.disaggregate_actor_n_gpus_per_node and "
                            "trainer.disaggregate_rollout_ref_n_gpus_per_node."
                        )
                    actor_gpus_per_node = total_gpus_per_node // 2
                    rollout_ref_gpus_per_node = total_gpus_per_node - actor_gpus_per_node
                elif actor_gpus_per_node is None:
                    rollout_ref_gpus_per_node = int(rollout_ref_gpus_per_node)
                    actor_gpus_per_node = total_gpus_per_node - rollout_ref_gpus_per_node
                elif rollout_ref_gpus_per_node is None:
                    actor_gpus_per_node = int(actor_gpus_per_node)
                    rollout_ref_gpus_per_node = total_gpus_per_node - actor_gpus_per_node
                else:
                    actor_gpus_per_node = int(actor_gpus_per_node)
                    rollout_ref_gpus_per_node = int(rollout_ref_gpus_per_node)

                if actor_gpus_per_node <= 0 or rollout_ref_gpus_per_node <= 0:
                    raise ValueError(
                        "Invalid fit_dis() GPU split: "
                        f"{actor_gpus_per_node=}, {rollout_ref_gpus_per_node=}"
                    )
                if actor_gpus_per_node + rollout_ref_gpus_per_node > total_gpus_per_node:
                    raise ValueError(
                        "fit_dis() GPU split exceeds available GPUs per node: "
                        f"{actor_gpus_per_node=}, {rollout_ref_gpus_per_node=}, {total_gpus_per_node=}"
                    )

                role_worker_mapping = {
                    Role.Actor: ray.remote(actor_rollout_cls),
                    Role.RolloutRef: ray.remote(actor_rollout_cls),
                    Role.Critic: ray.remote(CriticWorker),
                }
                resource_pool_spec = {
                    actor_pool_id: [actor_gpus_per_node] * config.trainer.nnodes,
                    rollout_ref_pool_id: [rollout_ref_gpus_per_node] * config.trainer.nnodes,
                }
                mapping = {
                    Role.Actor: actor_pool_id,
                    Role.RolloutRef: rollout_ref_pool_id,
                    # Critic/RM/reference are not supported yet in fit_dis() migration steps; keep mapping for
                    # compatibility (trainer will error if they are enabled).
                    Role.Critic: actor_pool_id,
                }
            else:
                role_worker_mapping = {
                    Role.ActorRollout: ray.remote(actor_rollout_cls),
                    Role.Critic: ray.remote(CriticWorker),
                }
                resource_pool_spec = {
                    global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
                }
                mapping = {
                    Role.ActorRollout: global_pool_id,
                    Role.Critic: global_pool_id,
                }
        elif placement == "disaggregated":
            if fit_disaggregate:
                raise NotImplementedError(
                    "trainer.disaggregate=true (fit_dis) is currently incompatible with TASK_PLACEMENT=disaggregated "
                    "(encoder/llm split placement)."
                )
            role_worker_mapping = {
                Role.LLMActorRollout: ray.remote(ActorRolloutRefWorker_llm),
                Role.EncoderActorRollout: ray.remote(ActorRolloutRefWorker_encoder),
                Role.Critic: ray.remote(CriticWorker),
            }
            resource_pool_spec = {
                actor_rollout_encoder_id: [2],
                actor_rollout_llm_id: [4],
                ref_encoder_id: [1],
                ref_llm_id: [1],
            }
            mapping = {
                Role.LLMActorRollout: actor_rollout_llm_id,
                Role.EncoderActorRollout: actor_rollout_encoder_id,
                Role.Critic: actor_rollout_llm_id,
            }
        
        # We should adopt a multi-source reward function here:vu
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # finally, we combine all the rewards together
        # The reward type depends on the tag of the data
        if config.reward_model.enable:
            if fit_disaggregate:
                raise NotImplementedError(
                    "reward_model.enable=true is currently unsupported with trainer.disaggregate=true (fit_dis)."
                )
            if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # Add a reference policy worker if KL loss or KL reward is used.
        if (config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss) and (not fit_disaggregate):
            if placement == "colocated":
                role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
                mapping[Role.RefPolicy] = global_pool_id
            elif placement == "disaggregated":
                role_worker_mapping[Role.EncoderRef] = ray.remote(ActorRolloutRefWorker_encoder)
                role_worker_mapping[Role.LLMRef] = ray.remote(ActorRolloutRefWorker_llm)
                mapping[Role.EncoderRef] = ref_encoder_id
                mapping[Role.LLMRef] = ref_llm_id

        # Load the reward manager for training and validation.
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        from verl.utils.dataset.rl_dataset import collate_fn

        # Create training and validation datasets.
        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor, is_train=True)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor, is_train=False)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # Initialize the PPO trainer.
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        if fit_disaggregate:
            trainer.fit_dis()
        else:
            # Initialize the workers of the trainer.
            trainer.init_workers()
            # Start the training process.
            trainer.fit()


def create_rl_dataset(data_paths, data_config, tokenizer, processor, is_train=True):
    """Create a dataset.

    Arguments:
        data_paths: List of paths to data files.
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """
    from torch.utils.data import Dataset

    from verl.utils.dataset.rl_dataset import RLHFDataset

    # Check if a custom dataset class is specified in the data configuration
    # and if the path to the custom class is provided
    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        # Dynamically load the custom dataset class
        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        # Verify that the custom dataset class inherits from torch.utils.data.Dataset
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(
                f"The custom dataset class '{data_config.custom_cls.name}' from "
                f"'{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset"
            )
    elif "datagen" in data_config and data_config.datagen.get("path", None) is not None and is_train:
        # If a data generation strategy is specified, use the DynamicGenDataset class
        from verl.utils.dataset.dynamicgen_dataset import DynamicGenDataset

        dataset_cls = DynamicGenDataset
        print("Using DynamicGenDataset for data generation.")

    else:
        # Use the default RLHFDataset class if no custom class is specified
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    # Instantiate the dataset using the determined dataset class
    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )

    return dataset


def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    if data_config.sampler is not None and data_config.sampler.get("class_path", None) is not None:
        curriculum_class = load_extern_type(
            data_config.sampler.class_path,
            data_config.sampler.class_name,
        )
        sampler = curriculum_class(
            data_source=dataset,
            data_config=data_config,
        )
        assert isinstance(sampler, AbstractSampler)
        assert data_config.get("dataloader_num_workers", 8) == 0, (
            "If using curriculum, num_workers must be 0 to prevent data caching. "
            "If the dataloader caches data before the batch is done the "
            "curriculum sampler won't have the opportunity to reorder it. "
        )

    # Use a sampler to facilitate checkpoint resumption.
    # If shuffling is enabled in the data configuration, create a random sampler.
    elif data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        # If shuffling is disabled, use a sequential sampler to iterate through the dataset in order.
        sampler = SequentialSampler(data_source=dataset)

    return sampler


if __name__ == "__main__":
    main()
