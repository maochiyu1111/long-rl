# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
import warnings
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Any, Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger

WorkerType = type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6
    EncoderRef = 7
    EncoderActorRollout = 8
    LLMRef = 9
    LLMActorRollout = 10
    RolloutRef = 11


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}"
                    + "cannot be satisfied in this ray cluster"
                )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", diffusion=False):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        diffusion (bool, optional): Whether this is for diffusion models. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    beta = kl_ctrl.value

    if diffusion:
        # For diffusion models: handle sequence-level KL penalty
        # In diffusion models, token_level_scores has shape (batch_size, 1)
        # and we use MSE-based KL between prev_sample_mean and ref_prev_sample_mean
        if "ref_prev_sample_mean" in data.batch and "prev_sample_mean" in data.batch:
            # Compute MSE-based KL for diffusion models
            prev_sample_mean = data.batch["prev_sample_mean"]
            ref_prev_sample_mean = data.batch["ref_prev_sample_mean"]
            
            # Compute MSE KL: (prev_sample_mean - ref_prev_sample_mean)^2 / (2 * std_dev^2)
            # For simplicity, we assume std_dev^2 = 1.0 here, but this could be made configurable
            kld = (prev_sample_mean - ref_prev_sample_mean) ** 2 / 2.0
            
            # Average over all dimensions except batch dimension
            kld = kld.mean(dim=tuple(range(1, kld.ndim)))
            
            # Apply KL penalty to sequence-level rewards
            token_level_rewards = token_level_scores - beta * kld.unsqueeze(-1)
            
            current_kl = torch.mean(kld, dim=0).item()
        else:
            # No reference data available, use original rewards
            token_level_rewards = token_level_scores
            current_kl = 0.0
    else:
        # For non-diffusion models: handle token-level KL penalty
        response_mask = data.batch["response_mask"]
        
        # compute kl between ref_policy and current policy
        # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
        kld = core_algos.kl_penalty(
            data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
        )  # (batch_size, response_length)
        kld = kld * response_mask
        
        token_level_rewards = token_level_scores - beta * kld
        
        current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
        current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


def compute_advantage_diffusion(data: DataProto, adv_estimator: AdvantageEstimator, gamma: float = 1.0, lam: float = 1.0) -> DataProto:
    """Compute advantage estimates for diffusion models using GRPO estimator.
    
    This function is specifically designed for diffusion models and uses the GRPO (Group Relative Policy Optimization)
    advantage estimator to compute advantages and returns for policy optimization in diffusion settings.
    
    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (should be GRPO for diffusion).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        
    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    token_level_rewards = data.batch["token_level_rewards"]  # (batch_size, 1)
    if adv_estimator == AdvantageEstimator.GRPO:
        batch_size = token_level_rewards.shape[0]
        response_mask = torch.ones((batch_size, 1), dtype=torch.bool)
        index = torch.zeros((batch_size), dtype=torch.long)
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards, response_mask, index)
    else:
        raise NotImplementedError

    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, and vLLM integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        # Trainer-level switches
        # - `diffusion` indicates whether to enable diffusion-specific codepaths.
        # - `diffusion_disaggregate` indicates whether to use fit_dis() topology (actor + rollout_ref split).
        self.diffusion = bool(getattr(config.trainer, "diffusion", False))
        self.diffusion_disaggregate = bool(getattr(config.trainer, "disaggregate", False))

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            if self.diffusion_disaggregate:
                required = {Role.Actor, Role.RolloutRef}
                missing = required.difference(role_worker_mapping.keys())
                assert not missing, (
                    "fit_dis() requires role_worker_mapping to include both Role.Actor and Role.RolloutRef; "
                    f"missing={sorted([m.name for m in missing])}, keys={list(role_worker_mapping.keys())}"
                )
            else:
                assert (
                    Role.ActorRollout in role_worker_mapping or Role.EncoderActorRollout in role_worker_mapping
                ), f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping or Role.EncoderRef in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.disaggregate_ref = Role.EncoderRef in role_worker_mapping
        self.disaggregate_actor_rollout = Role.EncoderActorRollout in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        if config.critic.enable is not None:
            self.use_critic = bool(config.critic.enable)
        elif self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            warnings.warn(
                "Disabled critic as algorithm.adv_estimator != gae. "
                "If it is not intended, please set critic.enable=True",
                stacklevel=2,
            )
            self.use_critic = False

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def init_workers_dis(self):
        """Initialize workers for fit_dis().
        """
        if self.disaggregate_actor_rollout or self.disaggregate_ref:
            raise NotImplementedError(
                "fit_dis() is currently incompatible with encoder/llm split placement "
                "(Role.Encoder*/Role.LLM*). Please use colocated placement for Step 1."
            )
        if not self.diffusion_disaggregate:
            raise ValueError("init_workers_dis() is only valid when trainer.disaggregate=true (fit_dis mode).")
        if self.use_critic:
            raise NotImplementedError("assumes no critic worker (GRPO). ")
        if self.use_rm:
            raise NotImplementedError(
                "assumes no reward model worker. "
            )
        if self.use_reference_policy:
            raise NotImplementedError(
                "expects reference to be handled by role='rollout_ref' workers. "
                "Do not create a separate ref worker in disaggregate mode."
            )

        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        actor_pool = self.resource_pool_manager.get_resource_pool(Role.Actor)
        rollout_ref_pool = self.resource_pool_manager.get_resource_pool(Role.RolloutRef)

        actor_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.Actor],
            config=self.config.actor_rollout_ref,
            role="actor",
            disaggregate=True,
            profile_option=self.config.trainer.npu_profile.options,
        )
        self.resource_pool_to_cls[actor_pool]["actor"] = actor_cls

        rollout_ref_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.RolloutRef],
            config=self.config.actor_rollout_ref,
            role="rollout_ref",
            disaggregate=True,
            profile_option=self.config.trainer.npu_profile.options,
        )
        self.resource_pool_to_cls[rollout_ref_pool]["rollout_ref"] = rollout_ref_cls

        wg_kwargs: dict[str, Any] = {}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.trainer, "profile_steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.trainer, "profile_steps")
            assert OmegaConf.select(self.config.trainer, "worker_nsight_options") is not None, (
                "worker_nsight_options must be set when profile_steps is set"
            )
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                OmegaConf.select(self.config.trainer, "worker_nsight_options")
            )
        wg_kwargs["device_name"] = self.device_name

        all_wg: dict[str, RayWorkerGroup] = {}
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            if not class_dict:
                continue
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        self.rollout_ref_wg = all_wg["rollout_ref"]
        self.actor_wg = all_wg["actor"]

        # Compatibility: existing colocated fit() codepaths still reference actor_rollout_wg.
        # In fit_dis(), rollout-related calls are expected to target rollout_ref_wg.
        self.actor_rollout_wg = self.rollout_ref_wg

        # `setup_dist()` uses torch.distributed rendezvous via MASTER_ADDR/MASTER_PORT.
        #
        # IMPORTANT:
        # - In multi-node Ray, the driver/head node IP may not be reachable by GPU workers (or head may have no GPUs),
        #   which can cause `dist.init_process_group(init_method="env://")` to hang indefinitely.
        # - Prefer using the worker-group rank-0 node ip + a free port picked by that worker (RayWorkerGroup does this
        #   internally via the register center actor).
        #
        # Allow explicit override via trainer.dist_master_addr/dist_master_port when the user has a known-good
        # rendezvous endpoint.
        dist_master_addr = OmegaConf.select(self.config.trainer, "dist_master_addr")
        dist_master_port = OmegaConf.select(self.config.trainer, "dist_master_port")

        master_addr = dist_master_addr or self.actor_wg.master_address
        master_port_raw = dist_master_port or self.actor_wg.master_port
        if not master_addr or master_port_raw is None:
            raise ValueError(
                "Failed to determine torch.distributed rendezvous endpoint. "
                "Set `trainer.dist_master_addr` and `trainer.dist_master_port` explicitly, or ensure the RayWorkerGroup "
                "has a valid `master_address/master_port`."
            )
        master_port = int(master_port_raw)

        n_actor = int(self.actor_wg.world_size)
        n_rollout_ref = int(self.rollout_ref_wg.world_size)
        world_size = n_actor + n_rollout_ref
        if world_size <= 0 or n_actor <= 0 or n_rollout_ref <= 0:
            raise ValueError(f"Invalid disaggregate world sizes: n_actor={n_actor}, n_rollout_ref={n_rollout_ref}")

        actor_ranks = list(range(0, n_actor))
        rollout_ref_ranks = list(range(n_actor, n_actor + n_rollout_ref))

        dist_timeout_s = int(OmegaConf.select(self.config.trainer, "dist_timeout_s") or 60)

        gpu_num_per_node = int(self.config.trainer.n_gpus_per_node)
        # Keep local rank assignment consistent with disco_rl migration baseline.
        actor_local = [idx % gpu_num_per_node for idx in range(n_actor)]
        roll_local = [idx % gpu_num_per_node for idx in range(n_rollout_ref)]

        refs: list[ray.ObjectRef] = []
        for idx, w in enumerate(self.actor_wg.workers):
            rank = actor_ranks[idx]
            ref = (
                w.actor_setup_dist.remote(
                    rank=rank,
                    world_size=world_size,
                    master_addr=master_addr,
                    master_port=master_port,
                    local_rank=actor_local[idx],
                    actor_group_ranks=actor_ranks,
                    rollout_ref_group_ranks=rollout_ref_ranks,
                )
            )
            refs.append(ref)
        for idx, w in enumerate(self.rollout_ref_wg.workers):
            rank = rollout_ref_ranks[idx]
            ref = (
                w.rollout_ref_setup_dist.remote(
                    rank=rank,
                    world_size=world_size,
                    master_addr=master_addr,
                    master_port=master_port,
                    local_rank=roll_local[idx],
                    actor_group_ranks=actor_ranks,
                    rollout_ref_group_ranks=rollout_ref_ranks,
                )
            )
            refs.append(ref)
        ray.get(refs, timeout=dist_timeout_s)

        # init_model() must happen after setup_dist() in diffusion+disaggregate mode.
        self.rollout_ref_wg.init_model()

        self.actor_wg.init_model()


    def fit_dis(self):
        """Disaggregate training entry (diffusion path)."""
        if self._is_dance_case3_enabled():
            reasons = self._dance_case3_mismatch_reasons()
            if reasons:
                reason_text = "; ".join(reasons)
                raise ValueError(
                    f"dance_case3_mode=true but Case3 conditions are not met ({reason_text}). "
                    "Expected: diffusion=true, disaggregate=true, pipelined_micro_batch=false, adv_estimator=grpo."
                )
            self.init_workers_dis()
            return self.fit_dance_case3_dis()

        self.init_workers_dis()
        if self.diffusion:
            pipeline_dtype = OmegaConf.select(
                self.config, "actor_rollout_ref.actor.fsdp_config.model_dtype"
            ) or "bfloat16"
            self.rollout_ref_wg.normalize_pipeline_dtype(dtype=str(pipeline_dtype))

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            if self.config.trainer.val_only:
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        self.data_iterator = iter(self.train_dataloader)
        fit_dis_epoch = 0

        # self._prof_start()
        while self.global_steps < self.total_training_steps:
            self.global_steps += 1
            metrics: dict[str, Any] = {}
            timing_raw: dict[str, float] = {}
            reward_result = None

            with marked_timer("step", timing_raw):
                with marked_timer("gen", timing_raw, color="red"):
                    if self.diffusion:
                        chunk_mb = int(OmegaConf.select(self.config, "trainer.disaggregate_sync_chunk_mb") or 256)
                        refs: list[ray.ObjectRef] = []
                        refs += self.rollout_ref_wg.execute_all_async(
                            "rollout_ref_sync_transformer_gdr_with_relay", chunk_mb=chunk_mb
                        )
                        refs += self.actor_wg.execute_all_async(
                            "actor_sync_transformer_gdr_with_relay", chunk_mb=chunk_mb
                        )
                        ray.get(refs)
                        batch = self._make_batch_data_dis(metrics=metrics)

                if not self.diffusion:
                    self._balance_batch(batch, metrics=metrics)
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                if "token_level_scores" not in batch.batch:
                    with marked_timer("reward", timing_raw, color="yellow"):
                        reward_result = compute_reward(batch, self.reward_fn)

                if not self.diffusion:
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_probs = self.actor_wg.compute_log_prob(batch)
                        if "entropys" in old_log_probs.batch:
                            old_log_probs.batch.pop("entropys")
                        batch = batch.union(old_log_probs)

                if self._need_ref_log_prob() or self.use_reference_policy:
                    with marked_timer("ref", timing_raw, color="olive"):
                        ref_log_probs = self.rollout_ref_wg.compute_ref_log_prob(batch)
                        batch = batch.union(ref_log_probs)

                if self.use_critic:
                    with marked_timer("values", timing_raw, color="cyan"):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                with marked_timer("adv", timing_raw, color="brown"):
                    if "token_level_scores" not in batch.batch:
                        reward_tensor, reward_metrics = reward_result
                        batch.batch["token_level_scores"] = reward_tensor
                        reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}
                        metrics.update(reward_metrics)

                    if self.config.algorithm.use_kl_in_reward:
                        batch, kl_metrics = apply_kl_penalty(
                            batch,
                            kl_ctrl=self.kl_ctrl_in_reward,
                            kl_penalty=self.config.algorithm.kl_penalty,
                            diffusion=self.diffusion,
                        )
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    if self.diffusion:
                        batch = compute_advantage_diffusion(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                        )
                    else:
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=self.config.algorithm.get(
                                "norm_adv_by_std_in_grpo", True
                            ),
                            config=self.config.algorithm,
                        )

                if self.use_critic:
                    with marked_timer("update_critic", timing_raw, color="pink"):
                        critic_output = self.critic_wg.update_critic(batch)
                    critic_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                    metrics.update(critic_metrics)

                if self.config.trainer.critic_warmup <= self.global_steps:
                    with marked_timer("update_actor", timing_raw, color="red"):
                        batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                        actor_output = self.actor_wg.update_actor(batch)
                    actor_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_metrics)

                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.val_freq > 0
                    and self.global_steps % self.config.trainer.val_freq == 0
                ):
                    with marked_timer("validation", timing_raw, color="green"):
                        val_metrics = self._validate()
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

            self._prof_step()
            self._log_step_timing(timing_raw=timing_raw, step=self.global_steps, epoch=fit_dis_epoch)
            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

            progress_bar.update(1)

        self._prof_stop()
        progress_bar.close()

        if self.val_reward_fn is not None:
            if self.config.trainer.val_freq <= 0 or self.global_steps % self.config.trainer.val_freq != 0:
                val_metrics = self._validate()
                pprint(f"Final validation metrics: {val_metrics}")

        if self.config.trainer.save_freq <= 0 or self.global_steps % self.config.trainer.save_freq != 0:
            self._save_checkpoint()

    def _make_batch_data_dis(self, metrics: dict[str, Any]) -> DataProto:
        batch = None
        all_metrics = defaultdict(list)
        num_try_make_batch = 0
        print("Start generating batch...")
        while True:
            num_try_make_batch += 1
            try:
                batch_dict = next(self.data_iterator)
            except StopIteration:
                self.data_iterator = iter(self.train_dataloader)
                batch_dict = next(self.data_iterator)

            meta_info = {
                "min_pixels": OmegaConf.select(self.config, "data.min_pixels"),
                "max_pixels": OmegaConf.select(self.config, "data.max_pixels"),
            }
            new_batch: DataProto = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
            if self.diffusion:
                required_keys = ["prompt_embeds", "negative_prompt_embeds"]
                missing_required = [k for k in required_keys if k not in new_batch.batch.keys()]
                if missing_required:
                    raise KeyError(
                        f"Diffusion batch missing required keys {missing_required}. "
                        f"available={list(new_batch.batch.keys())}"
                    )

                batch_keys = list(required_keys)
                for optional_key in ["pooled_prompt_embeds", "negative_pooled_prompt_embeds"]:
                    if optional_key in new_batch.batch.keys():
                        batch_keys.append(optional_key)
                gen_batch = new_batch.pop(batch_keys=batch_keys)
            else:
                gen_batch = new_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "ground_truth"],
                    meta_info_keys=["min_pixels", "max_pixels"],
                )

            gen_batch_output = self.rollout_ref_wg.generate_sequences(gen_batch)

            if self.config.algorithm.adv_estimator == "remax":
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["temperature"] = 0
                gen_baseline_batch.meta_info["n"] = 1
                gen_baseline_output = self.rollout_ref_wg.generate_sequences(gen_baseline_batch)

                new_batch = new_batch.union(gen_baseline_output)
                reward_baseline_tensor, _ = compute_reward(new_batch, self.reward_fn)
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                new_batch.batch["reward_baselines"] = reward_baseline_tensor
                del gen_baseline_batch, gen_baseline_output

            new_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
            )
            if not self.diffusion:
                new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
            new_batch = new_batch.union(gen_batch_output)

            if self.config.algorithm.online_filtering:
                reward_tensor, reward_metrics = compute_reward(new_batch, self.reward_fn)
                new_batch.batch["token_level_scores"] = reward_tensor
                for k, v in reward_metrics.items():
                    all_metrics[k].extend(v)

                filter_scores = reward_metrics[self.config.algorithm.filter_key]
                uids = new_batch.non_tensor_batch["uid"]
                uid2scores = defaultdict(list)
                for uid, score in zip(uids, filter_scores):
                    uid2scores[uid].append(score)

                uid2mean = {uid: np.mean(scores) for uid, scores in uid2scores.items()}
                kept_uids = [
                    uid
                    for uid, avg_score in uid2mean.items()
                    if avg_score > self.config.algorithm.filter_low and avg_score < self.config.algorithm.filter_high
                ]
                kept_sample_idxs = [idx for idx, uid in enumerate(uids) if uid in kept_uids]
                if len(kept_sample_idxs) > 0:
                    new_batch = new_batch[kept_sample_idxs]

            batch = DataProto.concat([batch, new_batch]) if batch is not None else new_batch
            rollout_repeat = 1 if self.diffusion else self.config.actor_rollout_ref.rollout.n
            current_batch_size = len(batch) // rollout_repeat
            rollout_batch_size = self.config.data.train_batch_size
            if current_batch_size < rollout_batch_size:
                print(f"{current_batch_size=} < {rollout_batch_size=}")
                max_try_make_batch = self.config.trainer.max_try_make_batch
                if max_try_make_batch <= 0 or num_try_make_batch < max_try_make_batch:
                    print(f"{num_try_make_batch=}. Continue generating...")
                else:
                    raise ValueError(
                        f"{num_try_make_batch=} >= {max_try_make_batch=}. Generated too many. Please check your data."
                    )
            else:
                print(f"{current_batch_size=} >= {rollout_batch_size=}. Finish generating.")
                if self.config.algorithm.online_filtering:
                    metrics.update({f"reward/{k}": v for k, v in reduce_metrics(all_metrics).items()})

                return batch[: self.config.data.train_batch_size * rollout_repeat]

    def _need_ref_log_prob(self) -> bool:
        return bool(self.config.algorithm.use_kl_in_reward or self.config.actor_rollout_ref.actor.use_kl_loss)

    def _sync_diffusion_disaggregate_before_rollout(self) -> None:
        """Sync actor->rollout_ref diffusion transformer weights via GDR relay.

        IMPORTANT: `sync_transformer_gdr_with_relay` is a global-collective method (Dispatch.ALL_TO_ALL) and must be
        scheduled on *both* WorkerGroups (actor + rollout_ref) before waiting, otherwise it can deadlock.
        """

        if not (self.diffusion and self.diffusion_disaggregate):
            return

        if not hasattr(self, "actor_wg") or not hasattr(self, "rollout_ref_wg"):
            raise RuntimeError("fit_dis() diffusion sync requires init_workers_dis() to be called first.")

        chunk_mb = int(OmegaConf.select(self.config, "trainer.disaggregate_sync_chunk_mb") or 256)
        refs: list[ray.ObjectRef] = []
        refs += self.rollout_ref_wg.execute_all_async("sync_transformer_gdr_with_relay", chunk_mb=chunk_mb)
        refs += self.actor_wg.execute_all_async("sync_transformer_gdr_with_relay", chunk_mb=chunk_mb)
        ray.get(refs)

    def _validate_config(self):
        config = self.config
        actor_cfg_raw = config.actor_rollout_ref.actor
        if hasattr(actor_cfg_raw, "keys") and "disco" in actor_cfg_raw.keys():
            raise ValueError(
                "actor_rollout_ref.actor.disco is obsolete and unsupported. "
                "Remove this key; Dance Case4 no longer uses it."
            )

        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = (
                config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size
                * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            )
            assert (
                n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0
            ), (
                f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times "
                f"context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            )
            megatron_dp = n_gpus // (
                model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size
            )
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size "
            f"({minimal_bsz})"
        )

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            """Validate mutually exclusive micro batch size configuration options.

            Ensures that users don't set both deprecated micro_batch_size and
            the new micro_batch_size_per_gpu parameters simultaneously.

            Args:
                mbs: Deprecated micro batch size parameter value.
                mbs_per_gpu: New micro batch size per GPU parameter value.
                name (str): Configuration section name for error messages.

            Raises:
                ValueError: If both parameters are set or neither is set.
            """
            settings = {
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'."
                    )

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                        f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                    )

        # Actor validation done in ActorConfig.__post_init__ and validate()
        actor_config = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
        actor_config.validate(n_gpus, config.data.train_batch_size, config.actor_rollout_ref.model)

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(
                config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model"
            )

        if self.config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic:
            critic_config = omega_conf_to_dataclass(config.critic)
            critic_config.validate(n_gpus, config.data.train_batch_size)

        if config.data.get("val_batch_size", None) is not None:
            print(
                "WARNING: val_batch_size is deprecated."
                + " Validation datasets are sent to inference engines as a whole batch,"
                + " which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, (
                "validation gen temperature should be greater than 0 when enabling do_sample"
            )

        # Diffusion-specific validation
        if self.diffusion:
            # Validate rollout name for diffusion
            if config.actor_rollout_ref.rollout.name != "diffusion":
                raise ValueError(
                    f"When trainer.diffusion=True, rollout.name must be 'diffusion', got '{config.actor_rollout_ref.rollout.name}'"
                )
            
            # Validate diffusion-specific parameters
            if config.actor_rollout_ref.rollout.num_steps is None:
                raise ValueError("When using diffusion models, rollout.num_steps must be specified")
            
            if config.actor_rollout_ref.rollout.num_steps <= 0:
                raise ValueError(f"rollout.num_steps must be positive, got {config.actor_rollout_ref.rollout.num_steps}")
            
            if config.actor_rollout_ref.rollout.guidance_scale is None:
                raise ValueError("When using diffusion models, rollout.guidance_scale must be specified")
            
            if config.actor_rollout_ref.rollout.guidance_scale < 1.0:
                raise ValueError(f"rollout.guidance_scale must be >= 1.0, got {config.actor_rollout_ref.rollout.guidance_scale}")
            
            # Validate resolution for SD3 image diffusion
            if config.actor_rollout_ref.rollout.resolution is not None:
                if not isinstance(config.actor_rollout_ref.rollout.resolution, int) or config.actor_rollout_ref.rollout.resolution <= 0:
                    raise ValueError(f"rollout.resolution must be a positive integer, got {config.actor_rollout_ref.rollout.resolution}")
            
            # Validate WAN video parameters
            if config.actor_rollout_ref.rollout.height is not None:
                if not isinstance(config.actor_rollout_ref.rollout.height, int) or config.actor_rollout_ref.rollout.height <= 0:
                    raise ValueError(f"rollout.height must be a positive integer, got {config.actor_rollout_ref.rollout.height}")
            
            if config.actor_rollout_ref.rollout.width is not None:
                if not isinstance(config.actor_rollout_ref.rollout.width, int) or config.actor_rollout_ref.rollout.width <= 0:
                    raise ValueError(f"rollout.width must be a positive integer, got {config.actor_rollout_ref.rollout.width}")
            
            if config.actor_rollout_ref.rollout.num_frames is not None:
                if not isinstance(config.actor_rollout_ref.rollout.num_frames, int) or config.actor_rollout_ref.rollout.num_frames <= 0:
                    raise ValueError(f"rollout.num_frames must be a positive integer, got {config.actor_rollout_ref.rollout.num_frames}")
            
            # Validate actor configuration for diffusion
            if not getattr(config.actor_rollout_ref.actor, 'extra', {}).get('diffusion', False):
                raise ValueError(
                    "When trainer.diffusion=True, actor.config.extra.diffusion must be True"
                )
            
            # Validate that we're not using incompatible settings
            if config.actor_rollout_ref.rollout.do_sample is False:
                raise ValueError("Diffusion models require rollout.do_sample=True")
            
            # Check that log_prob computation settings are compatible
            if config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu is None and config.actor_rollout_ref.rollout.log_prob_micro_batch_size is None:
                raise ValueError("For diffusion models, either rollout.log_prob_micro_batch_size_per_gpu or rollout.log_prob_micro_batch_size must be specified")

        print("[validate_config] All configuration checks passed successfully!")

    def _set_total_training_steps(self) -> None:
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _create_dance_latent_dataloader(self, mode_name: str) -> None:
        from fastvideo.dataset.latent_rl_datasets import LatentDataset, latent_collate_function

        data_json_path = self.config.data.get("data_json_path", None)
        if data_json_path is None:
            raise ValueError(f"{mode_name}=true requires `data.data_json_path`")

        num_latent_t = int(self.config.data.get("t", 1))
        cfg_rate = float(self.config.data.get("cfg", 0.0))
        num_workers = int(self.config.data.get("dataloader_num_workers", 0))
        train_batch_size = int(self.config.data.get("gen_batch_size", self.config.data.get("train_batch_size", 1)))

        self.train_dataset = LatentDataset(data_json_path, num_latent_t=num_latent_t, cfg_rate=cfg_rate)
        self.val_dataset = None
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=train_batch_size,
            num_workers=num_workers,
            drop_last=True,
            collate_fn=latent_collate_function,
            sampler=None,
        )
        self.val_dataloader = None

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        print(f"Size of train dataloader: {len(self.train_dataloader)}, validation disabled for {mode_name}")
        self._set_total_training_steps()

    def _create_dance_case3_dataloader(self) -> None:
        self._create_dance_latent_dataloader("dance_case3_mode")

    def _create_dance_case4_dataloader(self) -> None:
        self._create_dance_latent_dataloader("dance_case4_mode")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        if self._is_dance_case3_enabled():
            self._create_dance_case3_dataloader()
            return
        if self._is_dance_case4_enabled():
            self._create_dance_case4_dataloader()
            return

        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor
            )
        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataset, self.val_dataset = train_dataset, val_dataset
        num_workers = int(self.config.data.get("dataloader_num_workers", 0))
        train_batch_size = int(self.config.data.get("gen_batch_size", self.config.data.get("train_batch_size", 1)))

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=train_batch_size,
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.get("val_batch_size", None)  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )
        self._set_total_training_steps()

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        sample_turns = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            if not self.diffusion:
                test_batch = test_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
                )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            if self.diffusion:
                # For diffusion models, handle differently based on what's available
                if "text" in test_batch.non_tensor_batch:
                    input_texts = test_batch.non_tensor_batch["text"].tolist()
                else:
                    # Fallback to placeholder if no text prompts available
                    input_texts = [f"Diffusion prompt {i}" for i in range(len(test_batch))]
            else:
                input_ids = test_batch.batch["input_ids"]
                # TODO: Can we keep special tokens except for padding tokens?
                input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            if self.diffusion:
                required_keys = ["prompt_embeds", "negative_prompt_embeds"]
                missing_required = [k for k in required_keys if k not in test_batch.batch.keys()]
                if missing_required:
                    raise KeyError(
                        f"Diffusion val batch missing required keys {missing_required}. "
                        f"available={list(test_batch.batch.keys())}"
                    )

                batch_keys = list(required_keys)
                for optional_key in ["pooled_prompt_embeds", "negative_pooled_prompt_embeds"]:
                    if optional_key in test_batch.batch.keys():
                        batch_keys.append(optional_key)
                test_gen_batch = test_batch.pop(batch_keys=batch_keys)
                test_gen_batch.meta_info = self.config.actor_rollout_ref.rollout.val_kwargs
            else:
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in test_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in test_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in test_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "interaction_kwargs" in test_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("interaction_kwargs")
                if "agent_name" in test_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("agent_name")
                test_gen_batch = test_batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )
                test_gen_batch.meta_info = {
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "recompute_log_prob": False,
                    "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                    "validate": True,
                    "global_steps": self.global_steps,
                }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            if self.diffusion:
                # For diffusion, directly generate without padding/unpadding
                if not self.async_rollout_mode:
                    test_output_gen_batch = self.actor_rollout_wg.generate_sequences(test_gen_batch)
                else:
                    test_output_gen_batch = self.async_rollout_manager.generate_sequences(test_gen_batch)
            else:
                if self.disaggregate_actor_rollout:
                    size_divisor = (
                        self.actor_rollout_llm_wg.world_size
                        if not self.async_rollout_mode
                        else self.config.actor_rollout_ref_llm_wg.rollout.agent.num_workers
                    )
                else:
                    size_divisor = (
                        self.actor_rollout_wg.world_size
                        if not self.async_rollout_mode
                        else self.config.actor_rollout_ref.rollout.agent.num_workers
                    )
                test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
                if not self.async_rollout_mode:
                    test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
                else:
                    test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

                # unpad
                test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            if self.diffusion:
                # For diffusion models, handle differently based on what's available
                # Assume generated images are available in the batch
                if "responses" in test_output_gen_batch.batch:
                    # If responses exist, treat them as generated outputs
                    output_ids = test_output_gen_batch.batch["responses"]
                    output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
                else:
                    # Fallback to placeholder if no generated outputs available
                    output_texts = [f"Diffusion output {i}" for i in range(len(test_output_gen_batch))]
            else:
                output_ids = test_output_gen_batch.batch["responses"]
                output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            if not self.diffusion:
                test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)
            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
                    print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def _make_batch_data(self, metrics: dict[str, Any]) -> DataProto:
        """Create batch data with diffusion support."""
        batch = None
        all_metrics = defaultdict(list)
        num_try_make_batch = 0
        print("Start generating batch...")
        while True:
            num_try_make_batch += 1
            try:
                batch_dict = next(self.data_iterator)
            except StopIteration:
                self.data_iterator = iter(self.train_dataloader)
                batch_dict = next(self.data_iterator)

            meta_info = {
                "min_pixels": OmegaConf.select(self.config, "data.min_pixels"),
                "max_pixels": OmegaConf.select(self.config, "data.max_pixels"),
            }
            new_batch: DataProto = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
            
            if self.diffusion:
                required_keys = ["prompt_embeds", "negative_prompt_embeds"]
                missing_required = [k for k in required_keys if k not in new_batch.batch.keys()]
                if missing_required:
                    raise KeyError(
                        f"Diffusion batch missing required keys {missing_required}. "
                        f"available={list(new_batch.batch.keys())}"
                    )

                batch_keys = list(required_keys)
                for optional_key in ["pooled_prompt_embeds", "negative_pooled_prompt_embeds"]:
                    if optional_key in new_batch.batch.keys():
                        batch_keys.append(optional_key)
                gen_batch = new_batch.pop(batch_keys=batch_keys)
            else:
                # pop those keys for generation
                gen_batch = new_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "ground_truth"],
                    meta_info_keys=["min_pixels", "max_pixels"],
                )

            # generate a batch
            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

            if self.config.algorithm.adv_estimator == "remax":
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["temperature"] = 0
                gen_baseline_batch.meta_info["n"] = 1
                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                new_batch = new_batch.union(gen_baseline_output)
                reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                new_batch.batch["reward_baselines"] = reward_baseline_tensor
                del gen_baseline_batch, gen_baseline_output

            new_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
            )
            # repeat to align with repeated responses in rollout
            if not self.diffusion:
                new_batch = new_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
            new_batch = new_batch.union(gen_batch_output)

            # filter group
            if self.config.algorithm.online_filtering:
                reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                new_batch.batch["token_level_scores"] = reward_tensor
                for k, v in reward_metrics.items():
                    all_metrics[k].extend(v)

                filter_scores = reward_metrics[self.config.algorithm.filter_key]
                uids = new_batch.non_tensor_batch["uid"]
                uid2scores = defaultdict(list)
                for uid, score in zip(uids, filter_scores):
                    uid2scores[uid].append(score)

                uid2mean = {uid: np.mean(scores) for uid, scores in uid2scores.items()}
                kept_uids = [
                    uid
                    for uid, avg_score in uid2mean.items()
                    if avg_score > self.config.algorithm.filter_low and avg_score < self.config.algorithm.filter_high
                ]
                kept_sample_idxs = [idx for idx, uid in enumerate(uids) if uid in kept_uids]
                if len(kept_sample_idxs) > 0:
                    new_batch = new_batch[kept_sample_idxs]

            batch = DataProto.concat([batch, new_batch]) if batch is not None else new_batch
            rollout_repeat = 1 if self.diffusion else self.config.worker.rollout.n
            current_batch_size = len(batch) // rollout_repeat
            rollout_batch_size = self.config.data.train_batch_size
            if current_batch_size < rollout_batch_size:
                print(f"{current_batch_size=} < {rollout_batch_size=}")
                max_try_make_batch = self.config.trainer.max_try_make_batch
                if max_try_make_batch <= 0 or num_try_make_batch < max_try_make_batch:
                    print(f"{num_try_make_batch=}. Continue generating...")
                else:
                    raise ValueError(
                        f"{num_try_make_batch=} >= {max_try_make_batch=}. Generated too many. Please check your data."
                    )
            else:
                print(f"{current_batch_size=} >= {rollout_batch_size=}. Finish generating.")
                if self.config.algorithm.online_filtering:
                    metrics.update({f"reward/{k}": v for k, v in reduce_metrics(all_metrics).items()})

                return batch[: self.config.data.train_batch_size * rollout_repeat]

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            if self.disaggregate_actor_rollout:
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.EncoderActorRollout)
                encoder_ref_cls =  RayClassWithInitArgs(
                    cls=self.role_worker_mapping[Role.EncoderActorRollout],
                    config=self.config.actor_rollout_ref,
                    role="encoder_actor_rollout",
                    # profile_option=self.config.trainer.npu_profile.options,
                )
                self.resource_pool_to_cls[resource_pool]["encoder_actor_rollout"] = encoder_ref_cls
                
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.LLMActorRollout)
                llm_ref_cls =  RayClassWithInitArgs(
                    cls=self.role_worker_mapping[Role.LLMActorRollout],
                    config=self.config.actor_rollout_ref,
                    role="llm_actor_rollout",
                    # profile_option=self.config.trainer.npu_profile.options,
                )
                self.resource_pool_to_cls[resource_pool]["llm_actor_rollout"] = llm_ref_cls                
            else:
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
                actor_rollout_cls = RayClassWithInitArgs(
                    cls=self.role_worker_mapping[Role.ActorRollout],
                    config=self.config.actor_rollout_ref,
                    role="actor_rollout",
                    disaggregate=getattr(self, "diffusion_disaggregate", False),
                    profile_option=self.config.trainer.npu_profile.options,
                )
                self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # 在这里还要做修改，在fit里面的ref_wg改成ref_encoder_wg，需要在这里做修改
        # create reference policy if needed
        if self.use_reference_policy:
            if self.disaggregate_ref:
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.EncoderRef)
                encoder_ref_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.EncoderRef], config=self.config.actor_rollout_ref, role="encoder_ref")
                self.resource_pool_to_cls[resource_pool]["encoder_ref"] = encoder_ref_cls
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.LLMRef)
                llm_ref_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.LLMRef], config=self.config.actor_rollout_ref, role="llm_ref")
                self.resource_pool_to_cls[resource_pool]["llm_ref"] = llm_ref_cls
            else:
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
                ref_policy_cls = RayClassWithInitArgs(
                    self.role_worker_mapping[Role.RefPolicy],
                    config=self.config.actor_rollout_ref,
                    role="ref",
                    disaggregate=getattr(self, "diffusion_disaggregate", False),
                    profile_option=self.config.trainer.npu_profile.options,
                )
                self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.trainer, "profile_steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.trainer, "profile_steps")
            assert OmegaConf.select(self.config.trainer, "worker_nsight_options") is not None, (
                "worker_nsight_options must be set when profile_steps is set"
            )
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                OmegaConf.select(self.config.trainer, "worker_nsight_options")
            )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            if self.disaggregate_ref:
                self.ref_encoder_wg = all_wg["encoder_ref"]
                self.ref_llm_wg = all_wg["llm_ref"]
                self.ref_encoder_wg.init_model()
                self.ref_llm_wg.init_model()
            else:
                self.ref_policy_wg = all_wg["ref"]
                self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        if self.disaggregate_actor_rollout:
            self.actor_rollout_encoder_wg = all_wg["encoder_actor_rollout"]
            self.actor_rollout_llm_wg = all_wg["llm_actor_rollout"]
            self.actor_rollout_encoder_wg.init_model()
            self.actor_rollout_llm_wg.init_model()
        else:
            self.actor_rollout_wg = all_wg["actor_rollout"]
            n_gpus_video_init = OmegaConf.select(self.config, "trainer.n_gpus_video_init", default=0)
            if n_gpus_video_init is None:
                n_gpus_video_init = 0
            try:
                n_gpus_video_init = int(n_gpus_video_init)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"trainer.n_gpus_video_init must be an integer, got {n_gpus_video_init!r}") from exc
            serial_init_model = bool(
                OmegaConf.select(self.config, "trainer.serial_init_model", default=False)
            ) or os.environ.get("SERIAL_INIT_MODEL", "0").strip().lower() in {"1", "true", "yes", "on"}
            env_n_gpus_video_init = os.environ.get("N_GPUS_VIDEO_INIT")
            if env_n_gpus_video_init:
                try:
                    n_gpus_video_init = int(env_n_gpus_video_init)
                except ValueError as exc:
                    raise ValueError(
                        f"N_GPUS_VIDEO_INIT must be an integer, got {env_n_gpus_video_init!r}"
                    ) from exc

            if n_gpus_video_init < 0:
                raise ValueError(f"n_gpus_video_init must be non-negative, got {n_gpus_video_init}")

            if serial_init_model and n_gpus_video_init <= 0:
                n_gpus_video_init = 1

            if n_gpus_video_init == 1:
                self.actor_rollout_wg.execute_all_serial_sync("init_model")
            elif 1 < n_gpus_video_init < self.actor_rollout_wg.world_size:
                self.actor_rollout_wg.execute_all_batched_sync("init_model", n_gpus_video_init)
            else:
                self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        actor_ckpt_wg = self.actor_wg if getattr(self, "diffusion_disaggregate", False) else self.actor_rollout_wg
        actor_ckpt_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        actor_ckpt_wg = self.actor_wg if getattr(self, "diffusion_disaggregate", False) else self.actor_rollout_wg
        actor_ckpt_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            if getattr(self, "diffusion_disaggregate", False):
                if self.rollout_ref_wg is self.actor_wg:
                    self.rollout_ref_wg.prof_start()
                else:
                    self.rollout_ref_wg.prof_start()
                    self.actor_wg.prof_start()
            elif self.disaggregate_actor_rollout:
                self.actor_rollout_encoder_wg.start_profile()
                self.actor_rollout_llm_wg.start_profile()
            else:
                #self.actor_rollout_wg.start_profile()
                self.actor_rollout_wg.prof_start()
            if self.use_reference_policy:
                if self.disaggregate_ref:
                    self.ref_encoder_wg.start_profile()
                    self.ref_llm_wg.start_profile()
                else:
                    #self.ref_policy_wg.start_profile()
                    pass
            if self.use_critic:
                self.critic_wg.start_profile()
            if self.use_rm:
                self.rm_wg.start_profile()

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            if getattr(self, "diffusion_disaggregate", False):
                if self.rollout_ref_wg is self.actor_wg:
                    self.rollout_ref_wg.prof_stop()
                else:
                    self.rollout_ref_wg.prof_stop()
                    self.actor_wg.prof_stop()
            elif self.disaggregate_actor_rollout:
                self.actor_rollout_encoder_wg.stop_profile()
                self.actor_rollout_llm_wg.stop_profile()
            else:
                self.actor_rollout_wg.prof_stop()
            if self.use_reference_policy:
                if self.disaggregate_ref:
                    self.ref_encoder_wg.stop_profile()
                    self.ref_llm_wg.stop_profile()
                else:
                    #self.ref_policy_wg.stop_and_save()
                    pass
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _step_profiling(self, do_profile: bool) -> None:
        if do_profile:
            if getattr(self, "diffusion_disaggregate", False):
                if self.rollout_ref_wg is self.actor_wg:
                    self.rollout_ref_wg.prof_step()
                else:
                    self.rollout_ref_wg.prof_step()
                    self.actor_wg.prof_step()
            elif self.disaggregate_actor_rollout:
                self.actor_rollout_encoder_wg.step_profile()
                self.actor_rollout_llm_wg.step_profile()
            else:
                self.actor_rollout_wg.prof_step()
            if self.use_reference_policy:
                if self.disaggregate_ref:
                    self.ref_encoder_wg.step_profile()
                    self.ref_llm_wg.step_profile()
                else:
                    #self.ref_policy_wg.stop_and_save()
                    pass
            if self.use_critic:
                self.critic_wg.step_profile()
            if self.use_rm:
                self.rm_wg.step_profile()

    def _prof_start(self):
        if self.disaggregate_actor_rollout:
            self.actor_rollout_encoder_wg.prof_start()
            self.actor_rollout_llm_wg.prof_start()
        else:
            self.actor_rollout_wg.prof_start()
        if self.use_reference_policy:
            if self.disaggregate_ref:
                self.ref_encoder_wg.prof_start()
                self.ref_llm_wg.prof_start()
            else:
                # self.ref_policy_wg.prof_start()
                pass
        if self.use_critic:
            self.critic_wg.prof_start()
        if self.use_rm:
            self.rm_wg.prof_start()

    def _prof_step(self):
        if self.disaggregate_actor_rollout:
            self.actor_rollout_encoder_wg.prof_step()
            self.actor_rollout_llm_wg.prof_step()
        else:
            self.actor_rollout_wg.prof_step()
        if self.use_reference_policy:
            if self.disaggregate_ref:
                self.ref_encoder_wg.prof_step()
                self.ref_llm_wg.prof_step()
            else:
                # self.ref_policy_wg.prof_step()
                pass
        if self.use_critic:
            self.critic_wg.prof_step()
        if self.use_rm:
            self.rm_wg.prof_step()

    def _prof_stop(self):
        if self.disaggregate_actor_rollout:
            self.actor_rollout_encoder_wg.prof_stop()
            self.actor_rollout_llm_wg.prof_stop()
        else:
            self.actor_rollout_wg.prof_stop()
        if self.use_reference_policy:
            if self.disaggregate_ref:
                self.ref_encoder_wg.prof_stop()
                self.ref_llm_wg.prof_stop()
            else:
                # self.ref_policy_wg.prof_stop()
                pass
        if self.use_critic:
            self.critic_wg.prof_stop()
        if self.use_rm:
            self.rm_wg.prof_stop()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        if self.disaggregate_actor_rollout:
            world_size = self.actor_rollout_llm_wg.world_size
        else:
            world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _log_step_timing(self, timing_raw: dict[str, float], step: int, epoch: int):
        """Print per-step timing to help debug large variance between steps."""
        stage_order = [
            "gen",
            "gen_max",
            "reward",
            "old_log_prob",
            "ref",
            "values",
            "adv",
            "update_critic",
            "update_actor",
            "dump_rollout_generations",
            "testing",
            "save_checkpoint",
        ]
        step_time = timing_raw.get("step")
        stage_msgs = [f"{name}={timing_raw[name]:.4f}s" for name in stage_order if name in timing_raw]
        extra_names = [
            name
            for name in timing_raw.keys()
            if name
            not in set(stage_order + ["step", "start_profile", "stop_profile"])
        ]
        for name in sorted(extra_names):
            stage_msgs.append(f"{name}={timing_raw[name]:.4f}s")

        if step_time is not None:
            header = f"[timing] epoch={epoch} step={step} total={step_time:.4f}s"
        else:
            header = f"[timing] epoch={epoch} step={step}"
        if stage_msgs:
            print(f"{header} | " + ", ".join(stage_msgs))
        else:
            print(header)

    def _is_dance_case3_enabled(self) -> bool:
        return bool(self.config.actor_rollout_ref.actor.get("dance_case3_mode", False))

    def _dance_case3_mismatch_reasons(self) -> list[str]:
        reasons: list[str] = []
        if not self.diffusion:
            reasons.append("trainer.diffusion must be true")
        if not self.diffusion_disaggregate:
            reasons.append("trainer.disaggregate must be true")
        if bool(self.config.trainer.get("pipelined_micro_batch", False)):
            reasons.append("trainer.pipelined_micro_batch must be false")

        adv_estimator = str(self.config.algorithm.adv_estimator).lower()
        if adv_estimator not in {"grpo", "advantageestimator.grpo"}:
            reasons.append("algorithm.adv_estimator must be grpo")
        return reasons

    def _is_dance_case3_mode(self) -> bool:
        return self._is_dance_case3_enabled() and len(self._dance_case3_mismatch_reasons()) == 0

    def _is_dance_case4_enabled(self) -> bool:
        return bool(self.config.actor_rollout_ref.actor.get("dance_case4_mode", False))

    def _dance_case4_mismatch_reasons(self) -> list[str]:
        reasons: list[str] = []
        if not self.diffusion:
            reasons.append("trainer.diffusion must be true")
        if self.diffusion_disaggregate:
            reasons.append("trainer.disaggregate must be false")
        if bool(self.config.trainer.get("pipelined_micro_batch", False)):
            reasons.append("trainer.pipelined_micro_batch must be false")

        adv_estimator = str(self.config.algorithm.adv_estimator).lower()
        if adv_estimator not in {"grpo", "advantageestimator.grpo"}:
            reasons.append("algorithm.adv_estimator must be grpo")
        return reasons

    def _is_dance_case4_mode(self) -> bool:
        return self._is_dance_case4_enabled() and len(self._dance_case4_mismatch_reasons()) == 0

    def fit_dance_case3_dis(self):
        max_train_steps = self.config.trainer.get("max_train_steps", None)
        if max_train_steps is None:
            max_train_steps = self.total_training_steps
        max_train_steps = int(max_train_steps)
        if max_train_steps <= 0:
            raise ValueError(f"Invalid max_train_steps for dance case3: {max_train_steps}")

        self.global_steps = 0
        data_iterator = iter(self.train_dataloader)
        progress_bar = tqdm(total=max_train_steps, initial=self.global_steps, desc="Dance Case3")

        while self.global_steps < max_train_steps:
            try:
                batch = next(data_iterator)
            except StopIteration:
                data_iterator = iter(self.train_dataloader)
                batch = next(data_iterator)

            if not isinstance(batch, (list, tuple)) or len(batch) != 3:
                raise ValueError(
                    "dance_case3_mode requires dataloader to return "
                    "(encoder_hidden_states, encoder_attention_mask, caption)"
                )

            encoder_hidden_states, encoder_attention_mask, caption = batch
            new_batch = DataProto.from_single_dict(
                {
                    "encoder_hidden_states": encoder_hidden_states,
                    "encoder_attention_mask": encoder_attention_mask,
                },
                meta_info={"caption": caption},
            )

            self._sync_diffusion_disaggregate_before_rollout()
            rollout_batch = self.rollout_ref_wg.generate_sequences(new_batch)
            actor_output = self.actor_wg.update_actor(rollout_batch)
            actor_metrics = actor_output.meta_info.get("metrics", {}) if actor_output is not None else {}
            if actor_metrics:
                progress_bar.set_postfix({k: f"{v:.4f}" for k, v in actor_metrics.items() if isinstance(v, (int, float))})

            self.global_steps += 1
            progress_bar.update(1)

        progress_bar.close()
        return None

    def fit_dance_case4(self):
        max_train_steps = self.config.trainer.get("max_train_steps", None)
        if max_train_steps is None:
            max_train_steps = self.total_training_steps
        max_train_steps = int(max_train_steps)
        if max_train_steps <= 0:
            raise ValueError(f"Invalid max_train_steps for dance case4: {max_train_steps}")

        self.global_steps = 0
        data_iterator = iter(self.train_dataloader)
        progress_bar = tqdm(total=max_train_steps, initial=self.global_steps, desc="Dance Case4")

        while self.global_steps < max_train_steps:
            try:
                batch = next(data_iterator)
            except StopIteration:
                data_iterator = iter(self.train_dataloader)
                batch = next(data_iterator)

            if not isinstance(batch, (list, tuple)) or len(batch) != 3:
                raise ValueError(
                    "dance_case4_mode requires dataloader to return "
                    "(encoder_hidden_states, encoder_attention_mask, caption)"
                )

            encoder_hidden_states, encoder_attention_mask, caption = batch
            new_batch = DataProto.from_single_dict(
                {
                    "encoder_hidden_states": encoder_hidden_states,
                    "encoder_attention_mask": encoder_attention_mask,
                },
                meta_info={"caption": caption},
            )

            rollout_batch = self.actor_rollout_wg.generate_sequences(new_batch)
            actor_output = self.actor_rollout_wg.update_actor(rollout_batch)
            actor_metrics = actor_output.meta_info.get("metrics", {}) if actor_output is not None else {}
            if actor_metrics:
                progress_bar.set_postfix({k: f"{v:.4f}" for k, v in actor_metrics.items() if isinstance(v, (int, float))})

            self.global_steps += 1
            progress_bar.update(1)

        progress_bar.close()
        return None

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        if self._is_dance_case4_enabled():
            reasons = self._dance_case4_mismatch_reasons()
            if reasons:
                reason_text = "; ".join(reasons)
                raise ValueError(
                    f"dance_case4_mode=true but Case4 conditions are not met ({reason_text}). "
                    "Expected: diffusion=true, disaggregate=false, pipelined_micro_batch=false, adv_estimator=grpo."
                )
            return self.fit_dance_case4()

        # from verl.utils.tracking import Tracking

        # logger = Tracking(
        #     project_name=self.config.trainer.project_name,
        #     experiment_name=self.config.trainer.experiment_name,
        #     default_backend=self.config.trainer.logger,
        #     config=OmegaConf.to_container(self.config, resolve=True),
        # )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        # if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
        #     val_metrics = self._validate()
        #     assert val_metrics, f"{val_metrics=}"
        #     pprint(f"Initial validation metrics: {val_metrics}")
        #     logger.log(data=val_metrics, step=self.global_steps)
        #     if self.config.trainer.get("val_only", False):
        #         return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.trainer.profile_steps
            if self.config.trainer.profile_steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            
            self._prof_start()
            #self.actor_rollout_wg.prof_start()

            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.trainer.profile_continuous_steps
                        else curr_step_profile
                    )

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                if self.diffusion:
                    # diffusion 分支：直接使用嵌入，不做文本 token pop/重复
                    required_batch_keys = ["prompt_embeds", "negative_prompt_embeds"]
                    missing_required = [key for key in required_batch_keys if key not in batch.batch]
                    if missing_required:
                        raise KeyError(f"Diffusion batch missing required keys: {missing_required}")

                    optional_batch_keys = ["pooled_prompt_embeds", "negative_pooled_prompt_embeds"]
                    batch_keys_to_pop = required_batch_keys + [
                        key for key in optional_batch_keys if key in batch.batch
                    ]
                    gen_batch = batch.pop(batch_keys=batch_keys_to_pop)
                    gen_batch.meta_info["global_steps"] = self.global_steps
                else:
                    # 文本分支：pop 文本 token，按 rollout.n 重复
                    batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                    non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                    if "multi_modal_data" in batch.non_tensor_batch:
                        non_tensor_batch_keys_to_pop.append("multi_modal_data")
                    if "raw_prompt" in batch.non_tensor_batch:
                        non_tensor_batch_keys_to_pop.append("raw_prompt")
                    if "tools_kwargs" in batch.non_tensor_batch:
                        non_tensor_batch_keys_to_pop.append("tools_kwargs")
                    if "interaction_kwargs" in batch.non_tensor_batch:
                        non_tensor_batch_keys_to_pop.append("interaction_kwargs")
                    if "index" in batch.non_tensor_batch:
                        non_tensor_batch_keys_to_pop.append("index")
                    if "agent_name" in batch.non_tensor_batch:
                        non_tensor_batch_keys_to_pop.append("agent_name")

                    if self.disaggregate_actor_rollout:
                        modality_dict = {"multi_modal_inputs": batch.non_tensor_batch["multi_modal_inputs"]}
                        modality_batch = DataProto.from_dict(non_tensors=modality_dict)
                        modality_batch = modality_batch.repeat(
                            repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                        )

                    gen_batch = batch.pop(
                        batch_keys=batch_keys_to_pop,
                        non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                    )

                    # pass global_steps to trace
                    gen_batch.meta_info["global_steps"] = self.global_steps
                    gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            if self.disaggregate_actor_rollout and not self.diffusion:
                                modality_embeddings = self.actor_rollout_encoder_wg.rollout_forward(modality_batch)
                                gen_batch.non_tensor_batch["multi_modal_data"] = modality_embeddings.non_tensor_batch["multi_modal_data"]
                                gen_batch_output = self.actor_rollout_llm_wg.generate_sequences(gen_batch)
                            else:
                                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    if not self.diffusion:
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if not self.diffusion and "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if not self.diffusion and self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens（仅文本路径需要）
                    if not self.diffusion:
                        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # Handle reward computation based on diffusion mode
                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # recompute old_log_probs
                    # For diffusion rollouts, old_log_probs are already produced alongside trajectories.
                    # Recomputing them here can diverge from the stored values (e.g., different schedulers),
                    # which leads to union conflicts. Keep the rollout-provided log-probs intact.
                    if not self.diffusion:
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            # old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            if self.disaggregate_actor_rollout:
                                encoder_results = self.actor_rollout_encoder_wg.compute_log_prob_encoder(batch)
                                batch = batch.union_non_tensor(encoder_results)
                                old_log_prob = self.actor_rollout_llm_wg.compute_log_prob_llm(batch)
                                for key in encoder_results.non_tensor_batch.keys():
                                    batch.non_tensor_batch.pop(key)
                            else:
                                old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)

                            # Original non-diffusion entropy handling
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")

                            batch = batch.union(old_log_prob)

                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                rollout_old_log_probs = batch.batch["rollout_log_probs"]
                                actor_old_log_probs = batch.batch["old_log_probs"]

                                # Original non-diffusion logic
                                attention_mask = batch.batch["attention_mask"]
                                responses = batch.batch["responses"]
                                response_length = responses.size(1)
                                response_mask = attention_mask[:, -response_length:]

                                rollout_probs = torch.exp(rollout_old_log_probs)
                                actor_probs = torch.exp(actor_old_log_probs)
                                rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                                rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                                rollout_probs_diff_max = torch.max(rollout_probs_diff)
                                rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                                rollout_probs_diff_std = torch.std(rollout_probs_diff)

                                metrics.update(
                                    {
                                        "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                        "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                        "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                    }
                                )

                    if self._need_ref_log_prob():
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                if self.disaggregate_ref:
                                    encoder_results = self.ref_encoder_wg.compute_ref_log_prob_encoder(batch)
                                    batch = batch.union_non_tensor(encoder_results)
                                    ref_log_prob = self.ref_llm_wg.compute_ref_log_prob_llm(batch)
                                else:
                                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            
                            # Handle diffusion-specific reference log probability data
                            if self.diffusion:
                                # For diffusion models, reference log prob might include additional data
                                # like ref_prev_sample_mean for KL regularization
                                if "ref_prev_sample_mean" in ref_log_prob.batch:
                                    # Store this for potential KL computation in advantage calculation
                                    batch.batch["ref_prev_sample_mean"] = ref_log_prob.batch["ref_prev_sample_mean"]
                            
                            batch = batch.union(ref_log_prob)

                            if self.disaggregate_ref:
                                for key in encoder_results.non_tensor_batch.keys():
                                    batch.non_tensor_batch.pop(key)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        
                        # Handle different reward tensor structures for diffusion vs non-diffusion
                        if self.diffusion:
                            # For diffusion: reward_tensor shape is (batch_size, 1)
                            batch.batch["token_level_scores"] = reward_tensor
                        else:
                            # For non-diffusion: reward_tensor shape is (batch_size, sequence_length)
                            batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty, diffusion=self.diffusion
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        # Use diffusion-specific advantage computation if enabled
                        if self.diffusion:
                            batch = compute_advantage_diffusion(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                            )
                        else:
                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                config=self.config.algorithm,
                            )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)
                    
                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            if self.disaggregate_actor_rollout:
                            
                                encoder_embed = self.actor_rollout_encoder_wg.actor_forward(batch)
                                batch = batch.union_non_tensor(encoder_embed)
                                actor_output = self.actor_rollout_llm_wg.update_actor(batch)
                                encoder_input = batch.pop(non_tensor_batch_keys=["multi_modal_inputs"])
                                actor_output = self.actor_rollout_encoder_wg.update_actor(actor_output, encoder_input)
                            else:
                                actor_output = self.actor_rollout_wg.update_actor(batch)
                            # expecting None to be DataProto, but got <class 'NoneType'>                
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            if "request_id" in batch.non_tensor_batch:
                                reward_extra_infos_dict.setdefault(
                                    "request_id",
                                    batch.non_tensor_batch["request_id"].tolist(),
                                )
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    self._prof_step()
                    #self.actor_rollout_wg.prof_step()
                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    # Check if the conditions for saving a checkpoint are met.
                    # The conditions include a mandatory condition (1) and
                    # one of the following optional conditions (2/3/4):
                    # 1. The save frequency is set to a positive value.
                    # 2. It's the last training step.
                    # 3. The current step number is a multiple of the save frequency.
                    # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                    ):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()
                
                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.trainer.profile_steps
                        if self.config.trainer.profile_steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.trainer.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                self._log_step_timing(timing_raw=timing_raw, step=self.global_steps, epoch=epoch)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                # logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    self._prof_stop()
                    #self.actor_rollout_wg.prof_stop()
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
