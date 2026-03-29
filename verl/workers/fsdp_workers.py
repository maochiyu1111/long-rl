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
The main entry point to run the PPO algorithm
"""

import json
import logging
import math
import os
import warnings
from datetime import timedelta
from dataclasses import asdict
from typing import Any

import numpy as np
import psutil
import torch
import torch.distributed
import torch.distributed as dist
from codetiming import Timer
from omegaconf import DictConfig, OmegaConf, open_dict
from peft import LoraConfig, TaskType, get_peft_model
from safetensors.torch import save_file
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.activation_offload import enable_activation_offloading
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import (
    get_device_id,
    get_device_name,
    get_nccl_backend,
    get_torch_device,
    is_cuda_available,
    is_npu_available,
)
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    layered_summon_lora_params,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.utils.import_utils import import_external_libs
from verl.utils.model import compute_position_id_with_mask
from verl.utils.profiler import Profiler, DistProfiler, DistProfilerExtension, log_gpu_memory_usage, simple_timer
from verl.utils.profiler.performance import reduce_timing
from verl.utils.py_functional import convert_to_regular_types, dict_to
from verl.workers.config import FSDPCriticConfig, FSDPEngineConfig
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


def _get_dist_timeout() -> timedelta:
    timeout_s = int(os.getenv("VERL_DIST_TIMEOUT_SECONDS", "60"))
    if timeout_s <= 0:
        timeout_s = 60
    return timedelta(seconds=timeout_s)

def _pick_iface_by_subnet(prefix: str = "192.158.0.") -> str | None:
    import socket

    for name, addrs in psutil.net_if_addrs().items():
        for a in addrs:
            if a.family == socket.AF_INET and a.address.startswith(prefix):
                return name
    return None


def _setup_nic_env(prefix: str | None = None) -> None:
    # Respect explicit user settings first. If only one backend is specified,
    # mirror it to the other so Gloo/NCCL stay on the same NIC.
    nccl_iface = os.environ.get("NCCL_SOCKET_IFNAME")
    gloo_iface = os.environ.get("GLOO_SOCKET_IFNAME")
    if nccl_iface or gloo_iface:
        if nccl_iface and not gloo_iface:
            os.environ["GLOO_SOCKET_IFNAME"] = nccl_iface
        elif gloo_iface and not nccl_iface:
            os.environ["NCCL_SOCKET_IFNAME"] = gloo_iface
        logger.debug(
            "[net] keep preconfigured NCCL/GLOO iface, NCCL=%s GLOO=%s",
            os.environ.get("NCCL_SOCKET_IFNAME"),
            os.environ.get("GLOO_SOCKET_IFNAME"),
        )
        return

    prefix = prefix or os.environ.get("VERL_SOCKET_IFACE_PREFIX", "192.158.0.")
    iface = _pick_iface_by_subnet(prefix)
    if iface is None:
        os.environ["NCCL_SOCKET_IFNAME"] = "^lo,docker0,flannel,cni0,veth"
        os.environ["GLOO_SOCKET_IFNAME"] = os.environ["NCCL_SOCKET_IFNAME"]
        logger.debug("[net] no iface with %s; use exclude list for NCCL/GLOO", prefix)
    else:
        os.environ["NCCL_SOCKET_IFNAME"] = iface
        os.environ["GLOO_SOCKET_IFNAME"] = iface
        logger.debug("[net] use iface %s for NCCL/GLOO", iface)


def _bcast_cuda_chunks_into_(flat_tensor: torch.Tensor, *, src_group_rank: int, group, chunk_mb: int = 256) -> None:
    """Broadcast a 1D device tensor in chunks, writing into `flat_tensor` in-place on receivers."""

    if flat_tensor.device.type == "cpu":
        raise ValueError("_bcast_cuda_chunks_into_ requires a non-CPU tensor")
    if flat_tensor.ndim != 1:
        raise ValueError(f"_bcast_cuda_chunks_into_ requires a 1D tensor, got shape={tuple(flat_tensor.shape)}")
    if chunk_mb <= 0:
        raise ValueError(f"chunk_mb must be positive, got {chunk_mb}")

    numel = flat_tensor.numel()
    bytes_per_el = flat_tensor.element_size()
    chunk_elems = max(1, (chunk_mb * 1024 * 1024) // bytes_per_el)
    offset = 0
    while offset < numel:
        end = min(offset + chunk_elems, numel)
        dist.broadcast(flat_tensor[offset:end], src=src_group_rank, group=group)
        offset = end


def _bcast_cuda_chunks_maybe_cast_(
    *,
    src_flat_tensor: torch.Tensor | None = None,
    dst_flat_tensor: torch.Tensor | None = None,
    src_group_rank: int,
    group,
    chunk_mb: int = 256,
    wire_dtype: torch.dtype | None = None,
) -> None:
    """Broadcast a 1D CUDA tensor in chunks, optionally casting on the sender chunk-by-chunk."""

    if (src_flat_tensor is None) == (dst_flat_tensor is None):
        raise ValueError("Exactly one of src_flat_tensor or dst_flat_tensor must be provided")

    ref_tensor = src_flat_tensor if src_flat_tensor is not None else dst_flat_tensor
    assert ref_tensor is not None
    if ref_tensor.device.type == "cpu":
        raise ValueError("_bcast_cuda_chunks_maybe_cast_ requires a non-CPU tensor")
    if ref_tensor.ndim != 1:
        raise ValueError(
            "_bcast_cuda_chunks_maybe_cast_ requires a 1D tensor, "
            f"got shape={tuple(ref_tensor.shape)}"
        )
    if chunk_mb <= 0:
        raise ValueError(f"chunk_mb must be positive, got {chunk_mb}")

    wire_dtype = wire_dtype or ref_tensor.dtype
    if dst_flat_tensor is not None and dst_flat_tensor.dtype != wire_dtype:
        raise ValueError(f"Receiver dtype mismatch: expected {wire_dtype}, got {dst_flat_tensor.dtype}")

    # Casting on the sender allocates a temporary tensor, so keep those chunks small.
    effective_chunk_mb = min(chunk_mb, 8) if wire_dtype != ref_tensor.dtype or src_flat_tensor is None else chunk_mb
    bytes_per_el = torch.empty((), dtype=wire_dtype, device=ref_tensor.device).element_size()
    chunk_elems = max(1, (effective_chunk_mb * 1024 * 1024) // bytes_per_el)

    numel = ref_tensor.numel()
    offset = 0
    while offset < numel:
        end = min(offset + chunk_elems, numel)
        if src_flat_tensor is not None:
            chunk = src_flat_tensor[offset:end]
            if chunk.dtype != wire_dtype:
                chunk = chunk.to(dtype=wire_dtype)
            dist.broadcast(chunk, src=src_group_rank, group=group)
        else:
            dist.broadcast(dst_flat_tensor[offset:end], src=src_group_rank, group=group)
        offset = end


class _NoOpProfiler:
    def start(self, **kwargs) -> None:
        return None

    def stop(self) -> None:
        return None

    def step(self) -> None:
        return None

    def stop_and_save(self) -> None:
        return None


def create_device_mesh(world_size, fsdp_size):
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh(device_name, mesh_shape=(world_size,), mesh_dim_names=["fsdp"])
    else:
        device_mesh = init_device_mesh(
            device_name, mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=["ddp", "fsdp"]
        )
    return device_mesh


def create_device_mesh_from_ranks(ranks: list[int], fsdp_size: int) -> DeviceMesh:
    world_size = len(ranks)
    if world_size == 0:
        raise ValueError("ranks must be non-empty")

    ranks = sorted(ranks)
    if fsdp_size < 0 or fsdp_size >= world_size:
        mesh = ranks
        mesh_dim_names = ("fsdp",)
    else:
        if world_size % fsdp_size != 0:
            raise ValueError(f"world_size={world_size} must be divisible by fsdp_size={fsdp_size}")
        ddp_size = world_size // fsdp_size
        mesh = [ranks[i * fsdp_size : (i + 1) * fsdp_size] for i in range(ddp_size)]
        mesh_dim_names = ("ddp", "fsdp")

    return DeviceMesh(device_name, mesh=mesh, mesh_dim_names=mesh_dim_names)


def create_ulysses_device_mesh_from_ranks(ranks: list[int], sp_size: int) -> DeviceMesh | None:
    if sp_size <= 1:
        return None

    world_size = len(ranks)
    if world_size == 0:
        raise ValueError("ranks must be non-empty")
    if world_size % sp_size != 0:
        raise ValueError(f"world_size={world_size} must be divisible by sp_size={sp_size}")

    ranks = sorted(ranks)
    dp_size = world_size // sp_size
    mesh = [ranks[i * sp_size : (i + 1) * sp_size] for i in range(dp_size)]
    return DeviceMesh(device_name, mesh=mesh, mesh_dim_names=("dp", "sp"))


def get_sharding_strategy(device_mesh):
    from torch.distributed.fsdp import ShardingStrategy

    if device_mesh.ndim == 1:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif device_mesh.ndim == 2:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise NotImplementedError(f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2")
    return sharding_strategy


class ActorRolloutRefWorker(Worker, DistProfilerExtension):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str, disaggregate: bool | None = None, **kwargs):
        Worker.__init__(self)

        self.config = config
        self.profile_option = kwargs.get("profile_option", None)
        self.role = role
        assert self.role in ["actor", "rollout", "ref", "actor_rollout", "rollout_ref", "actor_rollout_ref"]

        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in ["rollout", "actor_rollout", "rollout_ref", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "rollout_ref", "actor_rollout_ref"]

        self._prof = None
        self._prof_enabled = False
        self._prof_active = False
        self._prof_logdir = os.getenv("PROF_LOGDIR", "/workspace/yym/RLHF/verl-disaggregate/log/trace/col")
        self._enable_prof_env = bool(int(os.getenv("ENABLE_PROFILER", "0")))
        self.generation_config = None

        self.ulysses_sequence_parallel_size = self.config.actor.get("ulysses_sequence_parallel_size", 1)
        self._lora_rank = self.config.model.get("lora_rank", 0)
        self._is_lora = self._lora_rank > 0

        # diffusion flag is configured via actor.extra / rollout.name / trainer.diffusion
        actor_extra = self.config.actor.get("extra", {})
        trainer_diffusion = OmegaConf.select(self.config, "trainer.diffusion")
        if trainer_diffusion is None:
            trainer_diffusion = getattr(self.config, "diffusion", False)
        self.diffusion = bool(
            bool(trainer_diffusion)
            or getattr(self.config.rollout, "name", "") == "diffusion"
            or getattr(self.config.ref, "diffusion", False)
            or (actor_extra.get("diffusion", False) if isinstance(actor_extra, dict) else False)
        )
        trainer_disaggregate = OmegaConf.select(self.config, "trainer.disaggregate")
        if trainer_disaggregate is None:
            trainer_disaggregate = getattr(self.config, "disaggregate", False)
        self.disaggregate = bool(trainer_disaggregate) if disaggregate is None else bool(disaggregate)
        self._local_model_path = None

        # Profiler relies on the default process group for rank discovery. In disaggregate mode we may
        # intentionally delay process group initialization, so we install a no-op profiler first and
        # replace it after distributed setup.
        self._profiler_config = omega_conf_to_dataclass(config.get("profiler"))
        DistProfilerExtension.__init__(self, _NoOpProfiler())

        import torch.distributed

        self._post_dist_init_done = False
        self.device_mesh = None
        self.ulysses_device_mesh = None
        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        self.global_rank: int | None = None
        self.global_world_size: int | None = None
        self.actor_group_ranks: list[int] | None = None
        self.rollout_ref_group_ranks: list[int] | None = None
        self.actor_pg = None
        self.rollout_ref_pg = None
        self._gdr_pair_ranks: tuple[int, int] | None = None
        self._gdr_pair_pg = None
        self._dist_mesh_ranks: list[int] | None = None
        self._delay_default_pg_init = bool(self.diffusion and self.disaggregate and self.role in ["actor", "rollout_ref"])
        if self._delay_default_pg_init and torch.distributed.is_initialized():
            raise RuntimeError(
                "disaggregate+diffusion worker requires delayed default process group initialization, but "
                "torch.distributed is already initialized before setup_dist()."
            )

        if (not self._delay_default_pg_init) and (not torch.distributed.is_initialized()):
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            _setup_nic_env()
            torch.distributed.init_process_group(
                backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}",
                rank=rank,
                world_size=world_size,
                init_method=os.environ.get("DIST_INIT_METHOD", None),
                timeout=_get_dist_timeout(),
            )

        if torch.distributed.is_initialized():
            self._init_dist_dependent_state()

        self._is_offload_param = False
        self._is_offload_optimizer = False
        if self._is_actor:
            self._is_offload_param = self.config.actor.fsdp_config.get("param_offload", False)
            self._is_offload_optimizer = self.config.actor.fsdp_config.get("optimizer_offload", False)
        elif self._is_ref:
            # TODO: it seems that manual offload is slowly than FSDP offload
            self._is_offload_param = self.config.ref.fsdp_config.get("param_offload", False)

    def _init_dist_dependent_state(self) -> None:
        if self._post_dist_init_done:
            return
        if not dist.is_initialized():
            raise RuntimeError("default process group is not initialized; call setup_dist() before using this worker.")

        if isinstance(getattr(self, "profiler", None), _NoOpProfiler):
            self.profiler = Profiler(config=self._profiler_config, task=self.role)

        mesh_ranks = self._dist_mesh_ranks
        if mesh_ranks is None:
            world_size = dist.get_world_size()
            self.device_mesh = create_device_mesh(
                world_size=world_size, fsdp_size=self.config.actor.fsdp_config.fsdp_size
            )
            self.ulysses_device_mesh = None
            dp = world_size // self.ulysses_sequence_parallel_size
            if self.ulysses_sequence_parallel_size > 1:
                self.ulysses_device_mesh = init_device_mesh(
                    device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
                )
        else:
            world_size = len(mesh_ranks)
            self.device_mesh = create_device_mesh_from_ranks(
                ranks=mesh_ranks, fsdp_size=self.config.actor.fsdp_config.fsdp_size
            )
            self.ulysses_device_mesh = create_ulysses_device_mesh_from_ranks(
                ranks=mesh_ranks, sp_size=self.ulysses_sequence_parallel_size
            )

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # normalize config
        if self._is_actor:
            self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
            self.config.actor.ppo_mini_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
            assert self.config.actor.ppo_mini_batch_size > 0, (
                f"ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be larger than 0 after "
                f"normalization"
            )
            # micro bsz
            if self.config.actor.ppo_micro_batch_size is not None:
                self.config.actor.ppo_micro_batch_size //= (
                    self.device_mesh.size() // self.ulysses_sequence_parallel_size
                )
                self.config.actor.ppo_micro_batch_size_per_gpu = self.config.actor.ppo_micro_batch_size

            if self.config.actor.ppo_micro_batch_size_per_gpu is not None:
                assert self.config.actor.ppo_mini_batch_size % self.config.actor.ppo_micro_batch_size_per_gpu == 0, (
                    f"normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be divisible by "
                    f"ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}"
                )
                assert self.config.actor.ppo_mini_batch_size // self.config.actor.ppo_micro_batch_size_per_gpu > 0, (
                    f"normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be larger than "
                    f"ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}"
                )

        # normalize rollout config
        if self._is_rollout and self.config.rollout.log_prob_micro_batch_size is not None:
            self.config.rollout.log_prob_micro_batch_size //= (
                self.device_mesh.size() // self.ulysses_sequence_parallel_size
            )
            self.config.rollout.log_prob_micro_batch_size_per_gpu = self.config.rollout.log_prob_micro_batch_size
        # normalize ref config
        if self._is_ref and self.config.ref.log_prob_micro_batch_size is not None:
            self.config.ref.log_prob_micro_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
            self.config.ref.log_prob_micro_batch_size_per_gpu = self.config.ref.log_prob_micro_batch_size

        self._post_dist_init_done = True

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def setup_dist(
        self,
        *,
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: int,
        local_rank: int = 0,
        actor_group_ranks: list[int] | None = None,
        rollout_ref_group_ranks: list[int] | None = None,
    ) -> None:
        # 设 env，使用 env:// rendezvous
        _setup_nic_env()
        os.environ["MASTER_ADDR"] = str(master_addr)
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(local_rank)

        os.environ["NCCL_DEBUG"] = os.environ.get("NCCL_DEBUG", "INFO")
        os.environ["NCCL_IB_DISABLE"] = "1"  # 强制禁用 RDMA
        os.environ["NCCL_NET"] = "Socket"  # 强制走 TCP
        os.environ["NCCL_COLLNET_ENABLE"] = "0"  # 关 SHARP/CollNet 等 IB 相关
        os.environ["NCCL_SHARP_DISABLE"] = "1"
        visible_gpu_count = torch.cuda.device_count()
        target_device = 0
        if visible_gpu_count > 1:
            target_device = int(local_rank) % visible_gpu_count
        torch.cuda.set_device(target_device)
        logger.debug(
            "[setup_dist] global_rank=%s local_rank=%s visible_gpu_count=%s cuda_visible_devices=%s target_device=%s",
            rank,
            local_rank,
            visible_gpu_count,
            os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
            target_device,
        )
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://", timeout=_get_dist_timeout())

        # 验证：这里打印的一定是全局 rank
        self.global_rank = rank
        self.actor_group_ranks = actor_group_ranks
        self.rollout_ref_group_ranks = rollout_ref_group_ranks
        self.actor_pg = (
            dist.new_group(ranks=actor_group_ranks) if actor_group_ranks is not None and len(actor_group_ranks) > 0 else None
        )
        self.rollout_ref_pg = (
            dist.new_group(ranks=rollout_ref_group_ranks)
            if rollout_ref_group_ranks is not None and len(rollout_ref_group_ranks) > 0
            else None
        )

        if self._is_actor:
            self._dist_mesh_ranks = actor_group_ranks
        if self._is_rollout or self._is_ref:
            self._dist_mesh_ranks = rollout_ref_group_ranks

        if not self._post_dist_init_done:
            self._init_dist_dependent_state()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def normalize_pipeline_dtype(self, module: str = "transformer", dtype: str = "bfloat16") -> None:
        """Normalize diffusion rollout pipeline dtype before cross-group sync.

        This is a no-op unless this worker includes a rollout role and has a diffusion pipeline.
        """

        if not self._is_rollout:
            return

        rollout = getattr(self, "rollout", None)
        pipeline = getattr(rollout, "pipeline", None)
        if pipeline is not None:
            pipeline_module = getattr(pipeline, module, None)
            if pipeline_module is None:
                raise AttributeError(f"rollout.pipeline has no module '{module}'")
        elif self._is_dance_case3_mode():
            pipeline_module = rollout
            if pipeline_module is None:
                raise RuntimeError("normalize_pipeline_dtype() requires dance case3 rollout to be initialized.")
        else:
            raise RuntimeError("normalize_pipeline_dtype() requires a rollout with a `pipeline` attribute.")

        from verl.utils.torch_dtypes import PrecisionType

        target_dtype = PrecisionType.to_dtype(dtype)

        with torch.no_grad():
            for param in pipeline_module.parameters(recurse=True):
                if param is None or param.data is None:
                    continue
                if not param.data.is_floating_point():
                    continue
                if param.data.dtype != target_dtype:
                    param.data = param.data.to(dtype=target_dtype)

            for submodule in pipeline_module.modules():
                for buffer_name, buffer in list(submodule._buffers.items()):
                    if buffer is None:
                        continue
                    if not torch.is_floating_point(buffer):
                        continue
                    if buffer.dtype != target_dtype:
                        submodule._buffers[buffer_name] = buffer.to(dtype=target_dtype)

    def _get_rollout_sync_module(self):
        rollout = getattr(self, "rollout", None)
        if rollout is None:
            raise RuntimeError("rollout is not initialized")

        pipeline = getattr(rollout, "pipeline", None)
        if pipeline is not None and getattr(pipeline, "transformer", None) is not None:
            return pipeline.transformer
        if self._is_dance_case3_mode():
            return rollout
        raise RuntimeError("rollout.pipeline or pipeline.transformer doesn't exist")

    @register(dispatch_mode=Dispatch.ALL_TO_ALL)
    def sync_transformer_gdr_with_relay(self, chunk_mb: int = 256) -> None:
        if not self.diffusion:
            return

        if self.global_rank is None or self.actor_group_ranks is None or self.rollout_ref_group_ranks is None:
            raise RuntimeError("sync_transformer_gdr_with_relay() requires setup_dist() to be called first.")

        world_rank = int(self.global_rank)
        if not self.actor_group_ranks or not self.rollout_ref_group_ranks:
            raise RuntimeError("sync_transformer_gdr_with_relay() requires non-empty actor/rollout_ref groups.")

        actor_src_global_rank = self.actor_group_ranks[0]
        relay_global_rank = self.rollout_ref_group_ranks[0]
        pair_ranks = [actor_src_global_rank, relay_global_rank]
        pair_pg = dist.new_group(ranks=pair_ranks)

        pipeline_transformer = None
        if world_rank in self.rollout_ref_group_ranks:
            pipeline_transformer = self._get_rollout_sync_module()

        logger.debug(
            "[GDR-Relay][Phase0][enter] world_rank=%s actor_src=%s relay=%s local_rank=%s cuda_device=%s pair_ranks=%s",
            world_rank,
            actor_src_global_rank,
            relay_global_rank,
            os.environ.get("LOCAL_RANK", "unset"),
            get_device_id(),
            tuple(pair_ranks),
        )

        target_dtype = None
        if world_rank == relay_global_rank:
            assert pipeline_transformer is not None, "relay has to have rollout.pipeline"

            try:
                p0 = next(pipeline_transformer.parameters())
            except StopIteration as exc:
                raise RuntimeError("pipeline has no parameters") from exc
            target_dtype = p0.dtype
            for p in pipeline_transformer.parameters():
                if p.dtype != target_dtype:
                    raise RuntimeError(f"pipeline has inconsistent dtype: {p.dtype} vs {target_dtype}")

            dtype_token = str(target_dtype).replace("torch.", "")
            obj = [dtype_token]
            dist.broadcast_object_list(obj, src=relay_global_rank, group=pair_pg)
            logger.debug("[GDR-Relay][Phase0][relay_done] world_rank=%s dtype=%s", world_rank, dtype_token)

        elif world_rank == actor_src_global_rank:
            obj = [None]
            dist.broadcast_object_list(obj, src=relay_global_rank, group=pair_pg)
            dtype_token = obj[0]
            if dtype_token is None:
                raise RuntimeError("relay did not provide dtype token")
            try:
                target_dtype = getattr(torch, dtype_token)
            except AttributeError as exc:
                raise RuntimeError(f"can't find dtype: {dtype_token}") from exc
            logger.debug("[GDR-Relay][Phase0][actor_done] world_rank=%s dtype=%s", world_rank, dtype_token)

        if world_rank in self.actor_group_ranks:
            actor_module_fsdp = getattr(self, "actor_module_fsdp", None)
            if actor_module_fsdp is None:
                raise RuntimeError("sync_transformer_gdr_with_relay() requires actor_module_fsdp to be initialized.")

            if self._is_offload_param:
                load_fsdp_model_to_gpu(actor_module_fsdp)

            try:
                summon_rank0_only = actor_src_global_rank == 0
                with FSDP.summon_full_params(
                    actor_module_fsdp,
                    writeback=False,
                    rank0_only=summon_rank0_only,
                    offload_to_cpu=False,
                ):
                    if world_rank == actor_src_global_rank:
                        named_params = list(actor_module_fsdp.named_parameters())
                        obj = [len(named_params)]
                        dist.broadcast_object_list(obj, src=actor_src_global_rank, group=pair_pg)
                        dist.barrier(pair_pg)

                        for name, param in named_params:
                            tensor = param.data
                            if not tensor.is_cuda:
                                raise RuntimeError(f"actor params are not on cuda {name}")
                            if not tensor.is_contiguous():
                                raise RuntimeError(f"actor params are not contiguous {name}")

                            meta = (name, tuple(tensor.shape))
                            obj = [meta]
                            dist.broadcast_object_list(obj, src=actor_src_global_rank, group=pair_pg)
                            dist.barrier(pair_pg)
                            _bcast_cuda_chunks_maybe_cast_(
                                src_flat_tensor=tensor.view(-1),
                                src_group_rank=actor_src_global_rank,
                                group=pair_pg,
                                chunk_mb=chunk_mb,
                                wire_dtype=target_dtype,
                            )
                            dist.barrier(pair_pg)
            finally:
                if self._is_offload_param:
                    offload_fsdp_model_to_cpu(actor_module_fsdp)

        elif world_rank == relay_global_rank:
            assert pipeline_transformer is not None
            obj = [None]
            dist.broadcast_object_list(obj, src=actor_src_global_rank, group=pair_pg)
            dist.barrier(pair_pg)
            num_tensors = obj[0]
            if num_tensors is None:
                raise RuntimeError("relay did not receive tensor count from actor_src")

            name2param = dict(pipeline_transformer.named_parameters())
            names_in_order = []
            for _ in range(num_tensors):
                obj = [None]
                dist.broadcast_object_list(obj, src=actor_src_global_rank, group=pair_pg)
                dist.barrier(pair_pg)
                name, shape = obj[0]
                names_in_order.append(name)

                if name not in name2param:
                    raise RuntimeError(f"relay lacks {name}")
                param = name2param[name]
                if tuple(param.shape) != tuple(shape):
                    raise RuntimeError(f"relay shape mapping error {name} recv={tuple(param.shape)} src={tuple(shape)}")
                if not param.data.is_cuda:
                    raise RuntimeError(f"relay params are not on cuda {name}")
                if not param.data.is_contiguous():
                    raise RuntimeError(f"relay params are not contiguous {name}")

                _bcast_cuda_chunks_maybe_cast_(
                    dst_flat_tensor=param.data.view(-1),
                    src_group_rank=actor_src_global_rank,
                    group=pair_pg,
                    chunk_mb=chunk_mb,
                    wire_dtype=target_dtype,
                )
                dist.barrier(pair_pg)

        if world_rank in self.rollout_ref_group_ranks:
            if self.rollout_ref_pg is None:
                raise RuntimeError("rollout_ref_pg is not initialized; call setup_dist() before sync.")

            if world_rank == relay_global_rank:
                assert pipeline_transformer is not None
                name2param = dict(pipeline_transformer.named_parameters())

                if "names_in_order" not in locals():
                    names_in_order = list(name2param.keys())

                obj = [len(names_in_order)]
                dist.broadcast_object_list(obj, src=relay_global_rank, group=self.rollout_ref_pg)
                dist.barrier(self.rollout_ref_pg)

                for name in names_in_order:
                    p = name2param[name]
                    meta = (name, tuple(p.shape))
                    obj = [meta]
                    dist.broadcast_object_list(obj, src=relay_global_rank, group=self.rollout_ref_pg)
                    dist.barrier(self.rollout_ref_pg)
                    flat_send = p.data.view(-1)
                    _bcast_cuda_chunks_into_(
                        flat_send, src_group_rank=relay_global_rank, group=self.rollout_ref_pg, chunk_mb=chunk_mb
                    )
                    dist.barrier(self.rollout_ref_pg)

            else:
                assert pipeline_transformer is not None, "rollout rank must have pipeline"
                name2param = dict(pipeline_transformer.named_parameters())

                obj = [None]
                dist.broadcast_object_list(obj, src=relay_global_rank, group=self.rollout_ref_pg)
                dist.barrier(self.rollout_ref_pg)
                num_tensors = obj[0]
                if num_tensors is None:
                    raise RuntimeError("rollout did not receive tensor count from relay")
                for _ in range(num_tensors):
                    obj = [None]
                    dist.broadcast_object_list(obj, src=relay_global_rank, group=self.rollout_ref_pg)
                    dist.barrier(self.rollout_ref_pg)
                    name, shape = obj[0]
                    if name not in name2param:
                        raise RuntimeError(f"rollout lacks {name}")
                    param = name2param[name]
                    if tuple(param.shape) != tuple(shape):
                        raise RuntimeError(
                            f"rollout shape mapping error {name} recv={tuple(param.shape)} src={tuple(shape)}"
                        )
                    if not param.data.is_cuda:
                        raise RuntimeError(f"rollout params are not on cuda {name}")
                    if not param.data.is_contiguous():
                        raise RuntimeError(f"rollout params are not contiguous {name}")

                    flat_recv = param.data.view(-1)
                    _bcast_cuda_chunks_into_(
                        flat_recv, src_group_rank=relay_global_rank, group=self.rollout_ref_pg, chunk_mb=chunk_mb
                    )
                    dist.barrier(self.rollout_ref_pg)

        if world_rank == 0:
            logger.debug("[GDR-Relay] pipeline sync finished relay=%s, chunk=%sMB", relay_global_rank, chunk_mb)

    def _get_local_model_path(self):
        """Memoized copy_to_local to avoid repeated downloads when diffusion is enabled."""
        if self._local_model_path is None:
            self._local_model_path = copy_to_local(
                self.config.model.path, use_shm=self.config.model.get("use_shm", False)
            )
        return self._local_model_path

    def _resolve_diffusion_scheduler_path(self, local_model_path: str) -> str:
        actor_extra = self.config.actor.get("extra", {}) if hasattr(self.config, "actor") else {}
        scheduler_path = None
        if hasattr(actor_extra, "get"):
            scheduler_path = actor_extra.get("diffusion_scheduler", actor_extra.get("scheduler", None))
        if scheduler_path is None and hasattr(self.config.actor, "diffusion_scheduler"):
            scheduler_path = getattr(self.config.actor, "diffusion_scheduler")
        if scheduler_path is None and hasattr(self.config.ref, "scheduler"):
            scheduler_path = getattr(self.config.ref, "scheduler")
        if scheduler_path is None and hasattr(self.config.ref, "diffusion_scheduler"):
            scheduler_path = getattr(self.config.ref, "diffusion_scheduler")
        if scheduler_path is None:
            scheduler_path = os.path.join(local_model_path, "scheduler")
        return scheduler_path

    def _is_dance_case3_enabled(self) -> bool:
        return bool(self.config.actor.get("dance_case3_mode", False))

    def _dance_case3_mismatch_reasons(self) -> list[str]:
        reasons: list[str] = []
        if not self.diffusion:
            reasons.append("trainer.diffusion must be true")
        if not self.disaggregate:
            reasons.append("trainer.disaggregate must be true")

        trainer_pipelined = OmegaConf.select(self.config, "trainer.pipelined_micro_batch")
        if trainer_pipelined is not None and bool(trainer_pipelined):
            reasons.append("trainer.pipelined_micro_batch must be false")

        adv_estimator = OmegaConf.select(self.config, "algorithm.adv_estimator")
        if adv_estimator is not None and str(adv_estimator).lower() != "grpo":
            reasons.append("algorithm.adv_estimator must be grpo")
        return reasons

    def _is_dance_case3_mode(self) -> bool:
        return self._is_dance_case3_enabled() and len(self._dance_case3_mismatch_reasons()) == 0

    def _is_dance_case4_enabled(self) -> bool:
        return bool(self.config.actor.get("dance_case4_mode", False))

    def _dance_case4_mismatch_reasons(self) -> list[str]:
        reasons: list[str] = []
        if not self.diffusion:
            reasons.append("trainer.diffusion must be true")
        if self.disaggregate:
            reasons.append("trainer.disaggregate must be false")

        trainer_pipelined = OmegaConf.select(self.config, "trainer.pipelined_micro_batch")
        if trainer_pipelined is not None and bool(trainer_pipelined):
            reasons.append("trainer.pipelined_micro_batch must be false")

        adv_estimator = OmegaConf.select(self.config, "algorithm.adv_estimator")
        if adv_estimator is not None and str(adv_estimator).lower() != "grpo":
            reasons.append("algorithm.adv_estimator must be grpo")
        return reasons

    def _is_dance_case4_mode(self) -> bool:
        return self._is_dance_case4_enabled() and len(self._dance_case4_mismatch_reasons()) == 0

    def _get_dance_case4_grad_clip(self) -> float:
        grad_clip = self.config.actor.get("grad_clip", None)
        if grad_clip is None:
            grad_clip = self.config.actor.get("max_grad_norm", 1.0)
        return float(grad_clip)

    def _materialize_batch_after_transfer(self, data_proto: DataProto) -> None:
        # Work around the NaN bug observed on NPU by eagerly materializing tensors
        # after transfer. We keep this enabled on both GPU and NPU for consistency.
        if data_proto.batch is None:
            return

        for key, tensor in data_proto.batch.items():
            if not torch.is_tensor(tensor):
                continue
            if tensor.device.type != "cpu":
                raise RuntimeError(
                    f"[dance_case4] expected CPU tensor after transfer for key={key}, got device={tensor.device}"
                )
            if tensor.numel() == 0:
                continue

            flattened = tensor.detach().reshape(-1)
            if torch.is_floating_point(flattened) or flattened.dtype == torch.bfloat16:
                _ = flattened.to(torch.float32).sum().item()
            elif flattened.dtype == torch.bool:
                _ = flattened.to(torch.int64).sum().item()
            else:
                _ = flattened.sum().item()

    def _build_model_optimizer_dance_dis(self) -> None:
        from accelerate.utils import set_seed
        from diffusers.optimization import get_scheduler
        from fastvideo.utils.fsdp_util import apply_fsdp_checkpointing, get_dit_fsdp_kwargs
        from fastvideo.utils.load import load_transformer, load_vae

        actor_extra = self.config.actor.get("extra", {})
        dance_cfg = actor_extra.get("dance", {}) if hasattr(actor_extra, "get") else {}

        def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
            if cfg is None:
                return default
            if hasattr(cfg, "get"):
                val = cfg.get(key, default)
            else:
                val = getattr(cfg, key, default)
            return default if val is None else val

        seed = OmegaConf.select(self.config, "algorithm.seed")
        if seed is not None:
            set_seed(seed)

        pretrained_model_name_or_path = _cfg_get(dance_cfg, "pretrained_model_name_or_path", self._get_local_model_path())
        model_type = _cfg_get(dance_cfg, "model_type", "hunyuan_hf")
        master_weight_type = _cfg_get(dance_cfg, "master_weight_type", "bf16")
        sharding_strategy = _cfg_get(dance_cfg, "fsdp_sharding_strategy", "full")
        use_cpu_offload = bool(_cfg_get(dance_cfg, "use_cpu_offload", False))
        gradient_checkpointing = bool(
            self.config.model.get("enable_gradient_checkpointing", False)
            or actor_extra.get("gradient_checkpointing", False)
        )

        self.inferencer = None
        if self.role == "rollout_ref":
            use_videoalign = bool(_cfg_get(dance_cfg, "use_videoalign", False))
            if use_videoalign:
                from fastvideo.models.videoalign.inference import VideoVLMRewardInference

                ckpt_path = _cfg_get(dance_cfg, "videoalign_ckpt_path", "/share/models/dancegrpo/videoalign_ckpt")
                base_model_name_or_path = _cfg_get(dance_cfg, "videoalign_base_model_name_or_path", None)
                self.inferencer = VideoVLMRewardInference(
                    load_from_pretrained=ckpt_path,
                    device=torch.device(get_device_name(), get_device_id()),
                    dtype=torch.bfloat16,
                    base_model_name_or_path=base_model_name_or_path,
                )

            self.rollout = load_transformer(
                model_type=model_type,
                dit_model_name_or_path=None,
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                master_weight_type=torch.bfloat16,
            ).to(torch.device(get_device_name(), get_device_id()))
            self.rollout.eval()

            vae_model_path = _cfg_get(dance_cfg, "vae_model_path", pretrained_model_name_or_path)
            self.vae, _, fps = load_vae(model_type, vae_model_path)
            self.rollout_fps = int(_cfg_get(self.config.rollout, "fps", fps))
            return

        if self.role != "actor":
            raise ValueError(f"dance_case3_mode only supports role='actor' or role='rollout_ref', got {self.role}")

        transformer = load_transformer(
            model_type=model_type,
            dit_model_name_or_path=None,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            master_weight_type=torch.float32 if master_weight_type == "fp32" else torch.bfloat16,
        )
        fsdp_kwargs, no_split_modules = get_dit_fsdp_kwargs(
            transformer=transformer,
            sharding_strategy=sharding_strategy,
            use_lora=False,
            cpu_offload=use_cpu_offload,
            master_weight_type=master_weight_type,
        )
        self.transformer = FSDP(transformer, process_group=self.actor_pg, **fsdp_kwargs)
        self.actor_module_fsdp = self.transformer

        if gradient_checkpointing:
            selective_checkpointing = actor_extra.get("selective_checkpointing", 1.0)
            apply_fsdp_checkpointing(transformer, no_split_modules, selective_checkpointing)

        self.transformer.train()
        params_to_optimize = [p for p in self.transformer.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=float(self.config.actor.optim.lr),
            betas=(0.9, 0.999),
            weight_decay=float(self.config.actor.optim.weight_decay),
            eps=1e-8,
        )
        self.lr_scheduler = get_scheduler(
            name=self.config.actor.optim.get("warmup_style", "constant"),
            optimizer=self.optimizer,
            num_warmup_steps=max(0, int(self.config.actor.optim.get("lr_warmup_steps", 0))),
            num_training_steps=max(1, int(self.config.actor.optim.get("total_training_steps", 1_000_000))),
            num_cycles=float(self.config.actor.optim.get("num_cycles", 0.5)),
            power=float(self.config.actor.optim.get("power", 1.0)),
            last_epoch=-1,
        )

    def _build_model_optimizer_dance(self) -> None:
        from accelerate.utils import set_seed
        from diffusers.optimization import get_scheduler
        from fastvideo.utils.fsdp_util import apply_fsdp_checkpointing, get_dit_fsdp_kwargs
        from fastvideo.utils.load import load_transformer, load_vae

        actor_extra = self.config.actor.get("extra", {})
        dance_cfg = actor_extra.get("dance", {}) if hasattr(actor_extra, "get") else {}

        def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
            if cfg is None:
                return default
            if hasattr(cfg, "get"):
                val = cfg.get(key, default)
            else:
                val = getattr(cfg, key, default)
            return default if val is None else val

        seed = OmegaConf.select(self.config, "algorithm.seed")
        if seed is not None:
            set_seed(seed)

        if self.rank == 0:
            logger.info("[dance_case4] building dance actor/rollout worker for role=%s", self.role)

        pretrained_model_name_or_path = _cfg_get(dance_cfg, "pretrained_model_name_or_path", self._get_local_model_path())
        model_type = _cfg_get(dance_cfg, "model_type", "hunyuan_hf")
        master_weight_type = _cfg_get(dance_cfg, "master_weight_type", "bf16")
        sharding_strategy = _cfg_get(dance_cfg, "fsdp_sharding_strategy", "full")
        use_cpu_offload = bool(_cfg_get(dance_cfg, "use_cpu_offload", False))

        self.inferencer = None
        if self.role in ["actor_rollout_ref", "rollout_ref", "actor_rollout"]:
            use_videoalign = bool(_cfg_get(dance_cfg, "use_videoalign", False))
            if use_videoalign:
                from fastvideo.models.videoalign.inference import VideoVLMRewardInference

                ckpt_path = _cfg_get(dance_cfg, "videoalign_ckpt_path", "/share/models/dancegrpo/videoalign_ckpt")
                base_model_name_or_path = _cfg_get(dance_cfg, "videoalign_base_model_name_or_path", None)
                self.inferencer = VideoVLMRewardInference(
                    load_from_pretrained=ckpt_path,
                    device=torch.device(get_device_name(), get_device_id()),
                    dtype=torch.bfloat16,
                    base_model_name_or_path=base_model_name_or_path,
                )

        transformer = load_transformer(
            model_type=model_type,
            dit_model_name_or_path=None,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            master_weight_type=torch.float32 if master_weight_type == "fp32" else torch.bfloat16,
        )
        fsdp_kwargs, no_split_modules = get_dit_fsdp_kwargs(
            transformer=transformer,
            sharding_strategy=sharding_strategy,
            use_lora=False,
            cpu_offload=use_cpu_offload,
            master_weight_type=master_weight_type,
        )
        self.transformer = FSDP(transformer, **fsdp_kwargs)

        gradient_checkpointing = bool(
            self.config.model.get("enable_gradient_checkpointing", False)
            or actor_extra.get("gradient_checkpointing", False)
        )
        if gradient_checkpointing:
            selective_checkpointing = actor_extra.get("selective_checkpointing", 1.0)
            apply_fsdp_checkpointing(transformer, no_split_modules, selective_checkpointing)

        self.transformer.train()
        params_to_optimize = [p for p in self.transformer.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=float(self.config.actor.optim.lr),
            betas=(0.9, 0.999),
            weight_decay=float(self.config.actor.optim.weight_decay),
            eps=1e-8,
        )
        self.lr_scheduler = get_scheduler(
            name=self.config.actor.optim.get("warmup_style", "constant"),
            optimizer=self.optimizer,
            num_warmup_steps=max(0, int(self.config.actor.optim.get("lr_warmup_steps", 0))),
            num_training_steps=max(1, int(self.config.actor.optim.get("total_training_steps", 1_000_000))),
            num_cycles=float(self.config.actor.optim.get("num_cycles", 0.5)),
            power=float(self.config.actor.optim.get("power", 1.0)),
            last_epoch=-1,
        )

        vae_model_path = _cfg_get(dance_cfg, "vae_model_path", pretrained_model_name_or_path)
        self.vae, _, fps = load_vae(model_type, vae_model_path)
        self.rollout_fps = int(_cfg_get(self.config.rollout, "fps", fps))

    def _generate_sequences_dance(self, prompts: DataProto) -> DataProto:
        from diffusers.video_processor import VideoProcessor
        from diffusers.utils import export_to_video

        actor_extra = self.config.actor.get("extra", {})
        dance_cfg = actor_extra.get("dance", {}) if hasattr(actor_extra, "get") else {}
        use_videoalign = bool(dance_cfg.get("use_videoalign", False))
        device = torch.device(get_device_name(), get_device_id())
        rollout_model = self.rollout if self._is_dance_case3_mode() else self.transformer
        if rollout_model is None:
            raise RuntimeError("dance rollout model is not initialized")

        encoder_hidden_states = prompts.batch["encoder_hidden_states"].to(device)
        encoder_attention_mask = prompts.batch["encoder_attention_mask"].to(device)
        caption = prompts.meta_info.get("caption")

        if self.config.rollout.get("use_group", False):
            encoder_hidden_states = torch.repeat_interleave(
                encoder_hidden_states, self.config.rollout.num_generations, dim=0
            )
            encoder_attention_mask = torch.repeat_interleave(
                encoder_attention_mask, self.config.rollout.num_generations, dim=0
            )
            if isinstance(caption, str):
                caption = [caption] * self.config.rollout.num_generations
            elif isinstance(caption, (list, tuple)):
                caption = [item for item in list(caption) for _ in range(self.config.rollout.num_generations)]
            else:
                raise ValueError(f"Unsupported caption type for dance case rollout: {type(caption)}")
        elif isinstance(caption, str):
            caption = [caption]
        elif isinstance(caption, tuple):
            caption = list(caption)

        def sd3_time_shift(shift: float, t: torch.Tensor) -> torch.Tensor:
            return (shift * t) / (1 + (shift - 1) * t)

        def flux_step(
            model_output: torch.Tensor,
            latents: torch.Tensor,
            eta: float,
            sigmas: torch.Tensor,
            index: int,
            prev_sample: torch.Tensor | None,
            grpo: bool,
            sde_solver: bool,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            sigma = sigmas[index]
            dsigma = sigmas[index + 1] - sigma
            prev_sample_mean = latents + dsigma * model_output
            pred_original_sample = latents - sigma * model_output
            delta_t = sigma - sigmas[index + 1]
            std_dev_t = eta * math.sqrt(float(delta_t))
            if sde_solver:
                score_estimate = -(latents - pred_original_sample * (1 - sigma)) / sigma**2
                log_term = -0.5 * eta**2 * score_estimate
                prev_sample_mean = prev_sample_mean + log_term * dsigma
            if grpo and prev_sample is None:
                prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t
            if not grpo:
                raise ValueError("dance case rollout expects GRPO mode")
            log_prob = (
                -((prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)) ** 2)
                / (2 * (std_dev_t**2))
            ) - math.log(std_dev_t) - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
            log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
            return prev_sample, pred_original_sample, log_prob

        w = int(self.config.rollout.width)
        h = int(self.config.rollout.height)
        t = int(self.config.rollout.num_frames)
        sample_steps = int(self.config.rollout.sampling_steps)
        sigma_schedule = sd3_time_shift(
            float(self.config.rollout.shift), torch.linspace(1, 0, sample_steps + 1, device=encoder_hidden_states.device)
        )

        spatial_downsample = 8
        temporal_downsample = 4
        in_channels = 16
        latent_t = ((t - 1) // temporal_downsample) + 1
        latent_w, latent_h = w // spatial_downsample, h // spatial_downsample

        all_latents: list[torch.Tensor] = []
        all_log_probs: list[torch.Tensor] = []
        all_vq_rewards: list[torch.Tensor] = []
        all_mq_rewards: list[torch.Tensor] = []
        rollout_stage_total_s = {
            "sampling_loop_s": 0.0,
            "vae_decode_s": 0.0,
            "video_export_s": 0.0,
            "videoalign_reward_s": 0.0,
        }
        os.makedirs("./videos", exist_ok=True)

        batch_indices = torch.chunk(torch.arange(encoder_hidden_states.shape[0], device=encoder_hidden_states.device), encoder_hidden_states.shape[0])
        shared_noise = None
        if self.config.rollout.get("use_same_noise", False):
            shared_noise = torch.randn(
                (1, in_channels, latent_t, latent_h, latent_w),
                device=encoder_hidden_states.device,
                dtype=torch.bfloat16,
            )

        for index, batch_idx in enumerate(batch_indices):
            batch_encoder_hidden_states = encoder_hidden_states[batch_idx]
            batch_encoder_attention_mask = encoder_attention_mask[batch_idx]
            batch_caption = [caption[int(i.item())] for i in batch_idx] if caption is not None else [""]

            if shared_noise is not None:
                input_latents = shared_noise.repeat(len(batch_idx), 1, 1, 1, 1)
            else:
                input_latents = torch.randn(
                    (len(batch_idx), in_channels, latent_t, latent_h, latent_w),
                    device=encoder_hidden_states.device,
                    dtype=torch.bfloat16,
                )

            with torch.no_grad():
                z = input_latents.clone()
                latents_path = [z]
                log_probs_path = []
                sample_timing_raw: dict[str, float] = {}
                with simple_timer("sampling_loop_s", sample_timing_raw):
                    for i in range(sample_steps):
                        sigma = sigma_schedule[i]
                        timestep_value = int(float(sigma) * 1000)
                        timesteps = torch.full(
                            [batch_encoder_hidden_states.shape[0]], timestep_value, device=z.device, dtype=torch.long
                        )
                        rollout_model.eval()
                        with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
                            model_pred = rollout_model(
                                hidden_states=z,
                                encoder_hidden_states=batch_encoder_hidden_states,
                                timestep=timesteps,
                                guidance=torch.tensor([6018.0], device=z.device, dtype=torch.bfloat16),
                                encoder_attention_mask=batch_encoder_attention_mask,
                                return_dict=False,
                            )[0]
                        z, pred_original, log_prob = flux_step(
                            model_output=model_pred,
                            latents=z.to(torch.float32),
                            eta=float(self.config.rollout.eta),
                            sigmas=sigma_schedule,
                            index=i,
                            prev_sample=None,
                            grpo=True,
                            sde_solver=True,
                        )
                        z = z.to(torch.bfloat16)
                        latents_path.append(z)
                        log_probs_path.append(log_prob)
                rollout_stage_total_s["sampling_loop_s"] += sample_timing_raw["sampling_loop_s"]
                latents = pred_original.to(torch.float32) / 0.476986

            batch_latents = torch.stack(latents_path, dim=1)
            batch_log_probs = torch.stack(log_probs_path, dim=1)
            all_latents.append(batch_latents)
            all_log_probs.append(batch_log_probs)

            self.vae.enable_tiling()
            video_processor = VideoProcessor(vae_scale_factor=8)
            sample_timing_raw = {}
            with simple_timer("vae_decode_s", sample_timing_raw):
                with torch.inference_mode():
                    with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
                        video = self.vae.decode(latents, return_dict=False)[0]
                        videos = video_processor.postprocess_video(video)
            rollout_stage_total_s["vae_decode_s"] += sample_timing_raw["vae_decode_s"]

            rank = int(os.environ.get("RANK", 0))
            video_path = os.path.abspath(f"./videos/hunyuan_{rank}_{index}.mp4")
            sample_timing_raw = {}
            with simple_timer("video_export_s", sample_timing_raw):
                export_to_video(videos[0], video_path, fps=self.rollout_fps)
            rollout_stage_total_s["video_export_s"] += sample_timing_raw["video_export_s"]

            vq_reward = torch.tensor(-1.0, device=encoder_hidden_states.device)
            mq_reward = torch.tensor(-1.0, device=encoder_hidden_states.device)
            if use_videoalign and self.inferencer is not None:
                sample_timing_raw = {}
                try:
                    with simple_timer("videoalign_reward_s", sample_timing_raw):
                        with torch.no_grad():
                            reward = self.inferencer.reward([video_path], [batch_caption[0]], use_norm=True)
                    vq_reward = torch.tensor(reward[0]["VQ"], device=encoder_hidden_states.device)
                    mq_reward = torch.tensor(reward[0]["MQ"], device=encoder_hidden_states.device)
                except Exception:
                    logger.exception("[dance_case] videoalign reward failed, fallback to -1 reward")
                rollout_stage_total_s["videoalign_reward_s"] += sample_timing_raw.get("videoalign_reward_s", 0.0)
            all_vq_rewards.append(vq_reward.unsqueeze(0))
            all_mq_rewards.append(mq_reward.unsqueeze(0))

        all_latents = torch.cat(all_latents, dim=0)
        all_log_probs = torch.cat(all_log_probs, dim=0)
        all_vq_rewards = torch.cat(all_vq_rewards, dim=0)
        all_mq_rewards = torch.cat(all_mq_rewards, dim=0)

        batch_size = all_latents.shape[0]
        timestep_value = [int(float(sigma) * 1000) for sigma in sigma_schedule][:sample_steps]
        timesteps = torch.tensor([timestep_value[:] for _ in range(batch_size)], device=all_latents.device, dtype=torch.long)

        samples = {
            "timesteps": timesteps.detach().clone()[:, :-1],
            "latents": all_latents[:, :-1][:, :-1],
            "next_latents": all_latents[:, 1:][:, :-1],
            "log_probs": all_log_probs[:, :-1],
            "vq_rewards": all_vq_rewards.to(torch.float32),
            "mq_rewards": all_mq_rewards.to(torch.float32),
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
        }
        rollout_num_samples = int(encoder_hidden_states.shape[0])
        rollout_stage_mean_s = {
            key: value / max(1, rollout_num_samples) for key, value in rollout_stage_total_s.items()
        }
        data_proto = DataProto.from_dict(
            tensors=samples,
            meta_info={
                "sigma_schedule": sigma_schedule.detach().cpu().numpy(),
                "rollout_num_samples": rollout_num_samples,
                "rollout_stage_total_s": rollout_stage_total_s,
                "rollout_stage_mean_s": rollout_stage_mean_s,
            },
        ).to("cpu")
        self._materialize_batch_after_transfer(data_proto)
        return data_proto

    def _update_actor_dance(self, data: DataProto) -> DataProto:
        device = torch.device(get_device_name(), get_device_id())
        samples = {k: data.batch[k].to(device) for k in data.batch.keys()}
        sigma_schedule = torch.as_tensor(data.meta_info["sigma_schedule"], device=device, dtype=torch.float32)
        self.optimizer.zero_grad()

        def flux_step(
            model_output: torch.Tensor,
            latents: torch.Tensor,
            eta: float,
            sigmas: torch.Tensor,
            index: int,
            prev_sample: torch.Tensor,
            grpo: bool,
            sde_solver: bool,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            sigma = sigmas[index]
            dsigma = sigmas[index + 1] - sigma
            prev_sample_mean = latents + dsigma * model_output
            pred_original_sample = latents - sigma * model_output
            delta_t = sigma - sigmas[index + 1]
            std_dev_t = eta * math.sqrt(float(delta_t))

            if sde_solver:
                score_estimate = -(latents - pred_original_sample * (1 - sigma)) / sigma**2
                log_term = -0.5 * eta**2 * score_estimate
                prev_sample_mean = prev_sample_mean + log_term * dsigma

            if grpo and prev_sample is None:
                prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t
            if not grpo:
                raise ValueError("dance case update expects GRPO mode")

            log_prob = (
                -((prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)) ** 2)
                / (2 * (std_dev_t**2))
            ) - math.log(std_dev_t) - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
            log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
            return prev_sample, pred_original_sample, log_prob

        def grpo_one_step(
            latents: torch.Tensor,
            pre_latents: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            encoder_attention_mask: torch.Tensor,
            timesteps: torch.Tensor,
            idx: int,
        ) -> torch.Tensor:
            with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
                self.transformer.train()
                model_pred = self.transformer(
                    hidden_states=latents,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timesteps,
                    guidance=torch.tensor([6018.0], device=latents.device, dtype=torch.bfloat16),
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
            _, _, log_prob = flux_step(
                model_output=model_pred,
                latents=latents.to(torch.float32),
                eta=float(self.config.rollout.eta),
                sigmas=sigma_schedule,
                index=idx,
                prev_sample=pre_latents.to(torch.float32),
                grpo=True,
                sde_solver=True,
            )
            return log_prob

        num_generations = int(self.config.rollout.num_generations)
        n_groups = len(samples["vq_rewards"]) // max(1, num_generations)
        vq_advantages = torch.zeros_like(samples["vq_rewards"])
        mq_advantages = torch.zeros_like(samples["mq_rewards"])
        for i in range(n_groups):
            start_idx = i * num_generations
            end_idx = (i + 1) * num_generations
            group_vq = samples["vq_rewards"][start_idx:end_idx]
            vq_advantages[start_idx:end_idx] = (group_vq - group_vq.mean()) / (group_vq.std() + 1e-8)
            group_mq = samples["mq_rewards"][start_idx:end_idx]
            mq_advantages[start_idx:end_idx] = (group_mq - group_mq.mean()) / (group_mq.std() + 1e-8)
        samples["vq_advantages"] = vq_advantages
        samples["mq_advantages"] = mq_advantages

        total_scores = self.config.rollout.vq_coef * vq_advantages + self.config.rollout.mq_coef * mq_advantages
        batch_size = int(samples["timesteps"].shape[0])
        bestofn = int(self.config.rollout.bestofn)
        if num_generations != bestofn and bestofn > 0 and bestofn <= batch_size:
            sorted_indices = torch.argsort(total_scores)
            top_indices = sorted_indices[-bestofn // 2 :]
            bottom_indices = sorted_indices[: bestofn // 2]
            selected_indices = torch.cat([top_indices, bottom_indices])
            selected_indices = selected_indices[torch.randperm(len(selected_indices), device=selected_indices.device)]
            for key in list(samples.keys()):
                samples[key] = samples[key][selected_indices]
            batch_size = len(selected_indices)

        perms = torch.stack(
            [torch.randperm(samples["timesteps"].shape[1], device=samples["timesteps"].device) for _ in range(batch_size)]
        )
        for key in ["timesteps", "latents", "next_latents", "log_probs"]:
            samples[key] = samples[key][torch.arange(batch_size, device=samples[key].device)[:, None], perms]

        samples_batched = {k: v.unsqueeze(1) for k, v in samples.items()}
        samples_batched_list = [dict(zip(samples_batched, values)) for values in zip(*samples_batched.values())]

        dance_cfg = self.config.actor.get("extra", {}).get("dance", {})
        timestep_fraction = float(dance_cfg.get("timestep_fraction", 1.0))
        train_timesteps = int(samples["timesteps"].shape[1] * timestep_fraction)
        train_timesteps = max(1, train_timesteps)
        avg_loss = torch.tensor(0.0, device=device)
        for i, sample in enumerate(samples_batched_list):
            for step_idx in range(train_timesteps):
                new_log_probs = grpo_one_step(
                    latents=sample["latents"][:, step_idx],
                    pre_latents=sample["next_latents"][:, step_idx],
                    encoder_hidden_states=sample["encoder_hidden_states"],
                    encoder_attention_mask=sample["encoder_attention_mask"],
                    timesteps=sample["timesteps"][:, step_idx],
                    idx=int(perms[i][step_idx].item()),
                )

                ratio = torch.exp(new_log_probs - sample["log_probs"][:, step_idx])
                clip_range = 1e-4
                adv_clip_max = 5.0

                vq_adv = torch.clamp(sample["vq_advantages"], -adv_clip_max, adv_clip_max)
                mq_adv = torch.clamp(sample["mq_advantages"], -adv_clip_max, adv_clip_max)

                vq_unclipped = -vq_adv * ratio
                vq_clipped = -vq_adv * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
                vq_loss = torch.mean(torch.maximum(vq_unclipped, vq_clipped))

                mq_unclipped = -mq_adv * ratio
                mq_clipped = -mq_adv * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
                mq_loss = torch.mean(torch.maximum(mq_unclipped, mq_clipped))

                final_loss = (
                    self.config.rollout.vq_coef * vq_loss + self.config.rollout.mq_coef * mq_loss
                ) / (max(1, int(self.config.actor.get("gradient_accumulation_steps", 1))) * train_timesteps)
                final_loss.backward()
                avg_loss = final_loss.detach()

            self.transformer.clip_grad_norm_(self._get_dance_case4_grad_clip())
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

        output = DataProto(meta_info={"metrics": {"actor/loss": float(avg_loss.item())}})
        return output.to("cpu")

    def _build_model_optimizer(
        self,
        model_path,
        fsdp_config: FSDPEngineConfig,
        optim_config,
        override_model_config,
        use_remove_padding=False,
        use_fused_kernels=False,
        enable_gradient_checkpointing=False,
        trust_remote_code=False,
        use_liger=False,
        role="actor",
        enable_activation_offload=False,
    ):
        if self.diffusion:
            return self._build_model_optimizer_diffusion(
                model_path=model_path, fsdp_config=fsdp_config, optim_config=optim_config, role=role
            )

        from torch import optim
        from torch.distributed.fsdp import CPUOffload, MixedPrecision
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq

        from verl.utils.model import get_generation_config, print_model_size, update_model_config
        from verl.utils.torch_dtypes import PrecisionType

        assert role in ["actor", "ref"]

        log_gpu_memory_usage(f"Before init {role} from HF AutoModel", logger=logger)
        local_path = model_path

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code)

        if self.config.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.model.custom_chat_template
            else:
                self.tokenizer.chat_template = self.config.model.custom_chat_template

        torch_dtype = fsdp_config.get("model_dtype", None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # override model kwargs
        actor_model_config = AutoConfig.from_pretrained(
            local_path, trust_remote_code=trust_remote_code, attn_implementation="flash_attention_2"
        )

        # patch for kimi-vl
        if getattr(actor_model_config, "model_type", None) == "kimi_vl":
            actor_model_config.text_config.topk_method = "greedy"

        self.generation_config = get_generation_config(local_path, trust_remote_code=trust_remote_code)

        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
        if self.rank == 0:
            print(f"Model config after override: {actor_model_config}")

        # NOTE(fix me): tie_word_embedding causes meta_tensor init to hang
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not actor_model_config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if type(actor_model_config) in AutoModelForVision2Seq._model_mapping.keys():
                actor_module_class = AutoModelForVision2Seq
            else:
                actor_module_class = AutoModelForCausalLM

            actor_module = actor_module_class.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch_dtype,
                config=actor_model_config,
                trust_remote_code=trust_remote_code,
            )

            # Apply Liger kernel to the model if use_liger is set to True
            if use_liger:
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(model=actor_module)

            fused_kernel_options = self.config.model.get("fused_kernel_options", None)
            fused_kernels_backend = (
                fused_kernel_options.get("impl_backend", None) if fused_kernel_options is not None else None
            )

            apply_monkey_patch(
                model=actor_module,
                use_remove_padding=use_remove_padding,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
                use_fused_kernels=use_fused_kernels,
                fused_kernels_backend=fused_kernels_backend,
            )

            # some parameters may not in torch_dtype. TODO(zhangchi.usc1992) remove this after we switch to fsdp2
            actor_module.to(torch_dtype)

            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if self._is_lora:
                print("Applying LoRA to actor module")
                actor_module.enable_input_require_grads()
                # Convert config to regular Python types before creating PEFT model
                lora_config = {
                    "task_type": TaskType.CAUSAL_LM,
                    "r": self.config.model.lora_rank,
                    "lora_alpha": self.config.model.lora_alpha,
                    "target_modules": convert_to_regular_types(self.config.model.target_modules),
                    "exclude_modules": convert_to_regular_types(self.config.model.exclude_modules),
                    "bias": "none",
                }
                actor_module = get_peft_model(actor_module, LoraConfig(**lora_config))
        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage(f"After init {role} from HF AutoModel", logger=logger)

        # We wrap FSDP for rollout as well
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(
            module=actor_module,
            config=fsdp_config.get("wrap_policy", None),
            is_lora=self.config.model.get("lora_rank", 0) > 0,
        )

        if self._is_rollout and self.config.rollout.name == "hf":
            # TODO(zhangchi.usc1992, shengguangming) fix me. Current, auto_wrap_policy causes HFRollout to hang in Gemma
            auto_wrap_policy = None

        if self.rank == 0:
            print(f"wrap_policy: {auto_wrap_policy}")

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        # TODO: add transformer policy
        # We force reference policy to use CPUOffload to save memory.
        # We force turn off CPUOffload for actor because it causes incorrect results when using grad accumulation
        cpu_offload = None if role == "actor" else CPUOffload(offload_params=True)
        fsdp_strategy = self.config.actor.strategy
        if fsdp_strategy == "fsdp":
            actor_module_fsdp = FSDP(
                actor_module,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,  # zero3
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                use_orig_params=fsdp_config.get("use_orig_params", False),
                forward_prefetch=fsdp_config.get("forward_prefetch", False),
            )
        elif fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(
                param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True
            )
            if role == "actor" and fsdp_config.offload_policy:
                cpu_offload = CPUOffloadPolicy(pin_memory=True)
                self._is_offload_param = False
                self._is_offload_optimizer = False
            else:
                cpu_offload = None if role == "actor" else CPUOffloadPolicy(pin_memory=True)

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": fsdp_config.reshard_after_forward,
            }
            full_state = actor_module.state_dict()
            apply_fsdp2(actor_module, fsdp_kwargs, fsdp_config)
            fsdp2_load_full_state_dict(actor_module, full_state, fsdp_mesh, cpu_offload)
            actor_module_fsdp = actor_module
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        if enable_activation_offload:
            enable_activation_offloading(actor_module_fsdp, fsdp_strategy, enable_gradient_checkpointing)

        log_gpu_memory_usage(f"After {role} FSDP init", logger=logger)

        # TODO: add more optimizer args into config
        if role == "actor" and optim_config is not None:
            from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

            actor_optimizer = optim.AdamW(
                actor_module_fsdp.parameters(),
                lr=optim_config.lr,
                betas=optim_config.get("betas", (0.9, 0.999)),
                weight_decay=optim_config.get("weight_decay", 1e-2),
            )

            total_steps = optim_config.get("total_training_steps", 0)
            num_warmup_steps = int(optim_config.get("lr_warmup_steps", -1))
            warmup_style = optim_config.get("warmup_style", "constant")
            min_lr_ratio = optim_config.get("min_lr_ratio", 0.0)
            num_cycles = optim_config.get("num_cycles", 0.5)
            if num_warmup_steps < 0:
                num_warmup_steps_ratio = optim_config.get("lr_warmup_steps_ratio", 0.0)
                num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            if self.rank == 0:
                print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

            if warmup_style == "constant":
                actor_lr_scheduler = get_constant_schedule_with_warmup(
                    optimizer=actor_optimizer, num_warmup_steps=num_warmup_steps
                )
            elif warmup_style == "cosine":
                actor_lr_scheduler = get_cosine_schedule_with_warmup(
                    optimizer=actor_optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=total_steps,
                    min_lr_ratio=min_lr_ratio,
                    num_cycles=num_cycles,
                )
            else:
                raise NotImplementedError(f"Warmup style {warmup_style} is not supported")

            log_gpu_memory_usage(f"After {role} optimizer init", logger=logger)
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config

    def _build_model_optimizer_diffusion(self, model_path, fsdp_config: FSDPEngineConfig, optim_config, role="actor"):
        from torch import optim
        from torch.distributed.fsdp import CPUOffload, MixedPrecision
        from diffusers import StableDiffusion3Pipeline, WanPipeline

        from verl.utils.torch_dtypes import PrecisionType

        assert role in ["actor", "ref"]

        log_gpu_memory_usage(f"Before init {role} diffusion transformer", logger=logger)
        torch_dtype = fsdp_config.get("model_dtype", None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if role == "actor" else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        model_path_lower = model_path.lower()
        if "wan" in model_path_lower:
            pipeline = WanPipeline.from_pretrained(model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True)
        else:
            pipeline = StableDiffusion3Pipeline.from_pretrained(
                model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True
            )
        # Diffusion pipelines still expose a tokenizer/processor we can persist in checkpoints.
        # Keep a reference before freeing the pipeline to avoid checkpoint manager assertions.
        self.processor = getattr(pipeline, "processor", None)
        self.tokenizer = getattr(pipeline, "tokenizer", None)
        diffusion_module = pipeline.transformer
        diffusion_module.to(torch_dtype)
        if role == "ref":
            diffusion_module.requires_grad_(False)
        del pipeline
        if self._is_actor and self.actor_pg is not None:
            dist.barrier(self.actor_pg)
        if self._is_ref and self.rollout_ref_pg is not None:
            dist.barrier(self.rollout_ref_pg)

        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)
        auto_wrap_policy = get_fsdp_wrap_policy(
            module=diffusion_module,
            config=fsdp_config.get("wrap_policy", None),
            is_lora=self.config.model.get("lora_rank", 0) > 0,
        )
        if self.rank == 0:
            print(f"wrap_policy (diffusion): {auto_wrap_policy}")

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)
        cpu_offload = None if role == "actor" else CPUOffload(offload_params=True)
        fsdp_strategy = self.config.actor.strategy
        # Keep diffusion init behavior closer to disco_rl: only sync module states
        # when explicitly enabled (e.g. rank0-init style flows).
        sync_module_states = bool(fsdp_config.get("enable_rank0_init", False))
        if fsdp_strategy == "fsdp":
            process_group = None
            device_mesh = fsdp_mesh
            if self.disaggregate:
                if self._is_actor and self.actor_pg is not None:
                    process_group = self.actor_pg
                elif (self._is_rollout or self._is_ref) and self.rollout_ref_pg is not None:
                    process_group = self.rollout_ref_pg
                if process_group is not None:
                    device_mesh = None
            module_fsdp = FSDP(
                diffusion_module,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,
                mixed_precision=mixed_precision,
                sync_module_states=sync_module_states,
                device_mesh=device_mesh,
                process_group=process_group,
                use_orig_params=fsdp_config.get("use_orig_params", False),
                forward_prefetch=fsdp_config.get("forward_prefetch", False),
            )
        elif fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(
                param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True
            )
            if role == "actor" and fsdp_config.offload_policy:
                cpu_offload = CPUOffloadPolicy(pin_memory=True)
                self._is_offload_param = False
                self._is_offload_optimizer = False
            else:
                cpu_offload = None if role == "actor" else CPUOffloadPolicy(pin_memory=True)

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": fsdp_config.reshard_after_forward,
            }
            full_state = diffusion_module.state_dict()
            apply_fsdp2(diffusion_module, fsdp_kwargs, fsdp_config)
            fsdp2_load_full_state_dict(diffusion_module, full_state, fsdp_mesh, cpu_offload)
            module_fsdp = diffusion_module
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        log_gpu_memory_usage(f"After {role} diffusion FSDP init", logger=logger)

        if role == "actor" and optim_config is not None:
            from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

            diffusion_optimizer = optim.AdamW(
                module_fsdp.parameters(),
                lr=optim_config.lr,
                betas=optim_config.get("betas", (0.9, 0.999)),
                weight_decay=optim_config.get("weight_decay", 1e-2),
            )
            total_steps = optim_config.get("total_training_steps", 0)
            num_warmup_steps = int(optim_config.get("lr_warmup_steps", -1))
            warmup_style = optim_config.get("warmup_style", "constant")
            min_lr_ratio = optim_config.get("min_lr_ratio", 0.0)
            num_cycles = optim_config.get("num_cycles", 0.5)
            if num_warmup_steps < 0:
                num_warmup_steps_ratio = optim_config.get("lr_warmup_steps_ratio", 0.0)
                num_warmup_steps = int(num_warmup_steps_ratio * total_steps)
            if warmup_style == "constant":
                diffusion_lr_scheduler = get_constant_schedule_with_warmup(
                    optimizer=diffusion_optimizer, num_warmup_steps=num_warmup_steps
                )
            elif warmup_style == "cosine":
                diffusion_lr_scheduler = get_cosine_schedule_with_warmup(
                    optimizer=diffusion_optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=total_steps,
                    min_lr_ratio=min_lr_ratio,
                    num_cycles=num_cycles,
                )
            else:
                raise NotImplementedError(f"Warmup style {warmup_style} is not supported")
        else:
            diffusion_optimizer = None
            diffusion_lr_scheduler = None

        return module_fsdp, diffusion_optimizer, diffusion_lr_scheduler, None

    def _build_rollout(self, trust_remote_code=False):
        rollout_name = self.config.rollout.name
        if self.diffusion or rollout_name == "diffusion":
            return self._build_rollout_diffusion()

        from torch.distributed.device_mesh import init_device_mesh

        # TODO(sgm): support FSDP hybrid shard for larger model
        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, (
            f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
        )
        rollout_device_mesh = init_device_mesh(
            device_name, mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"]
        )
        if rollout_name == "hf":
            from verl.workers.rollout import HFRollout
            from verl.workers.sharding_manager.base import BaseShardingManager

            rollout = HFRollout(module=self.actor_module_fsdp, config=self.config.rollout)
            rollout_sharding_manager = BaseShardingManager()
            # TODO: a sharding manager that do nothing?

        elif rollout_name == "vllm":
            from verl.workers.rollout.vllm_rollout import vLLMRollout
            from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager

            log_gpu_memory_usage(f"Before building {rollout_name} rollout", logger=logger)
            local_path = copy_to_local(self.config.model.path, use_shm=self.config.model.get("use_shm", False))
            lora_kwargs = (
                {"lora_kwargs": {"enable_lora": True, "max_loras": 1, "max_lora_rank": self._lora_rank}}
                if self._is_lora
                else {}
            )
            # lora_kwargs = {}
            from verl.workers.rollout.vllm_rollout import vLLMAsyncRollout

            vllm_rollout_cls = vLLMRollout if self.config.rollout.mode == "sync" else vLLMAsyncRollout
            rollout = vllm_rollout_cls(
                model_path=local_path,
                config=self.config.rollout,
                tokenizer=self.tokenizer,
                model_hf_config=self.actor_model_config,
                device_mesh=rollout_device_mesh,
                trust_remote_code=trust_remote_code,
                **lora_kwargs,
            )

            log_gpu_memory_usage(f"After building {rollout_name} rollout", logger=logger)
            full_params = torch.distributed.get_world_size() == 1
            rollout_sharding_manager = FSDPVLLMShardingManager(
                module=self.actor_module_fsdp,
                inference_engine=rollout.inference_engine,
                model_config=self.actor_model_config,
                rollout_config=self.config.rollout,
                full_params=full_params,
                device_mesh=rollout_device_mesh,
                offload_param=self._is_offload_param,
                load_format=self.config.rollout.load_format,
                layered_summon=self.config.rollout.get("layered_summon", False),
            )
            log_gpu_memory_usage("After building sharding manager", logger=logger)

        elif rollout_name == "sglang":
            from verl.workers.rollout.sglang_rollout.sglang_rollout import SGLangRollout

            # NOTE(linjunrong): Due to recent fp8 support in SGLang. Now importing any symbol relate to
            # SGLang's model_runner would check CUDA device capability. However, due to verl's setting,
            # the main process of ray can not find any CUDA device, which would potentially lead to:
            # "RuntimeError: No CUDA GPUs are available".
            # For this reason, sharding_manager.__init__ should not import FSDPSGLangShardingManager and
            # we import it here use the abs path.
            # check: https://github.com/sgl-project/sglang/blob/00f42707eaddfc2c0528e5b1e0094025c640b7a0/python/sglang/srt/layers/quantization/fp8_utils.py#L76
            from verl.workers.sharding_manager.fsdp_sglang import FSDPSGLangShardingManager

            local_path = copy_to_local(self.config.model.path)
            log_gpu_memory_usage(f"Before building {rollout_name} rollout", logger=logger)
            rollout = SGLangRollout(
                actor_module=local_path,
                config=self.config.rollout,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                model_hf_config=self.actor_model_config,
                trust_remote_code=trust_remote_code,
            )
            log_gpu_memory_usage(f"After building {rollout_name} rollout", logger=logger)

            if torch.distributed.get_world_size() == 1:
                self.config.rollout.load_format = "dummy_hf"
            rollout_sharding_manager = FSDPSGLangShardingManager(
                module=self.actor_module_fsdp,
                inference_engine=rollout._engine,
                model_config=self.actor_model_config,
                rollout_config=self.config.rollout,
                full_params="hf" in self.config.rollout.load_format,
                device_mesh=rollout_device_mesh,
                offload_param=self._is_offload_param,
                multi_stage_wake_up=self.config.rollout.multi_stage_wake_up,
            )
            log_gpu_memory_usage("After building sharding manager", logger=logger)

        else:
            raise NotImplementedError(f"Rollout name: {self.config.rollout.name} is not supported")

        return rollout, rollout_sharding_manager

    def _build_rollout_diffusion(self):
        from verl.workers.rollout import RolloutConfig, StableDiffusionRollout, WanRollout
        from verl.workers.sharding_manager.base import BaseShardingManager

        local_path = self._get_local_model_path()
        model_path_lower = local_path.lower()
        rollout_config = omega_conf_to_dataclass(self.config.rollout, dataclass_type=RolloutConfig)
        if "wan" in model_path_lower:
            rollout = WanRollout(model_path=local_path, config=rollout_config)
        elif "stable-diffusion" in model_path_lower or "stable_diffusion" in model_path_lower:
            rollout = StableDiffusionRollout(model_path=local_path, config=rollout_config)
        else:
            raise ValueError(f"Model {local_path} is not supported for diffusion rollout.")
        rollout_sharding_manager = BaseShardingManager()
        return rollout, rollout_sharding_manager

    @register(dispatch_mode=Dispatch.ALL_TO_ALL)
    def prof_start(self, wait=3, warmup=1, active=1, repeat=1):
        if self.rank != 0:
            return
        print(f"prof start rank{self.rank}!")
        if not self._enable_prof_env or self._prof_active:
            return
        subdir = f"{self.role}_rank{self.rank}_local{self._local_rank}"
        outdir = os.path.join(self._prof_logdir, subdir)
        os.makedirs(outdir, exist_ok=True)
        self._prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(outdir),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_modules=False,
        )
        self._prof.__enter__()
        self._prof_enabled = True
        self._prof_active  = True

    @register(dispatch_mode=Dispatch.ALL_TO_ALL)
    def prof_step(self):
        if self.rank != 0:
            return
        print(f"prof step rank{self.rank}!")
        if self._prof_enabled and self._prof is not None:
            self._prof.step()

    @register(dispatch_mode=Dispatch.ALL_TO_ALL)
    def prof_stop(self):
        if self.rank != 0:
            return
        print(f"prof stop rank{self.rank}!")
        save_file_name = f"/prof_{self.role}_rank_{self.rank}.json"
        self._prof.export_chrome_trace(self._prof_logdir + save_file_name)
        if self._prof_enabled and self._prof is not None:
            self._prof.__exit__(None, None, None)
        self._prof = None
        self._prof_enabled = False
        self._prof_active  = False

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl.workers.actor import DataParallelPPOActor

        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        if self._is_dance_case3_enabled():
            reasons = self._dance_case3_mismatch_reasons()
            if reasons:
                reason_text = "; ".join(reasons)
                raise ValueError(f"[dance_case3] dance_case3_mode=true but case3 conditions are not met: {reason_text}")
            self._build_model_optimizer_dance_dis()
            return

        if self._is_dance_case4_enabled():
            reasons = self._dance_case4_mismatch_reasons()
            if reasons:
                reason_text = "; ".join(reasons)
                raise ValueError(f"[dance_case4] dance_case4_mode=true but case4 conditions are not met: {reason_text}")
            self._build_model_optimizer_dance()
            return

        override_model_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))
        use_remove_padding = self.config.model.get("use_remove_padding", False)
        use_shm = self.config.model.get("use_shm", False)
        use_fused_kernels = self.config.model.get("use_fused_kernels", False)

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = omega_conf_to_dataclass(self.config.actor.fsdp_config)
            else:
                optim_config = None
                fsdp_config = FSDPEngineConfig()

            local_path = self._get_local_model_path()
            (
                self.actor_module_fsdp,
                self.actor_optimizer,
                self.actor_lr_scheduler,
                self.actor_model_config,
            ) = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                enable_gradient_checkpointing=self.config.model.get("enable_gradient_checkpointing", False),
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="actor",
                enable_activation_offload=self.config.model.get("enable_activation_offload", False),
            )
            # get the original unwrapped module
            if fsdp_version(self.actor_module_fsdp) == 1:
                self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                log_gpu_memory_usage("After offload actor model during init", logger=logger)

            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage("After offload actor optimizer during init", logger=logger)

        if self._is_actor:
            actor_cfg = omega_conf_to_dataclass(self.config.actor)
            if self.diffusion:
                scheduler_path = self._resolve_diffusion_scheduler_path(self._get_local_model_path())
                actor_cfg.diffusion = True
                actor_cfg.diffusion_scheduler = scheduler_path
                rollout_guidance = getattr(self.config.rollout, "guidance_scale", None)
                if rollout_guidance is not None:
                    actor_cfg.guidance_scale = rollout_guidance
                if not hasattr(actor_cfg, "model"):
                    actor_cfg.model = self.config.model

            self.actor = DataParallelPPOActor(config=actor_cfg, actor_module=self.actor_module_fsdp, actor_optimizer=self.actor_optimizer)

        if self._is_rollout:
            self.rollout, self.rollout_sharding_manager = self._build_rollout(
                trust_remote_code=self.config.model.get("trust_remote_code", False)
            )

        if self._is_ref:
            local_path = self._get_local_model_path()
            self.ref_module_fsdp = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=omega_conf_to_dataclass(self.config.ref.fsdp_config),
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="ref",
            )[0]
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                if not hasattr(self.config.ref, "model"):
                    # Ensure ref workers carry the shared model config so diffusion helpers can locate assets.
                    self.config.ref.model = self.config.model
                self.config.ref.use_remove_padding = use_remove_padding
                self.config.ref.use_fused_kernels = use_fused_kernels
                if self.diffusion:
                    self.config.ref.diffusion = True
                    rollout_guidance = getattr(self.config.rollout, "guidance_scale", None)
                    if rollout_guidance is not None:
                        self.config.ref.guidance_scale = rollout_guidance
                    if getattr(self.config.ref, "scheduler", None) is None:
                        self.config.ref.scheduler = self._resolve_diffusion_scheduler_path(local_path)
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)

        if self._is_actor:
            self.flops_counter = None if self.diffusion else FlopsCounter(self.actor_model_config)
            processing_class = self.processor if self.processor is not None else self.tokenizer
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                processing_class=processing_class,
                checkpoint_config=self.config.actor.checkpoint,
            )

        if not self._is_actor and self._is_rollout:
            # If ActorRolloutRefWorker is initialized as a standalone rollout,
            # create a checkpoint manager for FSDP model to allow loading FSDP checkpoints for rollout.

            checkpoint_contents = OmegaConf.create({"load_contents": ["model"], "save_contents": []})
            processing_class = self.processor if self.processor is not None else self.tokenizer
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=None,
                lr_scheduler=None,
                processing_class=processing_class,
                checkpoint_config=checkpoint_contents,
            )

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    #@DistProfiler.annotate(color="red", role="actor_update")
    def update_actor(self, data: DataProto):
        # Support all hardwares
        data = data.to(get_device_id())

        if self._is_dance_case3_enabled() and not self._is_dance_case3_mode():
            reasons = "; ".join(self._dance_case3_mismatch_reasons())
            raise ValueError(f"[dance_case3] worker update_actor rejected due to case3 condition mismatch: {reasons}")
        if self._is_dance_case3_mode():
            if self.role != "actor":
                raise ValueError(f"[dance_case3] update_actor is only valid on role='actor', got {self.role}")
            return self._update_actor_dance(data)

        if self._is_dance_case4_enabled() and not self._is_dance_case4_mode():
            reasons = "; ".join(self._dance_case4_mismatch_reasons())
            raise ValueError(f"[dance_case4] worker update_actor rejected due to case4 condition mismatch: {reasons}")
        if self._is_dance_case4_mode():
            return self._update_actor_dance(data)

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_device_id())

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            # perform training
            with Timer(name="update_policy", logger=None) as timer:
                metrics = self.actor.update_policy(data=data)
            delta_time = timer.last
            if self.flops_counter is not None:
                global_num_tokens = data.meta_info["global_token_num"]
                estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
                metrics["perf/mfu/actor"] = (
                    estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
                )
                metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
                metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
                metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            if self.actor_lr_scheduler is not None:
                lr = self.actor_lr_scheduler.get_last_lr()[0]
                metrics["actor/lr"] = lr
                self.actor_lr_scheduler.step()

            # TODO: here, we should return all metrics
            output = DataProto(meta_info={"metrics": metrics})

            output = self.ulysses_sharding_manager.postprocess_data(data=output)
            output = output.to("cpu")

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during update_actor", logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
            log_gpu_memory_usage("After offload actor optimizer during update_actor", logger=logger)

        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    #@DistProfiler.annotate(color="red", role="rollout_generate")
    def generate_sequences(self, prompts: DataProto):
        # Support all hardwares
        prompts = prompts.to(get_device_id())

        if self._is_dance_case3_enabled() and not self._is_dance_case3_mode():
            reasons = "; ".join(self._dance_case3_mismatch_reasons())
            raise ValueError(f"[dance_case3] worker generate_sequences rejected due to case3 condition mismatch: {reasons}")
        if self._is_dance_case3_mode():
            if self.role != "rollout_ref":
                raise ValueError(
                    f"[dance_case3] generate_sequences is only valid on role='rollout_ref', got {self.role}"
                )
            return self._generate_sequences_dance(prompts)

        if self._is_dance_case4_enabled() and not self._is_dance_case4_mode():
            reasons = "; ".join(self._dance_case4_mismatch_reasons())
            raise ValueError(f"[dance_case4] worker generate_sequences rejected due to case4 condition mismatch: {reasons}")
        if self._is_dance_case4_mode():
            return self._generate_sequences_dance(prompts)

        assert self._is_rollout

        meta_info = {}
        if not self.diffusion:
            meta_info = {
                "eos_token_id": self.generation_config.eos_token_id
                if self.generation_config is not None
                else self.tokenizer.eos_token_id,
                "pad_token_id": self.generation_config.pad_token_id
                if self.generation_config is not None
                else self.tokenizer.pad_token_id,
            }
            prompts.meta_info.update(meta_info)
        timing_generate = {}
        with self.rollout_sharding_manager:
            log_gpu_memory_usage("After entering rollout sharding manager", logger=logger)

            prompts = self.rollout_sharding_manager.preprocess_data(prompts)
            with simple_timer("generate_sequences", timing_generate):
                output = self.rollout.generate_sequences(prompts=prompts)

            log_gpu_memory_usage("After rollout generation", logger=logger)

            output = self.rollout_sharding_manager.postprocess_data(output)

        output = output.to("cpu")
        timing_generate.update(self.rollout_sharding_manager.timing)
        # Diffusion rollout may not always enter the same distributed comm path on every rank.
        # Avoid collective timing reduce in this path to prevent NCCL init timeout.
        reduce_timing_across_ranks = bool(int(os.getenv("VERL_REDUCE_TIMING_ACROSS_RANKS", "1"))) and not self.diffusion
        if reduce_timing_across_ranks:
            # We calculate the average timing across all ranks to make sure meta_info["timing"] is the same.
            timing_generate = reduce_timing(timing_generate)
        output.meta_info["timing"] = timing_generate

        # clear kv cache
        get_torch_device().empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    #@DistProfiler.annotate(color="blue", role="actor_compute_log_prob")
    def compute_log_prob(self, data: DataProto):
        # when is_lora is True, we use the actor without lora applied to calculate the log_prob
        # which is mostly used for ref log_prob calculation
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        # Support all hardwares
        from contextlib import nullcontext

        is_lora = data.meta_info.pop("is_lora", False)
        adapter_ctx = self.actor.actor_module.disable_adapter() if is_lora else nullcontext()
        data = data.to(get_device_id())
        # we should always recompute old_log_probs when it is HybridEngine
        data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = self.config.rollout.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature
        # perform recompute log_prob
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            with adapter_ctx:
                # diffusion branch: compute log-prob over diffusion steps and return prev_sample_mean optionally
                if self.diffusion:
                    log_probs, prev_sample_mean = self.actor.compute_log_prob(data=data)
                    tensors = {"old_log_probs": log_probs}
                    if prev_sample_mean is not None:
                        tensors["old_prev_sample_mean"] = prev_sample_mean
                    output = DataProto.from_dict(tensors=tensors, meta_info={"temperature": self.config.rollout.get("temperature", 0.0)})
                else:
                    output, entropys = self.actor.compute_log_prob(data=data, calculate_entropy=True)
                    output = DataProto.from_dict(
                        tensors={"old_log_probs": output, "entropys": entropys},
                        meta_info={"temperature": self.config.rollout.temperature},
                    )
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
            self.actor.actor_module._handle.reshard(True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during compute_log_prob", logger=logger)

        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    #@DistProfiler.annotate(color="olive", role="ref_compute_log_prob")
    def compute_ref_log_prob(self, data: DataProto):
        if self._is_lora:
            # if _is_lora, actor without lora applied is the ref
            data.meta_info["is_lora"] = True
            data = self.compute_log_prob(data)
            # this old_log_probs is in fact ref_log_prob
            data = DataProto.from_dict(tensors={"ref_log_prob": data.batch["old_log_probs"]})
            return data
        assert self._is_ref
        # else:
        # otherwise, the class have a standalone ref model
        # Support all hardwares
        data = data.to(get_device_id())

        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["temperature"] = self.config.rollout.temperature
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            if self.diffusion:
                # compute diffusion ref log-prob and prev_sample_mean for KL
                ref_log_prob, ref_prev_mean = self.ref_policy.compute_log_prob(data=data)
                output = DataProto.from_dict(tensors={"ref_log_prob": ref_log_prob, "ref_prev_sample_mean": ref_prev_mean})
            else:
                output, _ = self.ref_policy.compute_log_prob(data=data, calculate_entropy=False)
                output = DataProto.from_dict(tensors={"ref_log_prob": output})
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1 and fsdp_version(self.ref_policy.actor_module) == 1:
            self.ref_policy.actor_module._handle.reshard(True)

        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        from verl.utils.logger import log_with_rank

        # only support save and load ckpt for actor
        assert self._is_actor

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        self.checkpoint_manager.save_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
        )
        dist.barrier()

        if self._is_lora and hasattr(getattr(self, "actor_module", self.actor_module_fsdp), "peft_config"):
            lora_save_path = os.path.join(local_path, "lora_adapter")
            peft_model = getattr(self, "actor_module", self.actor_module_fsdp)
            peft_config = {}
            if dist.get_rank() == 0:
                os.makedirs(lora_save_path, exist_ok=True)
                peft_config = asdict(peft_model.peft_config.get("default", {}))
                peft_config["task_type"] = peft_config["task_type"].value
                peft_config["peft_type"] = peft_config["peft_type"].value
                peft_config["target_modules"] = list(peft_config["target_modules"])
            try:
                if fsdp_version(self.actor_module_fsdp) > 0:
                    self.actor_module_fsdp = self.actor_module_fsdp.to(get_device_name())
                    lora_params = layered_summon_lora_params(self.actor_module_fsdp)
                    if dist.get_rank() == 0:
                        save_file(lora_params, os.path.join(lora_save_path, "adapter_model.safetensors"))
                        with open(os.path.join(lora_save_path, "adapter_config.json"), "w", encoding="utf-8") as f:
                            json.dump(peft_config, f, ensure_ascii=False, indent=4)
            except Exception as e:
                log_with_rank(
                    f"Save LoRA Adapter Error ({e})", rank=dist.get_rank(), logger=logger, log_only_rank_0=True
                )

            dist.barrier()
            log_with_rank(
                f"[rank-{self.rank}]: Saved LoRA adapter to: {lora_save_path}",
                rank=dist.get_rank(),
                logger=logger,
                log_only_rank_0=True,
            )

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        assert self._is_actor or (not self._is_actor and self._is_rollout), (
            f"Checkpoint loading is only supported for Actor or standalone Rollout Workers, but got "
            f"{self._is_actor} and {self._is_rollout}"
        )

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        self.checkpoint_manager.load_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.actor_optimizer)
            
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def start_profile(self, **kwargs) -> None:
        """Start profiling for the current rank in the current training step."""
        self.profiler.start(**kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def stop_profile(self) -> None:
        """Stop profiling for the current rank in the current training step."""
        self.profiler.stop()

class ActorRolloutRefWorker_encoder(Worker, DistProfilerExtension):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str, **kwargs):
        Worker.__init__(self)

        self.config = config
        self.profile_option = kwargs.get("profile_option", None)
        import torch.distributed

        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(
                backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}",
                rank=rank,
                world_size=world_size,
                init_method=os.environ.get("DIST_INIT_METHOD", None),
                timeout=_get_dist_timeout(),
            )

        self._prof = None
        self._prof_enabled = False
        self._prof_active = False
        self._prof_logdir = os.getenv("PROF_LOGDIR", "/workspace/yym/RLHF/verl-disaggregate/log/trace/col")
        self._enable_prof_env = bool(int(os.getenv("ENABLE_PROFILER", "0")))

        # build device mesh for FSDP
        world_size = torch.distributed.get_world_size()
        # TODO(sgm): support FSDP hybrid shard for larger model
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=self.config.actor.fsdp_config.fsdp_size)

        # build device mesh for Ulysses Sequence Parallel
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.actor.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
            )

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        self.role = role
        assert self.role in ["encoder_ref", "encoder_actor_rollout"]

        self._is_actor = self.role == "encoder_actor_rollout"
        self._is_rollout = self.role == "encoder_actor_rollout"
        self._is_ref = self.role == "encoder_ref"

        profiler_config = omega_conf_to_dataclass(config.get("profiler"))
        DistProfilerExtension.__init__(
            self, Profiler(config=profiler_config, task=self.role)
        )
        
        self._is_offload_param = False
        self._is_offload_optimizer = False
        # 目前来看，可以保持原有设置，即在offload设置上，encoder保持原actor与ref的设置，不自行新增设置
        # qzy:严格来说应该删去actor和ref设置，新增encoder_actor和encoder_ref设置
        if self._is_actor:
            self._is_offload_param = self.config.actor.fsdp_config.get("param_offload", False)
            self._is_offload_optimizer = self.config.actor.fsdp_config.get("optimizer_offload", False)
        elif self._is_ref:
            # TODO: it seems that manual offload is slowly than FSDP offload
            self._is_offload_param = self.config.ref.fsdp_config.get("param_offload", False)

        # qzy:这部分代码先暂时不变，目前encoder_actor 和 llm_actor的batch size都和原actor对齐，但是后面应该是要改的，做到分离
        # normalize config
        if self._is_actor:
            self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
            self.config.actor.ppo_mini_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
            assert self.config.actor.ppo_mini_batch_size > 0, (
                f"ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be larger than 0 after "
                f"normalization"
            )
            # micro bsz
            if self.config.actor.ppo_micro_batch_size is not None:
                self.config.actor.ppo_micro_batch_size //= (
                    self.device_mesh.size() // self.ulysses_sequence_parallel_size
                )
                self.config.actor.ppo_micro_batch_size_per_gpu = self.config.actor.ppo_micro_batch_size

            if self.config.actor.ppo_micro_batch_size_per_gpu is not None:
                assert self.config.actor.ppo_mini_batch_size % self.config.actor.ppo_micro_batch_size_per_gpu == 0, (
                    f"normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be divisible by "
                    f"ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}"
                )
                assert self.config.actor.ppo_mini_batch_size // self.config.actor.ppo_micro_batch_size_per_gpu > 0, (
                    f"normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be larger than "
                    f"ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}"
                )

        # normalize rollout config
        if self._is_rollout and self.config.rollout.log_prob_micro_batch_size is not None:
            self.config.rollout.log_prob_micro_batch_size //= (
                self.device_mesh.size() // self.ulysses_sequence_parallel_size
            )
            self.config.rollout.log_prob_micro_batch_size_per_gpu = self.config.rollout.log_prob_micro_batch_size
        # normalize ref config
        if self._is_ref and self.config.ref.log_prob_micro_batch_size is not None:
            self.config.ref.log_prob_micro_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
            self.config.ref.log_prob_micro_batch_size_per_gpu = self.config.ref.log_prob_micro_batch_size

    def _build_model_optimizer(
        self,
        model_path,
        fsdp_config: FSDPEngineConfig,
        optim_config,
        override_model_config,
        use_remove_padding=False,
        use_fused_kernels=False,
        enable_gradient_checkpointing=False,
        trust_remote_code=False,
        use_liger=False,
        role="actor",
        enable_activation_offload=False,
    ):
        from torch import optim
        from torch.distributed.fsdp import CPUOffload, MixedPrecision
        # from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq

        from verl.utils.model import get_generation_config, print_model_size, update_model_config
        from verl.utils.torch_dtypes import PrecisionType

        assert role in ["actor", "ref"]

        log_gpu_memory_usage(f"Before init {role} from HF AutoModel", logger=logger)
        local_path = copy_to_local(model_path)

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        # encoder doesn't have tokenizer
        self.tokenizer = None
        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code)

        torch_dtype = fsdp_config.get("model_dtype", None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # override model kwargs
        actor_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)

        # encoder doesn't need generation_config
        # self.generation_config = get_generation_config(local_path, trust_remote_code=trust_remote_code)

        if self.rank == 0:
            print(f"Model config after override: {actor_model_config}")

        # NOTE(fix me): tie_word_embedding causes meta_tensor init to hang
        init_context = get_init_weight_context_manager(use_meta_tensor=not actor_model_config.tie_word_embeddings, mesh=self.device_mesh)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel
            # actor_module_class = Qwen2_5_VisionTransformerPretrainedModel
            from verl.models.transformers.qwen2_5_vl import CustomQwen2_5_VLEncoder
            actor_module_class = CustomQwen2_5_VLEncoder

            actor_module = actor_module_class.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch_dtype,
                config=actor_model_config,
                trust_remote_code=trust_remote_code,
            )

            # Apply Liger kernel to the model if use_liger is set to True
            if use_liger:
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(model=actor_module)
            
            # below are monkey patch in ActorRolloutRefWorker, we may not use them
            # fused_kernel_options = self.config.model.get("fused_kernel_options", None)
            # fused_kernels_backend = (
            #     fused_kernel_options.get("impl_backend", None) if fused_kernel_options is not None else None
            # )

            # apply_monkey_patch(
            #     model=actor_module,
            #     use_remove_padding=use_remove_padding,
            #     ulysses_sp_size=self.ulysses_sequence_parallel_size,
            #     use_fused_kernels=use_fused_kernels,
            #     fused_kernels_backend=fused_kernels_backend,
            # )

            # some parameters may not in torch_dtype. TODO(zhangchi.usc1992) remove this after we switch to fsdp2
            actor_module.to(torch_dtype)

            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage(f"After init {role} from HF AutoModel", logger=logger)

        # We wrap FSDP for rollout as well
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        wrap_config = {"transformer_layer_cls_to_wrap": ["Qwen2_5_VLVisionBlock"],}
        # auto_wrap_policy = get_fsdp_wrap_policy(module=actor_module, config=fsdp_config.get("wrap_policy", None))
        auto_wrap_policy = get_fsdp_wrap_policy(module=actor_module, config=wrap_config)

        if self._is_rollout and self.config.rollout.name == "hf":
            # TODO(zhangchi.usc1992, shengguangming) fix me. Current, auto_wrap_policy causes HFRollout to hang in Gemma
            auto_wrap_policy = None

        print(f"wrap_policy: {auto_wrap_policy}")

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        # TODO: add transformer policy
        # We force reference policy to use CPUOffload to save memory.
        # We force turn off CPUOffload for actor because it causes incorrect results when using grad accumulation
        cpu_offload = None if role == "actor" else CPUOffload(offload_params=True)
        # qzy:这个地方encoder_ref encoder_actor llm_ref llm_actor的strategy都和原actor保持一致，后续可能会进行修改
        fsdp_strategy = self.config.actor.strategy
        if fsdp_strategy == "fsdp":
            actor_module_fsdp = FSDP(
                actor_module,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,  # zero3
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                use_orig_params=fsdp_config.get("use_orig_params", False),
                forward_prefetch=fsdp_config.get("forward_prefetch", False),
            )
        elif fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True)
            if role == "actor" and fsdp_config.offload_policy:
                cpu_offload = CPUOffloadPolicy(pin_memory=True)
                self._is_offload_param = False
                self._is_offload_optimizer = False
            else:
                cpu_offload = None if role == "actor" else CPUOffloadPolicy(pin_memory=True)

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": fsdp_config.reshard_after_forward,
            }
            full_state = actor_module.state_dict()
            apply_fsdp2(actor_module, fsdp_kwargs, fsdp_config)
            fsdp2_load_full_state_dict(actor_module, full_state, fsdp_mesh, cpu_offload)
            actor_module_fsdp = actor_module
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        if enable_activation_offload:
            enable_activation_offloading(actor_module_fsdp, fsdp_strategy, enable_gradient_checkpointing)
            
        log_gpu_memory_usage(f"After {role} FSDP init", logger=logger)

        # TODO: add more optimizer args into config
        if role == "actor" and optim_config is not None:
            from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

            actor_optimizer = optim.AdamW(
                actor_module_fsdp.parameters(),
                lr=optim_config.lr,
                betas=optim_config.get("betas", (0.9, 0.999)),
                weight_decay=optim_config.get("weight_decay", 1e-2),
            )

            total_steps = optim_config.get("total_training_steps", 0)
            num_warmup_steps = int(optim_config.get("lr_warmup_steps", -1))
            warmup_style = optim_config.get("warmup_style", "constant")
            min_lr_ratio = optim_config.get("min_lr_ratio", 0.0)
            num_cycles = optim_config.get("num_cycles", 0.5)
            if num_warmup_steps < 0:
                num_warmup_steps_ratio = optim_config.get("lr_warmup_steps_ratio", 0.0)
                num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

            if warmup_style == "constant":
                actor_lr_scheduler = get_constant_schedule_with_warmup(optimizer=actor_optimizer, num_warmup_steps=num_warmup_steps)
            elif warmup_style == "cosine":
                actor_lr_scheduler = get_cosine_schedule_with_warmup(optimizer=actor_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)
            else:
                raise NotImplementedError(f"Warmup style {warmup_style} is not supported")

            log_gpu_memory_usage(f"After {role} optimizer init", logger=logger)
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config

    def _build_rollout(self, trust_remote_code=False):
        from torch.distributed.device_mesh import init_device_mesh

        # TODO(sgm): support FSDP hybrid shard for larger model
        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
        rollout_device_mesh = init_device_mesh(device_name, mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"])
        rollout_name = self.config.rollout.encoder.name
        if rollout_name == "hf":
            from verl.workers.rollout import HFRollout
            from verl.workers.sharding_manager.base import BaseShardingManager

            rollout = HFRollout(module=self.actor_module_fsdp, config=self.config.rollout)
            rollout_sharding_manager = BaseShardingManager()
            # TODO: a sharding manager that do nothing?

        elif rollout_name == "vllm":
            from verl.workers.rollout.vllm_rollout import vLLMRollout
            from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager

            log_gpu_memory_usage(f"Before building {rollout_name} rollout", logger=logger)
            local_path = copy_to_local(self.config.model.encoder.path, use_shm=self.config.model.get("use_shm", False))
            from verl.workers.rollout.vllm_rollout import vLLMAsyncRollout

            vllm_rollout_cls = vLLMRollout if self.config.rollout.mode == "sync" else vLLMAsyncRollout
            rollout = vllm_rollout_cls(
                model_path=local_path,
                config=self.config.rollout,
                tokenizer=self.tokenizer,
                model_hf_config=self.actor_model_config,
                device_mesh=rollout_device_mesh,
                trust_remote_code=trust_remote_code,
            )

            log_gpu_memory_usage(f"After building {rollout_name} rollout", logger=logger)
            full_params = torch.distributed.get_world_size() == 1
            rollout_sharding_manager = FSDPVLLMShardingManager(
                module=self.actor_module_fsdp,
                inference_engine=rollout.inference_engine,
                model_config=self.actor_model_config,
                rollout_config=self.config.rollout,
                full_params=full_params,
                device_mesh=rollout_device_mesh,
                offload_param=self._is_offload_param,
                load_format=self.config.rollout.load_format,
                layered_summon=self.config.rollout.get("layered_summon", False),
            )
            log_gpu_memory_usage("After building sharding manager", logger=logger)

        elif rollout_name == "sglang":
            from verl.workers.rollout.sglang_rollout.sglang_rollout import SGLangRollout

            # NOTE(linjunrong): Due to recent fp8 support in SGLang. Now importing any symbol relate to
            # SGLang's model_runner would check CUDA device capability. However, due to verl's setting,
            # the main process of ray can not find any CUDA device, which would potentially lead to:
            # "RuntimeError: No CUDA GPUs are available".
            # For this reason, sharding_manager.__init__ should not import FSDPSGLangShardingManager and
            # we import it here use the abs path.
            # check: https://github.com/sgl-project/sglang/blob/00f42707eaddfc2c0528e5b1e0094025c640b7a0/python/sglang/srt/layers/quantization/fp8_utils.py#L76
            from verl.workers.sharding_manager.fsdp_sglang import FSDPSGLangShardingManager

            log_gpu_memory_usage(f"Before building {rollout_name} rollout", logger=logger)
            local_path = copy_to_local(self.config.model.encoder.path)
            rollout = SGLangRollout(
                actor_module=local_path,
                config=self.config.rollout,
                tokenizer=self.tokenizer,
                model_hf_config=self.actor_model_config,
                trust_remote_code=trust_remote_code,
            )
            log_gpu_memory_usage(f"After building {rollout_name} rollout", logger=logger)

            if torch.distributed.get_world_size() == 1:
                self.config.rollout.load_format = "dummy_hf"
            rollout_sharding_manager = FSDPSGLangShardingManager(
                module=self.actor_module_fsdp,
                inference_engine=rollout.inference_engine,
                model_config=self.actor_model_config,
                full_params="hf" in self.config.rollout.load_format,
                device_mesh=rollout_device_mesh,
                offload_param=self._is_offload_param,
            )
            log_gpu_memory_usage("After building sharding manager", logger=logger)

        elif rollout_name == "sglang_async":
            from verl.workers.rollout.sglang_rollout import AsyncSGLangRollout
            from verl.workers.sharding_manager.fsdp_sglang import FSDPAsyncSGLangShardingManager

            log_gpu_memory_usage(f"Before building {rollout_name} rollout", logger=None)
            rollout = AsyncSGLangRollout(
                actor_module=self.config.model.encoder.path,
                config=self.config.rollout,
                tokenizer=self.tokenizer,
                model_hf_config=self.actor_model_config,
                trust_remote_code=trust_remote_code,
            )
            log_gpu_memory_usage(f"After building {rollout_name} rollout", logger=None)

            if torch.distributed.get_world_size() == 1:
                self.config.rollout.load_format = "dummy_hf"
            rollout_sharding_manager = FSDPAsyncSGLangShardingManager(
                module=self.actor_module_fsdp,
                inference_engine=rollout._engine,
                model_config=self.actor_model_config,
                full_params="hf" in self.config.rollout.load_format,
                device_mesh=rollout_device_mesh,
            )
            log_gpu_memory_usage("After building sharding manager", logger=None)

        else:
            raise NotImplementedError(f"Rollout name: {self.config.rollout.name} is not supported")

        return rollout, rollout_sharding_manager

    @register(dispatch_mode=Dispatch.ALL_TO_ALL)
    def prof_start(self, wait=3, warmup=1, active=1, repeat=1):
        if self.rank != 0:
            return
        print(f"prof start rank{self.rank}!")
        if not self._enable_prof_env or self._prof_active:
            return
        subdir = f"{self.role}_rank{self.rank}_local{self._local_rank}"
        outdir = os.path.join(self._prof_logdir, subdir)
        os.makedirs(outdir, exist_ok=True)
        self._prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(outdir),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_modules=False,
        )
        self._prof.__enter__()
        self._prof_enabled = True
        self._prof_active  = True

    @register(dispatch_mode=Dispatch.ALL_TO_ALL)
    def prof_step(self):
        if self.rank != 0:
            return
        print(f"prof step rank{self.rank}!")
        if self._prof_enabled and self._prof is not None:
            self._prof.step()

    @register(dispatch_mode=Dispatch.ALL_TO_ALL)
    def prof_stop(self):
        if self.rank != 0:
            return
        print(f"prof stop rank{self.rank}!")
        save_file_name = f"/prof_{self.role}_rank_{self.rank}.json"
        self._prof.export_chrome_trace(self._prof_logdir + save_file_name)
        if self._prof_enabled and self._prof is not None:
            self._prof.__exit__(None, None, None)
        self._prof = None
        self._prof_enabled = False
        self._prof_active  = False

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl.workers.encoder import DataParallelPPOEncoder 

        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        from omegaconf import OmegaConf

        override_model_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))

        use_remove_padding = self.config.model.get("use_remove_padding", False)
        use_shm = self.config.model.get("use_shm", False)
        use_fused_kernels = self.config.model.get("use_fused_kernels", False)

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = omega_conf_to_dataclass(self.config.actor.fsdp_config)
            else:
                optim_config = None
                fsdp_config = FSDPEngineConfig()
            self.actor_module_fsdp, self.actor_optimizer, self.actor_lr_scheduler, self.actor_model_config = self._build_model_optimizer(
                model_path=self.config.model.encoder.path,
                # model_path=self.config.model.path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                enable_gradient_checkpointing=self.config.model.get("enable_gradient_checkpointing", False),
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="actor",
                enable_activation_offload=self.config.model.get("enable_activation_offload", False),
            )

            # get the original unwrapped module
            if fsdp_version(self.actor_module_fsdp) == 1:
                self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                log_gpu_memory_usage("After offload actor model during init", logger=logger)

            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage("After offload actor optimizer during init", logger=logger)
        # load from checkpoint
        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            with open_dict(self.config.actor):
                self.config.actor.use_remove_padding = use_remove_padding
            self.actor = DataParallelPPOEncoder(config=self.config.actor, encoder_module=self.actor_module_fsdp, encoder_optimizer=self.actor_optimizer)

        if self._is_rollout:
            # self.rollout, self.rollout_sharding_manager = self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))
            pass

        if self._is_ref:
            self.ref_module_fsdp = self._build_model_optimizer(
                model_path=self.config.model.encoder.path,
                fsdp_config=omega_conf_to_dataclass(self.config.ref.fsdp_config),
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="ref",
            )[0]
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
                self.config.ref.use_fused_kernels = use_fused_kernels
            self.ref_policy = DataParallelPPOEncoder(config=self.config.ref, encoder_module=self.ref_module_fsdp)

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            # self.checkpoint_manager = FSDPCheckpointManager(
            #     model=self.actor_module_fsdp,
            #     optimizer=self.actor.encoder_optimizer,
            #     lr_scheduler=self.actor_lr_scheduler,
            #     processing_class=self.processor if self.processor is not None else self.tokenizer,
            #     checkpoint_config=self.config.actor.checkpoint,
            # )

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto, encoder_input: DataProto):
        # Support all hardwares
        grad = DataProto(non_tensor_batch=data.non_tensor_batch)
        grad = grad.to(get_device_id())
        encoder_input = encoder_input.to(get_device_id())

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_device_id())

        # with self.ulysses_sharding_manager:
            # data = self.ulysses_sharding_manager.preprocess_data(data=data)
            # perform training
            # with Timer(name="update_policy", logger=None) as timer:
            # metrics = self.actor.update_policy(data=data)
        self.actor.update_policy(data=grad, encoder_input=encoder_input)
            # delta_time = timer.last
            # global_num_tokens = data.meta_info["global_token_num"]
            # estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            # metrics["perf/mfu/actor"] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
            # metrics["perf/max_memory_allocated_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
            # metrics["perf/max_memory_reserved_gb"] = torch.cuda.max_memory_reserved() / (1024**3)
            # metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

        self.actor_lr_scheduler.step()
            # lr = self.actor_lr_scheduler.get_last_lr()[0]
            # metrics["actor/lr"] = lr

            # TODO: here, we should return all metrics
        output = DataProto(meta_info=data.meta_info)

            # output = self.ulysses_sharding_manager.postprocess_data(data=output)
        output = output.to("cpu")

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during update_actor", logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
            log_gpu_memory_usage("After offload actor optimizer during update_actor", logger=logger)

        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def actor_forward(self, data: DataProto):
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)  
        data = data.to(get_device_id())     
        image_embed, video_embed, image_sizes, video_sizes = self.actor.extract_feature_train(data=data)    
        if image_embed is not None and video_embed is not None:
                embeds = {"image_embed": image_embed, "video_embed": video_embed, "image_sizes": image_sizes, "video_sizes": video_sizes}
        elif image_embed is not None and video_embed is None:
            embeds = {"image_embed": image_embed, "image_sizes": image_sizes}
        elif image_embed is None and video_embed is not None:
            embeds = {"video_embed": video_embed, "video_sizes": video_sizes}
        else:
            raise ValueError("Both image_embed and video_embed are None. At least one of them must be provided.")
        output = DataProto.from_dict(non_tensors=embeds)
        output = output.to("cpu")
        
        if self.world_size > 1 and fsdp_version(self.actor.encoder_module) == 1:
            self.actor.encoder_module._handle.reshard(True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during compute_log_prob", logger=logger)

        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def rollout_forward(self, data: DataProto):
        assert self._is_rollout
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)  

        # Support all hardwares
        data = data.to(get_device_id())
        # we should always recompute old_log_probs when it is HybridEngine
        data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        data.meta_info["use_dynamic_bsz"] = False

        assert "multi_modal_inputs" in data.non_tensor_batch.keys(), f'{data.non_tensor_batch.keys()}'

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            # dist_log(f'rollout_forward: {len(data.non_tensor_batch["multi_modal_data"])} {data.non_tensor_batch["multi_modal_data"][0]["pixel_values"].shape}')
            image_embed, video_embed = self.actor.extract_feature(data=data, split=True)
            if image_embed is not None and video_embed is not None:
                embeds = {"image_embed": image_embed, "video_embed": video_embed}
            elif image_embed is not None and video_embed is None:
                embeds = {"image_embed": image_embed}
            elif image_embed is None and video_embed is not None:
                embeds = {"video_embed": video_embed}
            else:
                raise ValueError("Both image_embed and video_embed are None. At least one of them must be provided.")
            #dist_log(f'img_grid_thw: {len(data.non_tensor_batch["multi_modal_data"])}, img_embd: {len(image_embed)}')
            vllm_input = []
            #print(f'DEBUG: {len(image_embed)} {len(data.non_tensor_batch["multi_modal_inputs"])}')
            for embd, grid in zip(image_embed,data.non_tensor_batch["multi_modal_inputs"]):
                vllm_input.append({"image": {"image_embeds": embd, "image_grid_thw": grid["image_grid_thw"]}})
            output = DataProto.from_dict(non_tensors={"multi_modal_data":vllm_input})
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1 and fsdp_version(self.actor.encoder_module) == 1:
            self.actor.encoder_module._handle.reshard(True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during compute_log_prob", logger=logger)

        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob_encoder(self, data: DataProto):
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        # Support all hardwares
        data = data.to(get_device_id())
        # we should always recompute old_log_probs when it is HybridEngine
        data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = self.config.rollout.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature
        # perform recompute log_prob
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            # output, entropys = self.actor.compute_log_prob(data=data, calculate_entropy=True)
            # output = DataProto.from_dict(
            #     tensors={"old_log_probs": output, "entropys": entropys},
            #     meta_info={"temperature": self.config.rollout.temperature},
            # )
            image_embed, video_embed = self.actor.extract_feature(data=data, split=True)

            if image_embed is not None and video_embed is not None:
                embeds = {"image_embed": image_embed, "video_embed": video_embed}
            elif image_embed is not None and video_embed is None:
                embeds = {"image_embed": image_embed}
            elif image_embed is None and video_embed is not None:
                embeds = {"video_embed": video_embed}
            else:
                raise ValueError("Both image_embed and video_embed are None. At least one of them must be provided.")

            output = DataProto.from_dict(non_tensors=embeds)
            output = self.ulysses_sharding_manager.postprocess_data(output)

        #print(f'DEBUG: compute_log_prob_encoder: {len(output.non_tensor_batch["image_embed"])}')

        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1 and fsdp_version(self.actor.encoder_module) == 1:
            self.actor.encoder_module._handle.reshard(True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during compute_log_prob", logger=logger)

        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob_encoder(self, data: DataProto):
        assert self._is_ref

        # Support all hardwares
        data = data.to(get_device_id())

        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["temperature"] = self.config.rollout.temperature
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            # output, _ = self.ref_policy.compute_log_prob(data=data, calculate_entropy=False)
            # output = DataProto.from_dict(tensors={"ref_log_prob": output})
            image_embed, video_embed = self.ref_policy.extract_feature(data=data, split=True)
            if image_embed is not None and video_embed is not None:
                embeds = {"image_embed": image_embed, "video_embed": video_embed}
            elif image_embed is not None and video_embed is None:
                embeds = {"image_embed": image_embed}
            elif image_embed is None and video_embed is not None:
                embeds = {"video_embed": video_embed}
            else:
                raise ValueError("Both image_embed and video_embed are None. At least one of them must be provided.")

            output = DataProto.from_dict(non_tensors=embeds)
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1 and fsdp_version(self.ref_policy.encoder_module) == 1:
            self.ref_policy.encoder_module._handle.reshard(True)

        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        # only support save and load ckpt for actor
        assert self._is_actor
        import torch

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        self.checkpoint_manager.save_checkpoint(local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        self.checkpoint_manager.load_checkpoint(local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.actor_optimizer)
            
            

class ActorRolloutRefWorker_llm(Worker, DistProfilerExtension):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str):
        super().__init__()
        self.config = config
        import torch.distributed

        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(
                backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}",
                rank=rank,
                world_size=world_size,
                init_method=os.environ.get("DIST_INIT_METHOD", None),
                timeout=_get_dist_timeout(),
            )

        self._prof = None
        self._prof_enabled = False
        self._prof_active = False
        self._prof_logdir = os.getenv("PROF_LOGDIR", "/workspace/yym/RLHF/verl-disaggregate/log/trace/col")
        self._enable_prof_env = bool(int(os.getenv("ENABLE_PROFILER", "0")))

        # build device mesh for FSDP
        world_size = torch.distributed.get_world_size()
        # TODO(sgm): support FSDP hybrid shard for larger model
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=self.config.actor.fsdp_config.fsdp_size)

        # build device mesh for Ulysses Sequence Parallel
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.actor.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"])

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        self.role = role
        # 这部分改动同encoder
        assert self.role in ["llm_ref", "llm_actor_rollout"]

        self._is_actor = self.role == "llm_actor_rollout"
        self._is_rollout = self.role == "llm_actor_rollout"
        self._is_ref = self.role == "llm_ref"

        profiler_config = omega_conf_to_dataclass(config.get("profiler"))
        DistProfilerExtension.__init__(
            self, Profiler(config=profiler_config, task=self.role)
        )

        self._is_offload_param = False
        self._is_offload_optimizer = False
        if self._is_actor:
            self._is_offload_param = self.config.actor.fsdp_config.get("param_offload", False)
            self._is_offload_optimizer = self.config.actor.fsdp_config.get("optimizer_offload", False)
        elif self._is_ref:
            # TODO: it seems that manual offload is slowly than FSDP offload
            self._is_offload_param = self.config.ref.fsdp_config.get("param_offload", False)

        # normalize config
        if self._is_actor:
            self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
            self.config.actor.ppo_mini_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
            assert self.config.actor.ppo_mini_batch_size > 0, (
                f"ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be larger than 0 after "
                f"normalization"
            )
            # micro bsz
            if self.config.actor.ppo_micro_batch_size is not None:
                self.config.actor.ppo_micro_batch_size //= (
                    self.device_mesh.size() // self.ulysses_sequence_parallel_size
                )
                self.config.actor.ppo_micro_batch_size_per_gpu = self.config.actor.ppo_micro_batch_size

            if self.config.actor.ppo_micro_batch_size_per_gpu is not None:
                assert self.config.actor.ppo_mini_batch_size % self.config.actor.ppo_micro_batch_size_per_gpu == 0, (
                    f"normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be divisible by "
                    f"ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}"
                )
                assert self.config.actor.ppo_mini_batch_size // self.config.actor.ppo_micro_batch_size_per_gpu > 0, (
                    f"normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be larger than "
                    f"ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}"
                )

        # normalize rollout config
        if self._is_rollout and self.config.rollout.log_prob_micro_batch_size is not None:
            self.config.rollout.log_prob_micro_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
            self.config.rollout.log_prob_micro_batch_size_per_gpu = self.config.rollout.log_prob_micro_batch_size
        # normalize ref config
        if self._is_ref and self.config.ref.log_prob_micro_batch_size is not None:
            self.config.ref.log_prob_micro_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
            self.config.ref.log_prob_micro_batch_size_per_gpu = self.config.ref.log_prob_micro_batch_size

    def _build_model_optimizer(
        self,
        model_path,
        fsdp_config: FSDPEngineConfig,
        optim_config,
        override_model_config,
        use_remove_padding=False,
        use_fused_kernels=False,
        enable_gradient_checkpointing=False,
        trust_remote_code=False,
        use_liger=False,
        role="actor",
        enable_activation_offload=False,
    ):
        from torch import optim
        from torch.distributed.fsdp import CPUOffload, MixedPrecision
        # from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq, Qwen2_5_VLTextModel

        from verl.utils.model import get_generation_config, print_model_size, update_model_config
        from verl.utils.torch_dtypes import PrecisionType

        assert role in ["actor", "ref"]

        log_gpu_memory_usage(f"Before init {role} from HF AutoModel", logger=logger)
        local_path = copy_to_local(model_path)

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code)

        torch_dtype = fsdp_config.get("model_dtype", None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # override model kwargs
        actor_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code, attn_implementation="flash_attention_2")

        self.generation_config = get_generation_config(local_path, trust_remote_code=trust_remote_code)

        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
        if self.rank == 0:
            print(f"Model config after override: {actor_model_config}")

        # NOTE(fix me): tie_word_embedding causes meta_tensor init to hang
        init_context = get_init_weight_context_manager(use_meta_tensor=not actor_model_config.tie_word_embeddings, mesh=self.device_mesh)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from verl.models.transformers.qwen2_5_vl import CustomQwen2_5_VLModel
            actor_module_class = CustomQwen2_5_VLModel

            actor_module = actor_module_class.from_pretrained(
                pretrained_model_name_or_path=local_path,
                # only for disaggregate test
                torch_dtype=torch.float16,
                config=actor_model_config,
                trust_remote_code=trust_remote_code,
            )
            # 遇到报错，actor_module在meta，fsdp在cuda:0
            actor_module = actor_module.to_empty(device=torch.device(device_name), recurse=True)
            #lm_head_sd = torch.load("/workspace/models/qwen2.5vl-lm_head.pt")
            llm_sd = torch.load("/workspace/models/qwen2.5vl-3b-llm.pt")
            #actor_module.lm_head.load_state_dict(lm_head_sd)
            actor_module.language_model.load_state_dict(llm_sd)

            # Apply Liger kernel to the model if use_liger is set to True
            if use_liger:
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(model=actor_module)
                
            fused_kernel_options = self.config.model.get("fused_kernel_options", None)
            fused_kernels_backend = (
                fused_kernel_options.get("impl_backend", None) if fused_kernel_options is not None else None
            )

            apply_monkey_patch(
                model=actor_module,
                use_remove_padding=use_remove_padding,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
                use_fused_kernels=use_fused_kernels,
                fused_kernels_backend=fused_kernels_backend,
            )
            actor_module.to(torch_dtype)

            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage(f"After init {role} from HF AutoModel", logger=logger)

        # We wrap FSDP for rollout as well
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)
        # 由于改了类名，这个地方匹配不到，自行编写config，仅针对本测试的代码
        wrap_config = {"transformer_layer_cls_to_wrap": ["Qwen2_5_VLDecoderLayer"],}
        # auto_wrap_policy = get_fsdp_wrap_policy(module=actor_module, config=fsdp_config.get("wrap_policy", None))
        auto_wrap_policy = get_fsdp_wrap_policy(module=actor_module, config=wrap_config)

        if self._is_rollout and self.config.rollout.name == "hf":
            # TODO(zhangchi.usc1992, shengguangming) fix me. Current, auto_wrap_policy causes HFRollout to hang in Gemma
            auto_wrap_policy = None

        print(f"wrap_policy: {auto_wrap_policy}")

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        # TODO: add transformer policy
        # We force reference policy to use CPUOffload to save memory.
        # We force turn off CPUOffload for actor because it causes incorrect results when using grad accumulation
        cpu_offload = None if role == "actor" else CPUOffload(offload_params=True)
        fsdp_strategy = self.config.actor.strategy
        if fsdp_strategy == "fsdp":
            actor_module_fsdp = FSDP(
                actor_module,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,  # zero3
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                use_orig_params=fsdp_config.get("use_orig_params", False),
                forward_prefetch=fsdp_config.get("forward_prefetch", False),
            )
        elif fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True)
            if role == "actor" and fsdp_config.offload_policy:
                cpu_offload = CPUOffloadPolicy(pin_memory=True)
                self._is_offload_param = False
                self._is_offload_optimizer = False
            else:
                cpu_offload = None if role == "actor" else CPUOffloadPolicy(pin_memory=True)

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": fsdp_config.reshard_after_forward,
            }
            full_state = actor_module.state_dict()
            apply_fsdp2(actor_module, fsdp_kwargs, fsdp_config)
            fsdp2_load_full_state_dict(actor_module, full_state, fsdp_mesh, cpu_offload)
            actor_module_fsdp = actor_module
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        if enable_activation_offload:
            enable_activation_offloading(actor_module_fsdp, fsdp_strategy, enable_gradient_checkpointing)
        log_gpu_memory_usage(f"After {role} FSDP init", logger=logger)

        # TODO: add more optimizer args into config
        if role == "actor" and optim_config is not None:
            from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

            actor_optimizer = optim.AdamW(
                actor_module_fsdp.parameters(),
                lr=optim_config.lr,
                betas=optim_config.get("betas", (0.9, 0.999)),
                weight_decay=optim_config.get("weight_decay", 1e-2),
            )

            total_steps = optim_config.get("total_training_steps", 0)
            num_warmup_steps = int(optim_config.get("lr_warmup_steps", -1))
            warmup_style = optim_config.get("warmup_style", "constant")
            min_lr_ratio = optim_config.get("min_lr_ratio", 0.0)
            num_cycles = optim_config.get("num_cycles", 0.5)
            if num_warmup_steps < 0:
                num_warmup_steps_ratio = optim_config.get("lr_warmup_steps_ratio", 0.0)
                num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

            if warmup_style == "constant":
                actor_lr_scheduler = get_constant_schedule_with_warmup(optimizer=actor_optimizer, num_warmup_steps=num_warmup_steps)
            elif warmup_style == "cosine":
                actor_lr_scheduler = get_cosine_schedule_with_warmup(
                    optimizer=actor_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps,
                    min_lr_ratio=min_lr_ratio,
                    num_cycles=num_cycles,)
            else:
                raise NotImplementedError(f"Warmup style {warmup_style} is not supported")

            log_gpu_memory_usage(f"After {role} optimizer init", logger=logger)
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config

    def _build_rollout(self, trust_remote_code=False):
        from torch.distributed.device_mesh import init_device_mesh

        # TODO(sgm): support FSDP hybrid shard for larger model
        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
        rollout_device_mesh = init_device_mesh(device_name, mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"])
        rollout_name = self.config.rollout.name
        if rollout_name == "hf":
            from verl.workers.rollout import HFRollout
            from verl.workers.sharding_manager.base import BaseShardingManager

            rollout = HFRollout(module=self.actor_module_fsdp, config=self.config.rollout)
            rollout_sharding_manager = BaseShardingManager()
            # TODO: a sharding manager that do nothing?

        elif rollout_name == "vllm":
            from verl.workers.rollout.vllm_rollout import vLLMRollout
            from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager

            log_gpu_memory_usage(f"Before building {rollout_name} rollout", logger=logger)
            local_path = copy_to_local(self.config.model.path, use_shm=self.config.model.get("use_shm", False))
            from verl.workers.rollout.vllm_rollout import vLLMAsyncRollout
            from verl.models.vllm.monkey_patch import vllm_monkey_patch_llm
            vllm_monkey_patch_llm()

            vllm_rollout_cls = vLLMRollout if self.config.rollout.mode == "sync" else vLLMAsyncRollout
            rollout = vllm_rollout_cls(
                model_path=local_path,
                config=self.config.rollout,
                tokenizer=self.tokenizer,
                model_hf_config=self.actor_model_config,
                device_mesh=rollout_device_mesh,
                trust_remote_code=trust_remote_code,
            )

            log_gpu_memory_usage(f"After building {rollout_name} rollout", logger=logger)
            full_params = torch.distributed.get_world_size() == 1
            rollout_sharding_manager = FSDPVLLMShardingManager(
                module=self.actor_module_fsdp,
                inference_engine=rollout.inference_engine,
                model_config=self.actor_model_config,
                rollout_config=self.config.rollout,
                full_params=full_params,
                device_mesh=rollout_device_mesh,
                offload_param=self._is_offload_param,
                load_format=self.config.rollout.load_format,
                layered_summon=self.config.rollout.get("layered_summon", False),
            )
            log_gpu_memory_usage("After building sharding manager", logger=logger)

        elif rollout_name == "sglang":
            from verl.workers.rollout.sglang_rollout.sglang_rollout import SGLangRollout

            # NOTE(linjunrong): Due to recent fp8 support in SGLang. Now importing any symbol relate to
            # SGLang's model_runner would check CUDA device capability. However, due to verl's setting,
            # the main process of ray can not find any CUDA device, which would potentially lead to:
            # "RuntimeError: No CUDA GPUs are available".
            # For this reason, sharding_manager.__init__ should not import FSDPSGLangShardingManager and
            # we import it here use the abs path.
            # check: https://github.com/sgl-project/sglang/blob/00f42707eaddfc2c0528e5b1e0094025c640b7a0/python/sglang/srt/layers/quantization/fp8_utils.py#L76
            from verl.workers.sharding_manager.fsdp_sglang import FSDPSGLangShardingManager

            log_gpu_memory_usage(f"Before building {rollout_name} rollout", logger=logger)
            local_path = copy_to_local(self.config.model.llm.path)
            rollout = SGLangRollout(
                actor_module=local_path,
                config=self.config.rollout,
                tokenizer=self.tokenizer,
                model_hf_config=self.actor_model_config,
                trust_remote_code=trust_remote_code,
            )
            log_gpu_memory_usage(f"After building {rollout_name} rollout", logger=logger)

            if torch.distributed.get_world_size() == 1:
                self.config.rollout.load_format = "dummy_hf"
            rollout_sharding_manager = FSDPSGLangShardingManager(
                module=self.actor_module_fsdp,
                inference_engine=rollout.inference_engine,
                model_config=self.actor_model_config,
                full_params="hf" in self.config.rollout.load_format,
                device_mesh=rollout_device_mesh,
                offload_param=self._is_offload_param,
            )
            log_gpu_memory_usage("After building sharding manager", logger=logger)

        elif rollout_name == "sglang_async":
            from verl.workers.rollout.sglang_rollout import AsyncSGLangRollout
            from verl.workers.sharding_manager.fsdp_sglang import FSDPAsyncSGLangShardingManager

            log_gpu_memory_usage(f"Before building {rollout_name} rollout", logger=None)
            rollout = AsyncSGLangRollout(
                actor_module=self.config.model.llm.path,
                config=self.config.rollout,
                tokenizer=self.tokenizer,
                model_hf_config=self.actor_model_config,
                trust_remote_code=trust_remote_code,
            )
            log_gpu_memory_usage(f"After building {rollout_name} rollout", logger=None)

            if torch.distributed.get_world_size() == 1:
                self.config.rollout.load_format = "dummy_hf"
            rollout_sharding_manager = FSDPAsyncSGLangShardingManager(
                module=self.actor_module_fsdp,
                inference_engine=rollout._engine,
                model_config=self.actor_model_config,
                full_params="hf" in self.config.rollout.load_format,
                device_mesh=rollout_device_mesh,
            )
            log_gpu_memory_usage("After building sharding manager", logger=None)

        else:
            raise NotImplementedError(f"Rollout name: {self.config.rollout.name} is not supported")

        return rollout, rollout_sharding_manager

    @register(dispatch_mode=Dispatch.ALL_TO_ALL)
    def prof_start(self, wait=3, warmup=1, active=1, repeat=1):
        if self.rank != 0:
            return
        print(f"prof start rank{self.rank}!")
        if not self._enable_prof_env or self._prof_active:
            return
        subdir = f"{self.role}_rank{self.rank}_local{self._local_rank}"
        outdir = os.path.join(self._prof_logdir, subdir)
        os.makedirs(outdir, exist_ok=True)
        self._prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(outdir),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_modules=False,
        )
        self._prof.__enter__()
        self._prof_enabled = True
        self._prof_active  = True

    @register(dispatch_mode=Dispatch.ALL_TO_ALL)
    def prof_step(self):
        if self.rank != 0:
            return
        print(f"prof step rank{self.rank}!")
        if self._prof_enabled and self._prof is not None:
            self._prof.step()

    @register(dispatch_mode=Dispatch.ALL_TO_ALL)
    def prof_stop(self):
        if self.rank != 0:
            return
        print(f"prof stop rank{self.rank}!")
        save_file_name = f"/prof_{self.role}_rank_{self.rank}.json"
        self._prof.export_chrome_trace(self._prof_logdir + save_file_name)
        if self._prof_enabled and self._prof is not None:
            self._prof.__exit__(None, None, None)
        self._prof = None
        self._prof_enabled = False
        self._prof_active  = False

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl.workers.actor import DataParallelPPOActor

        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        from omegaconf import OmegaConf

        override_model_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))

        use_remove_padding = self.config.model.get("use_remove_padding", False)
        use_shm = self.config.model.get("use_shm", False)
        use_fused_kernels = self.config.model.get("use_fused_kernels", False)
        
        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = omega_conf_to_dataclass(self.config.actor.fsdp_config)
            else:
                optim_config = None
                fsdp_config = FSDPEngineConfig()
            self.actor_module_fsdp, self.actor_optimizer, self.actor_lr_scheduler, self.actor_model_config = self._build_model_optimizer(
                model_path=self.config.model.llm.path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                enable_gradient_checkpointing=self.config.model.get("enable_gradient_checkpointing", False),
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="actor",
                enable_activation_offload=self.config.model.get("enable_activation_offload", False),
            )

            # get the original unwrapped module
            if fsdp_version(self.actor_module_fsdp) == 1:
                self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                log_gpu_memory_usage("After offload actor model during init", logger=logger)

            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage("After offload actor optimizer during init", logger=logger)
        # load from checkpoint
        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            with open_dict(self.config.actor):
                self.config.actor.use_remove_padding = use_remove_padding
            self.actor = DataParallelPPOActor(config=self.config.actor, actor_module=self.actor_module_fsdp, actor_optimizer=self.actor_optimizer)

        if self._is_rollout:
            self.rollout, self.rollout_sharding_manager = self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))
            # pass

        if self._is_ref:
            self.ref_module_fsdp = self._build_model_optimizer(
                model_path=self.config.model.llm.path,
                fsdp_config=self.config.ref.fsdp_config,
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="ref",
            )[0]
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
                self.config.ref.use_fused_kernels = use_fused_kernels
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)

        if self._is_actor and self._is_rollout :
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_config=self.config.actor.checkpoint,
            )

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        # Support all hardwares
        data = data.to(get_device_id())

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_device_id())

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            # perform training
            with Timer(name="update_policy", logger=None) as timer:
                metrics, encoder_gradients = self.actor.update_policy_llm(data=data)
            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu/actor"] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
            metrics["perf/max_memory_allocated_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
            metrics["perf/max_memory_reserved_gb"] = torch.cuda.max_memory_reserved() / (1024**3)
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            self.actor_lr_scheduler.step()
            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr

            # TODO: here, we should return all metrics
            # output = DataProto(meta_info={"metrics": metrics})
            output = DataProto.from_dict(meta_info={"metrics": metrics}, non_tensors=encoder_gradients)
            output = self.ulysses_sharding_manager.postprocess_data(data=output)
            output = output.to("cpu")

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during update_actor", logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
            log_gpu_memory_usage("After offload actor optimizer during update_actor", logger=logger)

        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        # Support all hardwares
        prompts = prompts.to(get_device_id())

        assert self._is_rollout

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id if self.generation_config is not None else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id if self.generation_config is not None else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        timing_generate = {}
        with self.rollout_sharding_manager:
            log_gpu_memory_usage("After entering rollout sharding manager", logger=logger)

            prompts = self.rollout_sharding_manager.preprocess_data(prompts)
            with simple_timer("generate_sequences", timing_generate):
                output = self.rollout.generate_sequences(prompts=prompts)

            log_gpu_memory_usage("After rollout generation", logger=logger)

            output = self.rollout_sharding_manager.postprocess_data(output)

        output = output.to("cpu")

        timing_generate.update(self.rollout_sharding_manager.timing)
        reduce_timing_across_ranks = bool(int(os.getenv("VERL_REDUCE_TIMING_ACROSS_RANKS", "1")))
        if reduce_timing_across_ranks:
            # We calculate the average timing across all ranks to make sure meta_info["timing"] is the same.
            timing_generate = reduce_timing(timing_generate)
        output.meta_info["timing"] = timing_generate

        # clear kv cache
        get_torch_device().empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob_llm(self, data: DataProto):
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        # Support all hardwares
        data = data.to(get_device_id())
        # we should always recompute old_log_probs when it is HybridEngine
        data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = self.config.rollout.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature
        # perform recompute log_prob
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output, entropys = self.actor.compute_log_prob_llm(data=data, calculate_entropy=True)
            output = DataProto.from_dict(
                tensors={"old_log_probs": output, "entropys": entropys},
                meta_info={"temperature": self.config.rollout.temperature},
            )
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
            self.actor.actor_module._handle.reshard(True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during compute_log_prob", logger=logger)

        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob_llm(self, data: DataProto):
        assert self._is_ref

        # Support all hardwares
        data = data.to(get_device_id())

        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["temperature"] = self.config.rollout.temperature
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output, _ = self.ref_policy.compute_log_prob_llm(data=data, calculate_entropy=False)
            output = DataProto.from_dict(tensors={"ref_log_prob": output})
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1 and fsdp_version(self.ref_policy.actor_module) == 1:
            self.ref_policy.actor_module._handle.reshard(True)

        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        # only support save and load ckpt for actor
        assert self._is_actor
        import torch

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        self.checkpoint_manager.save_checkpoint(local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        self.checkpoint_manager.load_checkpoint(local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.actor_optimizer)



class CriticWorker(Worker, DistProfilerExtension):
    def __init__(self, config: FSDPCriticConfig):
        Worker.__init__(self)
        DistProfilerExtension.__init__(self, DistProfiler(rank=self.rank, config=config.get("profiler")))
        import torch.distributed

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=get_nccl_backend(),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
                timeout=_get_dist_timeout(),
            )
        self.config: FSDPCriticConfig = config

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
            )

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # set FSDP offload params
        self._is_offload_param = self.config.model.fsdp_config.param_offload
        self._is_offload_optimizer = self.config.model.fsdp_config.optimizer_offload

        # normalize config
        self.config.ppo_mini_batch_size *= self.config.rollout_n
        self.config.ppo_mini_batch_size //= torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
        if self.config.ppo_micro_batch_size is not None:
            self.config.ppo_micro_batch_size //= (
                torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
            )
            self.config.forward_micro_batch_size //= (
                torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
            )
            self.config.ppo_micro_batch_size_per_gpu = self.config.ppo_micro_batch_size
            self.config.forward_micro_batch_size_per_gpu = self.config.forward_micro_batch_size

        if self.config.ppo_micro_batch_size_per_gpu is not None:
            assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size_per_gpu == 0, (
                f"normalized ppo_mini_batch_size {self.config.ppo_mini_batch_size} should be divisible by "
                f"ppo_micro_batch_size_per_gpu {self.config.ppo_micro_batch_size_per_gpu}"
            )
            assert self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu > 0, (
                f"normalized ppo_mini_batch_size {self.config.ppo_mini_batch_size} should be larger than "
                f"ppo_micro_batch_size_per_gpu {self.config.ppo_micro_batch_size_per_gpu}"
            )
        self._is_lora = self.config.model.get("lora_rank", 0) > 0

    def _build_critic_model_optimizer(self, config):
        # the following line is necessary
        from torch import optim
        from torch.distributed.fsdp import MixedPrecision

        from verl.utils.model import load_valuehead_model, print_model_size
        from verl.utils.torch_dtypes import PrecisionType

        use_shm = config.model.get("use_shm", False)
        local_path = copy_to_local(config.model.path, use_shm=use_shm)
        # note that the tokenizer between actor and critic may be different. So override tokenizer info with actor info
        # using random initialized model from any architecture. May not be the same as Actor.

        tokenizer_path = copy_to_local(config.model.tokenizer_path, use_shm=use_shm)
        self.tokenizer = hf_tokenizer(tokenizer_path, trust_remote_code=config.model.get("trust_remote_code", False))
        self.processor = hf_processor(tokenizer_path, trust_remote_code=config.model.get("trust_remote_code", False))

        if self.config.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.model.custom_chat_template
            else:
                self.tokenizer.chat_template = self.config.model.custom_chat_template
        override_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_config)
        if self.rank == 0:
            print(f"Critic overriding config {override_config_kwargs}")

        torch_dtype = self.config.model.fsdp_config.get("model_dtype", "fp32")
        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        from transformers import AutoConfig

        critic_model_config = AutoConfig.from_pretrained(
            local_path,
            attn_implementation="flash_attention_2",
            trust_remote_code=config.model.get("trust_remote_code", False),
        )
        critic_model_config.num_labels = 1
        # patch for kimi-vl
        if getattr(critic_model_config, "model_type", None) == "kimi_vl":
            critic_model_config.text_config.topk_method = "greedy"

        init_context = get_init_weight_context_manager(
            use_meta_tensor=not critic_model_config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            critic_model_config.classifier_dropout = 0.0
            critic_model_config.hidden_dropout = "0"
            critic_model_config.summary_dropout_prob = 0.0

            critic_module = load_valuehead_model(
                local_path,
                torch_dtype,
                critic_model_config,
                config.model.get("trust_remote_code", False),
            )

            use_remove_padding = config.model.get("use_remove_padding", False)

            apply_monkey_patch(
                model=critic_module,
                use_remove_padding=use_remove_padding,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
            )

            # some parameters may not in torch_dtype
            critic_module.to(torch_dtype)

            if config.model.get("enable_gradient_checkpointing", False):
                critic_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        if self._is_lora:
            print("Applying LoRA to critic module")
            critic_module.enable_input_require_grads()
            # Convert config to regular Python types before creating PEFT model
            lora_config = {
                "task_type": TaskType.CAUSAL_LM,
                "r": self.config.model.lora_rank,
                "lora_alpha": self.config.model.lora_alpha,
                "target_modules": convert_to_regular_types(self.config.model.target_modules),
                "bias": "none",
            }
            critic_module = get_peft_model(critic_module, LoraConfig(**lora_config))

        if self.rank == 0:
            print_model_size(critic_module)

        self.critic_model_config = critic_model_config

        fsdp_config = self.config.model.fsdp_config
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(
            module=critic_module,
            config=self.config.model.fsdp_config.wrap_policy,
            is_lora=self.config.model.get("lora_rank", 0) > 0,
        )

        log_gpu_memory_usage("Before critic FSDP", logger=None)

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        # Note: We force turn off CPUOffload for critic because it causes incorrect results when using grad accumulation
        if config.strategy == "fsdp":
            critic_module = FSDP(
                critic_module,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                forward_prefetch=self.config.model.fsdp_config.forward_prefetch,
                device_mesh=self.device_mesh,
                cpu_offload=None,
            )
        elif config.strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(
                param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True
            )
            offload_policy = None
            if fsdp_config.offload_policy:
                self._is_offload_param = False
                self._is_offload_optimizer = False
                offload_policy = CPUOffloadPolicy(pin_memory=True)

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": offload_policy,
                "reshard_after_forward": fsdp_config.reshard_after_forward,
            }
            full_state = critic_module.state_dict()
            apply_fsdp2(critic_module, fsdp_kwargs, fsdp_config)
            fsdp2_load_full_state_dict(critic_module, full_state, fsdp_mesh, offload_policy)
        else:
            raise NotImplementedError(f"Unknown strategy {config.strategy}")

        if config.model.get("enable_activation_offload", False):
            enable_gradient_checkpointing = config.model.get("enable_gradient_checkpointing", False)
            enable_activation_offloading(critic_module, config.strategy, enable_gradient_checkpointing)

        log_gpu_memory_usage("After critic FSDP", logger=None)

        critic_optimizer = optim.AdamW(
            critic_module.parameters(),
            lr=config.optim.lr,
            betas=config.optim.get("betas", (0.9, 0.999)),
            weight_decay=config.optim.get("weight_decay", 1e-2),
        )

        total_steps = config.optim.get("total_training_steps", 0)
        num_warmup_steps = int(config.optim.get("lr_warmup_steps", -1))
        warmup_style = config.optim.get("warmup_style", "constant")
        if num_warmup_steps < 0:
            num_warmup_steps_ratio = config.optim.get("lr_warmup_steps_ratio", 0.0)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        if self.rank == 0:
            print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

        from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

        if warmup_style == "constant":
            critic_lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=critic_optimizer, num_warmup_steps=num_warmup_steps
            )
        elif warmup_style == "cosine":
            min_lr_ratio = config.optim.get("min_lr_ratio", 0.0)
            num_cycles = config.optim.get("num_cycles", 0.5)
            critic_lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=critic_optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps,
                min_lr_ratio=min_lr_ratio,
                num_cycles=num_cycles,
            )
        else:
            raise NotImplementedError(f"Warmup style {warmup_style} is not supported")

        return critic_module, critic_optimizer, critic_lr_scheduler

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        from verl.workers.critic import DataParallelPPOCritic

        self.critic_module, self.critic_optimizer, self.critic_lr_scheduler = self._build_critic_model_optimizer(
            self.config
        )

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
            log_gpu_memory_usage("After offload critic model during init", logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)
            log_gpu_memory_usage("After offload critic optimizer during init", logger=logger)

        self.critic = DataParallelPPOCritic(
            config=self.config, critic_module=self.critic_module, critic_optimizer=self.critic_optimizer
        )

        self.flops_counter = FlopsCounter(self.critic_model_config)
        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.critic_module,
            optimizer=self.critic_optimizer,
            lr_scheduler=self.critic_lr_scheduler,
            processing_class=self.processor if self.processor is not None else self.tokenizer,
            checkpoint_config=self.config.checkpoint,
        )

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    #@DistProfiler.annotate(color="cyan")
    def compute_values(self, data: DataProto):
        # Support all hardwares
        data = data.to(get_device_id())

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)
        micro_batch_size = self.config.forward_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["max_token_len"] = self.config.forward_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz
        # perform forward computation
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            values = self.critic.compute_values(data=data)
            output = DataProto.from_dict(tensors={"values": values})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        output = output.to("cpu")
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    #@DistProfiler.annotate(color="pink")
    def update_critic(self, data: DataProto):
        # Support all hardwares
        data = data.to(get_device_id())
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.critic_optimizer, device_id=get_device_id())

        # perform forward computation
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)

            with Timer(name="update_critic", logger=None) as timer:
                metrics = self.critic.update_critic(data=data)
            delta_time = timer.last

            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu/critic"] = estimated_flops * self.config.ppo_epochs / promised_flops / self.world_size

            lr = self.critic_lr_scheduler.get_last_lr()[0]
            metrics["critic/lr"] = lr
            self.critic_lr_scheduler.step()

            output = DataProto(batch=None, meta_info={"metrics": metrics})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        import torch

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)

        self.checkpoint_manager.save_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
        )

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=True):
        import torch

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)

        self.checkpoint_manager.load_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.critic_optimizer)


# TODO(sgm): we may need to extract it to dp_reward_model.py
class RewardModelWorker(Worker, DistProfilerExtension):
    """
    Note that we only implement the reward model that is subclass of AutoModelForTokenClassification.
    """

    def __init__(self, config):
        Worker.__init__(self)
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=omega_conf_to_dataclass(config.get("profiler")))
        )

        import torch.distributed

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=get_nccl_backend(),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
                timeout=_get_dist_timeout(),
            )
        self.config = config

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
            )

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        self.use_remove_padding = self.config.model.get("use_remove_padding", False)

        # normalize config
        if self.config.micro_batch_size is not None:
            self.config.micro_batch_size //= torch.distributed.get_world_size()
            self.config.micro_batch_size_per_gpu = self.config.micro_batch_size

    def _build_model(self, config):
        # the following line is necessary
        from torch.distributed.fsdp import CPUOffload
        from transformers import AutoConfig, AutoModelForTokenClassification

        use_shm = config.model.get("use_shm", False)
        # download the checkpoint from hdfs
        local_path = copy_to_local(config.model.path, use_shm=use_shm)

        if self.config.model.input_tokenizer is None:
            self._do_switch_chat_template = False
        else:
            self._do_switch_chat_template = True
            input_tokenizer_local_path = copy_to_local(config.model.input_tokenizer, use_shm=use_shm)
            self.input_tokenizer = hf_tokenizer(
                input_tokenizer_local_path, trust_remote_code=config.model.get("trust_remote_code", False)
            )
            self.tokenizer = hf_tokenizer(local_path, trust_remote_code=config.model.get("trust_remote_code", False))

        trust_remote_code = config.model.get("trust_remote_code", False)
        model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        model_config.num_labels = 1

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not model_config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_config.classifier_dropout = 0.0
            reward_module = AutoModelForTokenClassification.from_pretrained(
                pretrained_model_name_or_path=local_path,
                config=model_config,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )

            apply_monkey_patch(
                model=reward_module,
                use_remove_padding=config.model.get("use_remove_padding", False),
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
            )

            reward_module.to(torch.bfloat16)

        auto_wrap_policy = get_fsdp_wrap_policy(module=reward_module, config=self.config.model.fsdp_config)

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        if config.strategy == "fsdp":
            reward_module = FSDP(
                reward_module,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,  # zero3
                sync_module_states=True,
                cpu_offload=CPUOffload(offload_params=True),
                forward_prefetch=self.config.model.fsdp_config.forward_prefetch,
                device_mesh=self.device_mesh,
            )
        elif config.strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            cpu_offload = CPUOffloadPolicy(pin_memory=True)
            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "offload_policy": cpu_offload,
                "reshard_after_forward": config.model.fsdp_config.reshard_after_forward,
            }
            full_state = reward_module.state_dict()
            apply_fsdp2(reward_module, fsdp_kwargs, config.model.fsdp_config)
            fsdp2_load_full_state_dict(reward_module, full_state, fsdp_mesh, cpu_offload)
        else:
            raise NotImplementedError(f"Unknown strategy: {config.strategy}")
        return reward_module

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))
        self.reward_module = self._build_model(config=self.config)

    def _forward_micro_batch(self, micro_batch):
        if is_cuda_available:
            from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
        elif is_npu_available:
            from transformers.integrations.npu_flash_attention import (
                index_first_axis,
                pad_input,
                rearrange,
                unpad_input,
            )

        from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs

        with torch.no_grad(), torch.autocast(device_type=device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                # pad and slice the inputs if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size
                    )

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.reward_module(
                    input_ids=input_ids_rmpad, attention_mask=None, position_ids=position_ids_rmpad, use_cache=False
                )
                reward_rmpad = output.logits
                reward_rmpad = reward_rmpad.squeeze(0)  # (total_nnz)

                # gather output if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    reward_rmpad = gather_outputs_and_unpad(
                        reward_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                    )

                # pad it back
                rm_score = pad_input(reward_rmpad, indices=indices, batch=batch_size, seqlen=seqlen).squeeze(-1)
            else:
                output = self.reward_module(
                    input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False
                )
                rm_score = output.logits  # (batch_size, seq_len, 1)
                rm_score = rm_score.squeeze(-1)

            # extract the result of the last valid token
            eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
            rm_score = rm_score[torch.arange(batch_size), eos_mask_idx]
            return rm_score

    def _expand_to_token_level(self, data: DataProto, scores: torch.Tensor):
        batch_size = data.batch.batch_size[0]
        # expand as token_level_reward
        attention_mask = data.batch["attention_mask"]
        position_ids = data.batch["position_ids"]
        response_length = data.batch["responses"].shape[-1]
        if position_ids.dim() == 3:  # qwen2vl mrope [bs, 3, seq_len]
            position_ids = position_ids[:, 0, :]
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
        token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)  # (bsz, seqlen)
        token_level_scores[torch.arange(batch_size), eos_mask_idx] = scores

        # select the response part
        token_level_scores = token_level_scores[:, -response_length:]

        return token_level_scores

    def _switch_chat_template(self, data: DataProto):
        src_max_length = data.batch["attention_mask"].shape[-1]

        src_tokenizer = self.input_tokenizer
        target_tokenizer = self.tokenizer

        rm_input_ids = []
        rm_attention_mask = []

        for i in range(data.batch.batch_size[0]):
            if not isinstance(data.non_tensor_batch["raw_prompt"][i], list | np.ndarray):
                raise TypeError(
                    f"raw_prompt must be a list or numpy array, got {type(data.non_tensor_batch['raw_prompt'][i])}"
                )

            # extract raw prompt
            chat: list = list(data.non_tensor_batch["raw_prompt"][i])

            # extract response
            response_ids = data.batch["responses"][i]
            response_length = response_ids.shape[-1]
            valid_response_length = data.batch["attention_mask"][i][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            response = src_tokenizer.decode(valid_response_ids)
            # remove bos and eos
            response = response.replace(src_tokenizer.eos_token, "")

            chat.append({"role": "assistant", "content": response})

            prompt_with_chat_template = target_tokenizer.apply_chat_template(
                chat, add_generation_prompt=False, tokenize=False
            )
            if self.rank == 0 and i == 0:
                # for debugging purpose
                print(f"Switch template. chat: {prompt_with_chat_template}")

            # the maximum length is actually determined by the reward model itself
            max_length = self.config.get("max_length", src_max_length)
            if max_length is None:
                max_length = src_max_length

            model_inputs = target_tokenizer(prompt_with_chat_template, return_tensors="pt", add_special_tokens=False)
            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                max_length=max_length,
                pad_token_id=target_tokenizer.pad_token_id,
                left_pad=False,  # right padding
                truncation=self.config.get("truncation", "right"),
            )  # truncate from the right

            rm_input_ids.append(input_ids)
            rm_attention_mask.append(attention_mask)

        rm_input_ids = torch.cat(rm_input_ids, dim=0)
        rm_attention_mask = torch.cat(rm_attention_mask, dim=0)

        rm_position_ids = compute_position_id_with_mask(rm_attention_mask)

        rm_inputs = {"input_ids": rm_input_ids, "attention_mask": rm_attention_mask, "position_ids": rm_position_ids}

        return DataProto.from_dict(rm_inputs)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    #@DistProfiler.annotate(color="brown")
    def compute_rm_score(self, data: DataProto):
        import itertools

        from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches

        # Support all hardwares
        data = data.to(get_device_id())
        if self._do_switch_chat_template:
            rm_data = self._switch_chat_template(data)
        else:
            rm_input_ids = data.batch["input_ids"]
            rm_attention_mask = data.batch["attention_mask"]
            rm_position_ids = data.batch["position_ids"]
            rm_inputs = {
                "input_ids": rm_input_ids,
                "attention_mask": rm_attention_mask,
                "position_ids": rm_position_ids,
            }
            rm_data = DataProto.from_dict(rm_inputs)

        # Support all hardwares
        rm_data.batch = rm_data.batch.to(get_device_id())

        # perform forward computation
        with self.ulysses_sharding_manager:
            rm_data = self.ulysses_sharding_manager.preprocess_data(data=rm_data)
            data = self.ulysses_sharding_manager.preprocess_data(data=data)

            use_dynamic_bsz = self.config.use_dynamic_bsz
            if use_dynamic_bsz:
                max_token_len = self.config.forward_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, indices = rearrange_micro_batches(batch=rm_data.batch, max_token_len=max_token_len)
            else:
                micro_batches = rm_data.batch.split(self.config.micro_batch_size_per_gpu)
            output = []
            for micro_batch in micro_batches:
                rm_score = self._forward_micro_batch(micro_batch)
                output.append(rm_score)
            scores = torch.cat(output, dim=0)  # (batch_size)

            if use_dynamic_bsz:
                indices = list(itertools.chain.from_iterable(indices))
                assert len(indices) == scores.size(0), f"{len(indices)} vs. {scores.size()}"
                revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                scores = scores[revert_indices]

            token_level_scores = self._expand_to_token_level(data, scores)
            # Note that this is only the scores, may not be the final rewards used to train RL
            output = DataProto.from_dict(tensors={"rm_scores": token_level_scores})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1 and fsdp_version(self.reward_module) == 1:
            self.reward_module._handle.reshard(True)

        output = output.to("cpu")
        return output


# ================================= Async related workers =================================
class AsyncActorRolloutRefWorker(ActorRolloutRefWorker):
    def _build_rollout(self, trust_remote_code=False):
        rollout, rollout_sharding_manager = super()._build_rollout(trust_remote_code)

        # NOTE: rollout is not actually initialized here, it's deferred
        # to be initialized by AsyncvLLMServer.

        self.vllm_tp_size = self.config.rollout.tensor_model_parallel_size
        self.vllm_dp_rank = int(os.environ["RANK"]) // self.vllm_tp_size
        self.vllm_tp_rank = int(os.environ["RANK"]) % self.vllm_tp_size

        # used for sleep/wake_up
        rollout.sharding_manager = rollout_sharding_manager

        return rollout, rollout_sharding_manager

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        raise NotImplementedError("AsyncActorRolloutRefWorker does not support generate_sequences")

    # ============================ vLLM related ============================

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    def execute_method(self, method: str | bytes, *args, **kwargs):
        """Called by ExternalRayDistributedExecutor collective_rpc."""
        return self.rollout.execute_method(method, *args, **kwargs)

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    def get_zeromq_address(self):
        return self.rollout.get_zeromq_address()

    # ============================ SGLang related ============================

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD, blocking=False)
    async def chat_completion(self, json_request):
        ret = await self.rollout.chat_completion(json_request)
        return ret

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD, blocking=False)
    async def generate(self, prompt_ids: list[int], sampling_params: dict[str, Any], request_id: str) -> list[int]:
        ret = await self.rollout.generate(prompt_ids, sampling_params, request_id)
        return ret

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    async def wake_up(self):
        if self.config.rollout.free_cache_engine:
            await self.rollout.wake_up()
        # return something to block the caller
        return True

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    async def sleep(self):
        if self.config.rollout.free_cache_engine:
            await self.rollout.sleep()
        # return something to block the caller
        return True
