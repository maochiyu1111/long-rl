# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
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
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any, Dict, Optional, Union

import torch
import torch.distributed
from tensordict import TensorDict

from ...protocol import DataProto
from .base import BaseRollout
from .config import RolloutConfig
from diffusers import StableDiffusion3Pipeline, WanPipeline
from ..diffusion_helper import sd3_pipeline_with_logprob, wan_pipeline_with_logprob


def _repeat_if_group(tensor: torch.Tensor, repeat_times: int, enabled: bool) -> torch.Tensor:
    if not enabled:
        return tensor
    return torch.repeat_interleave(tensor, repeat_times, dim=0)


def _is_dance_mode(prompts: DataProto) -> bool:
    return prompts.meta_info.get("diffusion_algo") == "dancegrpo"


def _is_async_rollout_request(config: RolloutConfig, prompts: DataProto) -> bool:
    meta_info = prompts.meta_info or {}
    if bool(meta_info.get("async_rollout", False)):
        return True
    return str(getattr(config, "mode", "sync")).lower() == "async"


def _should_repeat_for_group(config: RolloutConfig, prompts: DataProto) -> bool:
    if not _is_dance_mode(prompts):
        return False
    if not config.use_group:
        return False

    # Keep disco_rl parity:
    # - sync: repeat when use_group is enabled
    # - async: repeat only when use_group and gen_seed=True
    use_seed = bool((prompts.meta_info or {}).get("use_seed", False))
    gen_seed = not use_seed
    if _is_async_rollout_request(config, prompts):
        return gen_seed
    return True


def _is_sync_mode(config: RolloutConfig) -> bool:
    return str(getattr(config, "mode", "sync")).lower() == "sync"


def _build_seed_generators(device: torch.device, seeds: torch.Tensor) -> list[torch.Generator]:
    generators: list[torch.Generator] = []
    for seed in seeds.detach().to(torch.long).reshape(-1):
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed.item()))
        generators.append(gen)
    return generators


def _make_placeholder_rewards(batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    vq_rewards = torch.full((batch_size,), -1.0, dtype=torch.float32, device=device)
    mq_rewards = torch.full((batch_size,), -1.0, dtype=torch.float32, device=device)
    return vq_rewards, mq_rewards


def _align_trajectory_tensors(
    latents: torch.Tensor,
    log_probs: torch.Tensor,
    kls: torch.Tensor,
    timesteps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    train_steps = min(timesteps.size(1), max(latents.size(1) - 1, 0), log_probs.size(1), kls.size(1))
    if train_steps <= 0:
        raise ValueError(
            "Invalid rollout trajectory with non-positive train steps: "
            f"timesteps={timesteps.size(1)}, latents={latents.size(1)}, log_probs={log_probs.size(1)}, kls={kls.size(1)}"
        )
    timesteps = timesteps[:, :train_steps]
    latents = latents[:, : train_steps + 1]
    log_probs = log_probs[:, :train_steps]
    kls = kls[:, :train_steps]
    return latents, log_probs, kls, timesteps, train_steps


class StableDiffusionRollout(BaseRollout):
    def __init__(
        self,
        model_path: str,
        config: RolloutConfig,
    ):
        """A diffusion rollout based on SD3.5

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
        """
        super().__init__()
        self.rank = int(os.getenv("RANK", "0"))
        self.config = config
        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")

        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")
        self.pipeline = StableDiffusion3Pipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="balanced",
        )
        # freeze parameters of models to save more memory
        self.pipeline.vae.requires_grad_(False)
        self.pipeline.text_encoder.requires_grad_(False)
        self.pipeline.text_encoder_2.requires_grad_(False)
        self.pipeline.text_encoder_3.requires_grad_(False)

        # disable safety checker
        self.pipeline.safety_checker = None
        # make the progress bar nicer
        self.pipeline.set_progress_bar_config(
            position=1,
            disable=not torch.distributed.get_rank() == 0,
            leave=False,
            desc="Timestep",
            dynamic_ncols=True,
        )


    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        should_repeat = _should_repeat_for_group(self.config, prompts)
        prompt_embeds = _repeat_if_group(prompts.batch["prompt_embeds"], self.config.num_generations, should_repeat).squeeze(1)
        pooled_prompt_embeds = _repeat_if_group(
            prompts.batch["pooled_prompt_embeds"], self.config.num_generations, should_repeat
        ).squeeze(1)
        negative_prompt_embeds = _repeat_if_group(
            prompts.batch["negative_prompt_embeds"], self.config.num_generations, should_repeat
        ).squeeze(1)
        negative_pooled_prompt_embeds = _repeat_if_group(
            prompts.batch["negative_pooled_prompt_embeds"], self.config.num_generations, should_repeat
        ).squeeze(1)
        batch_size = prompt_embeds.size(0)

        use_seed = bool(prompts.meta_info.get("use_seed", False))
        seed_tensor = prompts.batch["seed"] if "seed" in prompts.batch.keys() else None
        if should_repeat and use_seed and seed_tensor is not None:
            seed_tensor = _repeat_if_group(seed_tensor, self.config.num_generations, True)

        sampling_kwargs: dict[str, Any] = {}
        if _is_dance_mode(prompts):
            if self.config.use_same_noise:
                num_channels_latents = self.pipeline.transformer.config.in_channels
                base_latents = self.pipeline.prepare_latents(
                    1,
                    num_channels_latents,
                    self.config.resolution,
                    self.config.resolution,
                    prompt_embeds.dtype,
                    prompt_embeds.device,
                    None,
                    None,
                )
                repeat_shape = (batch_size,) + (1,) * (base_latents.ndim - 1)
                sampling_kwargs["latents"] = base_latents.repeat(*repeat_shape)
            elif use_seed and seed_tensor is not None:
                if seed_tensor.numel() != batch_size:
                    raise ValueError(f"seed size {seed_tensor.numel()} does not match batch size {batch_size}")
                sampling_kwargs["generator"] = _build_seed_generators(prompt_embeds.device, seed_tensor)
            else:
                if _is_sync_mode(self.config):
                    auto_seeds = torch.full((batch_size,), 42, dtype=torch.long, device=prompt_embeds.device)
                else:
                    auto_seeds = torch.arange(42, 42 + batch_size, dtype=torch.long, device=prompt_embeds.device)
                sampling_kwargs["generator"] = _build_seed_generators(prompt_embeds.device, auto_seeds)

        with torch.no_grad():
            images, latents, log_probs, kls, timesteps = sd3_pipeline_with_logprob(
                self.pipeline,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                num_inference_steps=self.config.num_steps,
                guidance_scale=self.config.guidance_scale,
                output_type="pt",
                return_dict=False,
                height=self.config.resolution,
                width=self.config.resolution,
                **sampling_kwargs,
            )

        latents = torch.stack(latents, dim=1)  # (batch_size, num_steps + 1, ...)
        log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps)
        kls = torch.stack(kls, dim=1)
        kl = kls.detach()

        timesteps = timesteps.to(prompt_embeds.device).repeat(batch_size, 1)  # (batch_size, num_steps)
        latents, log_probs, kl, timesteps, train_steps = _align_trajectory_tensors(latents, log_probs, kl, timesteps)
        vq_rewards, mq_rewards = _make_placeholder_rewards(batch_size=batch_size, device=prompt_embeds.device)

        batch = TensorDict(
            {
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                "negative_prompt_embeds": negative_prompt_embeds,
                "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
                "timesteps": timesteps,
                "images": images,
                "latents": latents[:, :train_steps],  # latent before timestep t
                "next_latents": latents[:, 1 : train_steps + 1],  # latent after timestep t
                "log_probs": log_probs,
                "old_log_probs": log_probs,
                "kl": kl,
                "vq_rewards": vq_rewards,
                "mq_rewards": mq_rewards,
            },
            batch_size=batch_size,
        )
        return DataProto(batch=batch, non_tensor_batch={}, meta_info={})


class WanRollout(BaseRollout):
    def __init__(
        self,
        model_path: str,
        config: RolloutConfig,
    ):
        """A diffusion rollout based on Wan2.1-T2V-1.3B

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
        """
        super().__init__()
        self.rank = int(os.getenv("RANK", "0"))
        self.config = config
        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")

        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")
        self.pipeline = WanPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="balanced",
        )
        # freeze parameters of models to save more memory
        self.pipeline.vae.requires_grad_(False)
        self.pipeline.text_encoder.requires_grad_(False)

        # disable safety checker
        self.pipeline.safety_checker = None
        # make the progress bar nicer
        self.pipeline.set_progress_bar_config(
            position=1,
            disable=not torch.distributed.get_rank() == 0,
            leave=False,
            desc="Timestep",
            dynamic_ncols=True,
        )


    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        should_repeat = _should_repeat_for_group(self.config, prompts)
        prompt_embeds = _repeat_if_group(prompts.batch["prompt_embeds"], self.config.num_generations, should_repeat).squeeze(1)
        negative_prompt_embeds = _repeat_if_group(
            prompts.batch["negative_prompt_embeds"], self.config.num_generations, should_repeat
        ).squeeze(1)
        batch_size = prompt_embeds.size(0)

        use_seed = bool(prompts.meta_info.get("use_seed", False))
        seed_tensor = prompts.batch["seed"] if "seed" in prompts.batch.keys() else None
        if should_repeat and use_seed and seed_tensor is not None:
            seed_tensor = _repeat_if_group(seed_tensor, self.config.num_generations, True)

        sampling_kwargs: dict[str, Any] = {}
        if _is_dance_mode(prompts):
            if self.config.use_same_noise:
                num_channels_latents = self.pipeline.transformer.config.in_channels
                base_latents = self.pipeline.prepare_latents(
                    1,
                    num_channels_latents,
                    self.config.height,
                    self.config.width,
                    self.config.num_frames,
                    torch.float32,
                    prompt_embeds.device,
                    None,
                    None,
                )
                repeat_shape = (batch_size,) + (1,) * (base_latents.ndim - 1)
                sampling_kwargs["latents"] = base_latents.repeat(*repeat_shape)
            elif use_seed and seed_tensor is not None:
                if seed_tensor.numel() != batch_size:
                    raise ValueError(f"seed size {seed_tensor.numel()} does not match batch size {batch_size}")
                sampling_kwargs["generator"] = _build_seed_generators(prompt_embeds.device, seed_tensor)
            else:
                if _is_sync_mode(self.config):
                    auto_seeds = torch.full((batch_size,), 42, dtype=torch.long, device=prompt_embeds.device)
                else:
                    auto_seeds = torch.arange(42, 42 + batch_size, dtype=torch.long, device=prompt_embeds.device)
                sampling_kwargs["generator"] = _build_seed_generators(prompt_embeds.device, auto_seeds)

        with torch.no_grad():
            videos, latents, log_probs, kls, timesteps = wan_pipeline_with_logprob(
                self.pipeline,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                num_inference_steps=self.config.num_steps,
                guidance_scale=self.config.guidance_scale,
                output_type="pt",
                return_dict=False,
                height=self.config.height,
                width=self.config.width,
                num_frames=self.config.num_frames,
                **sampling_kwargs,
            )

        latents = torch.stack(latents, dim=1)  # (batch_size, num_steps + 1, ...)
        log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps)
        kls = torch.stack(kls, dim=1)
        kl = kls.detach()

        timesteps = timesteps.to(prompt_embeds.device).repeat(batch_size, 1)  # (batch_size, num_steps)
        latents, log_probs, kl, timesteps, train_steps = _align_trajectory_tensors(latents, log_probs, kl, timesteps)
        vq_rewards, mq_rewards = _make_placeholder_rewards(batch_size=batch_size, device=prompt_embeds.device)

        batch = TensorDict(
            {
                "prompt_embeds": prompt_embeds,
                "negative_prompt_embeds": negative_prompt_embeds,
                "timesteps": timesteps,
                "videos": videos,
                "latents": latents[:, :train_steps],  # latent before timestep t
                "next_latents": latents[:, 1 : train_steps + 1],  # latent after timestep t
                "log_probs": log_probs,
                "old_log_probs": log_probs,
                "kl": kl,
                "vq_rewards": vq_rewards,
                "mq_rewards": mq_rewards,
            },
            batch_size=batch_size,
        )
        return DataProto(batch=batch, non_tensor_batch={}, meta_info={})
