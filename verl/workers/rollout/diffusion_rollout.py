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
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.distributed
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin

from ...protocol import DataProto
from ...utils import torch_functional as VF
from ...utils.dataset import process_image
from ...utils.torch_dtypes import PrecisionType
from .base import BaseRollout
from .config import RolloutConfig
import torch.nn.functional as F
from verl.utils.vila_remote_code.constants import IGNORE_INDEX
from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler, WanPipeline
from ..diffusion_helper import sd3_pipeline_with_logprob, wan_pipeline_with_logprob


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
        prompt_embeds = prompts.batch["prompt_embeds"].squeeze(1)
        pooled_prompt_embeds = prompts.batch["pooled_prompt_embeds"].squeeze(1)
        negative_prompt_embeds = prompts.batch["negative_prompt_embeds"].squeeze(1)
        negative_pooled_prompt_embeds = prompts.batch["negative_pooled_prompt_embeds"].squeeze(1)
        batch_size = prompt_embeds.size(0)
        # print("*** rollout prompt_embeds ***", prompt_embeds.shape)
        # sample
        # with autocast():
        with torch.no_grad():
            images, latents, log_probs, kls = sd3_pipeline_with_logprob(
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
        )
            
        latents = torch.stack(
            latents, dim=1
        )  # (batch_size, num_steps + 1, 16, 96, 96)
        log_probs = torch.stack(log_probs, dim=1)  # shape after stack (batch_size, num_steps)
        kls = torch.stack(kls, dim=1) 
        kl = kls.detach()

        timesteps = self.pipeline.scheduler.timesteps.repeat(
            batch_size, 1
        )  # (batch_size, num_steps)
        # print("*** prompt_embeds ***", prompt_embeds.shape)
        # print("*** pooled_prompt_embeds ***", pooled_prompt_embeds.shape)
        # print("*** timesteps ***", timesteps.shape)
        # print("*** images ***", images.shape)
        # print("*** latents ***", latents.shape)
        # print("*** log_probs ***", log_probs.shape)
        # print("*** kl ***", kl.shape)
        batch = TensorDict(
            {
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                "negative_prompt_embeds": negative_prompt_embeds,
                "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
                "timesteps": timesteps.to(prompt_embeds.device),
                "images": images,
                "latents": latents[
                    :, :-1
                ],  # each entry is the latent before timestep t
                "next_latents": latents[
                    :, 1:
                ],  # each entry is the latent after timestep t
                "old_log_probs": log_probs,
                "kl": kl,
                # "rewards": rewards,
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
        prompt_embeds = prompts.batch["prompt_embeds"].squeeze(1)
        negative_prompt_embeds = prompts.batch["negative_prompt_embeds"].squeeze(1)
        batch_size = prompt_embeds.size(0)
        # print("*** rollout prompt_embeds ***", prompt_embeds.shape)
        # sample
        # with autocast():
        with torch.no_grad():
            videos, latents, log_probs, kls = wan_pipeline_with_logprob(
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
        )
            
        latents = torch.stack(
            latents, dim=1
        )  # (batch_size, num_steps + 1, 16, 96, 96)
        log_probs = torch.stack(log_probs, dim=1)  # shape after stack (batch_size, num_steps)
        kls = torch.stack(kls, dim=1) 
        kl = kls.detach()

        timesteps = self.pipeline.scheduler.timesteps.repeat(
            batch_size, 1
        )  # (batch_size, num_steps)
        # print("*** prompt_embeds ***", prompt_embeds.shape)
        # print("*** pooled_prompt_embeds ***", pooled_prompt_embeds.shape)
        # print("*** timesteps ***", timesteps.shape)
        # print("*** images ***", images.shape)
        # print("*** latents ***", latents.shape)
        # print("*** log_probs ***", log_probs.shape)
        # print("*** kl ***", kl.shape)
        batch = TensorDict(
            {
                "prompt_embeds": prompt_embeds,
                "negative_prompt_embeds": negative_prompt_embeds,
                "timesteps": timesteps.to(prompt_embeds.device),
                "videos": videos,
                "latents": latents[
                    :, :-1
                ],  # each entry is the latent before timestep t
                "next_latents": latents[
                    :, 1:
                ],  # each entry is the latent after timestep t
                "old_log_probs": log_probs,
                "kl": kl,
                # "rewards": rewards,
            },
            batch_size=batch_size,
        )
        return DataProto(batch=batch, non_tensor_batch={}, meta_info={})
