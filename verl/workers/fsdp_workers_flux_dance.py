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
"""Flux + DanceGRPO FSDP worker.

Keep class names aligned with the migrated Dance stack while isolating Flux-only
reward/model logic in a dedicated module.
"""

from __future__ import annotations

import math
import os
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch.profiler import record_function

from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.device import get_device_id, get_device_name
from verl.workers.fsdp_workers import ActorRolloutRefWorker


class FSDPWorkerDance(ActorRolloutRefWorker):
    def __init__(self, config: DictConfig, role: str, disaggregate: bool | None = None, **kwargs):
        super().__init__(config=config, role=role, disaggregate=disaggregate, **kwargs)
        self.colocated = self.role in {"actor_rollout", "actor_rollout_ref"}
        self.device = torch.device(get_device_name(), get_device_id())
        self.model_name = str(self._select("trainer.model_name", default="flux"))

        self.transformer = None
        self.vae = None
        self.optimizer = None
        self.lr_scheduler = None

        self.preprocess_val = None
        self.reward_model = None
        self.reward_processor = None

    def _cfg_get(self, cfg: Any, key: str, default: Any = None) -> Any:
        if cfg is None:
            return default
        if hasattr(cfg, "get"):
            val = cfg.get(key, default)
        else:
            val = getattr(cfg, key, default)
        return default if val is None else val

    def _select(self, path: str, default: Any = None) -> Any:
        val = OmegaConf.select(self.config, path)
        if val is not None:
            return val
        if path.startswith("actor_rollout_ref."):
            val = OmegaConf.select(self.config, path.removeprefix("actor_rollout_ref."))
            if val is not None:
                return val
        return default

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        if self.model_name != "flux":
            raise ValueError(f"fsdp_workers_flux_dance only supports trainer.model_name=flux, got {self.model_name}")
        self._build_model_optimizer_dance()

    def _build_model_optimizer_dance(self) -> None:
        from accelerate.utils import set_seed
        from diffusers.optimization import get_scheduler
        from fastvideo.utils.fsdp_util import apply_fsdp_checkpointing, get_dit_fsdp_kwargs
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        from diffusers import AutoencoderKL, FluxTransformer2DModel

        seed = self._select("algorithm.seed")
        if seed is not None:
            set_seed(seed)

        actor_extra = self._select("actor_rollout_ref.actor.extra", {}) or {}
        dance_cfg = actor_extra.get("dance", {}) if hasattr(actor_extra, "get") else {}

        pretrained_model_name_or_path = self._cfg_get(
            dance_cfg,
            "pretrained_model_name_or_path",
            self._select("actor_rollout_ref.model.path"),
        )
        vae_model_path = self._cfg_get(dance_cfg, "vae_model_path", pretrained_model_name_or_path)
        master_weight_type = str(self._cfg_get(dance_cfg, "master_weight_type", "fp32")).lower()
        sharding_strategy = self._cfg_get(dance_cfg, "fsdp_sharding_strategy", "full")
        gradient_checkpointing = bool(actor_extra.get("gradient_checkpointing", False))

        if self.role in {"actor_rollout", "actor_rollout_ref", "rollout_ref"}:
            use_hpsv2 = bool(self._cfg_get(dance_cfg, "use_hpsv2", True))
            use_pickscore = bool(self._cfg_get(dance_cfg, "use_pickscore", False))
            if use_hpsv2:
                try:
                    # DIscoRL worker source uses this package layout.
                    from HPSv2.hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
                except ModuleNotFoundError:
                    # Existing long-rl/fastvideo Flux scripts use the lowercase package layout.
                    from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer

                open_clip_ckpt = os.environ.get(
                    "OPEN_CLIP_CKPT_PATH", "/home/qzy/models/open_clip_pytorch_model.bin"
                )
                hps_ckpt = os.environ.get("HPSV2_CKPT_PATH", "/home/qzy/models/HPS_v2.1_compressed.pt")
                model, _, preprocess_val = create_model_and_transforms(
                    "ViT-H-14",
                    open_clip_ckpt,
                    precision="amp",
                    device=self.device,
                    jit=False,
                    force_quick_gelu=False,
                    force_custom_text=False,
                    force_patch_dropout=False,
                    force_image_size=None,
                    pretrained_image=False,
                    image_mean=None,
                    image_std=None,
                    light_augmentation=True,
                    aug_cfg={},
                    output_dict=True,
                    with_score_predictor=False,
                    with_region_predictor=False,
                )
                checkpoint = torch.load(hps_ckpt, map_location=str(self.device))
                model.load_state_dict(checkpoint["state_dict"])
                self.reward_model = model.to(self.device).eval()
                self.preprocess_val = preprocess_val
                self.reward_processor = get_tokenizer("ViT-H-14")
            elif use_pickscore:
                from transformers import AutoModel, AutoProcessor

                processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
                model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"
                self.reward_processor = AutoProcessor.from_pretrained(processor_name_or_path)
                self.reward_model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(self.device)

        transformer = FluxTransformer2DModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="transformer",
            torch_dtype=torch.float32,
        )
        fsdp_kwargs, no_split_modules = get_dit_fsdp_kwargs(
            transformer=transformer,
            sharding_strategy=sharding_strategy,
            use_lora=False,
            cpu_offload=False,
            master_weight_type=master_weight_type,
        )
        self.transformer = FSDP(transformer, forward_prefetch=True, **fsdp_kwargs)
        if gradient_checkpointing:
            selective_checkpointing = float(actor_extra.get("selective_checkpointing", 1.0))
            apply_fsdp_checkpointing(transformer, no_split_modules, selective_checkpointing)
        self.transformer.train()

        params_to_optimize = [p for p in self.transformer.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=float(self._select("actor_rollout_ref.actor.optim.lr", 1e-6) or 1e-6),
            betas=(0.9, 0.999),
            weight_decay=float(self._select("actor_rollout_ref.actor.optim.weight_decay", 1e-2) or 1e-2),
            eps=1e-8,
        )
        self.lr_scheduler = get_scheduler(
            name=str(self._select("actor_rollout_ref.actor.optim.warmup_style", "constant") or "constant"),
            optimizer=self.optimizer,
            num_warmup_steps=max(0, int(self._select("actor_rollout_ref.actor.optim.lr_warmup_steps", 0) or 0)),
            num_training_steps=max(1, int(self._select("actor_rollout_ref.actor.optim.total_training_steps", 1_000_000) or 1_000_000)),
            num_cycles=float(self._select("actor_rollout_ref.actor.optim.num_cycles", 0.5) or 0.5),
            power=float(self._select("actor_rollout_ref.actor.optim.power", 1.0) or 1.0),
            last_epoch=-1,
        )

        self.vae = AutoencoderKL.from_pretrained(
            vae_model_path,
            subfolder="vae",
            torch_dtype=torch.bfloat16,
        ).to(self.device)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        prompts = prompts.to(get_device_id())
        return self._generate_sequences_dance(prompts)

    def _generate_sequences_dance(self, prompts: DataProto) -> DataProto:
        from diffusers.image_processor import VaeImageProcessor

        def sd3_time_shift(shift: float, t: torch.Tensor) -> torch.Tensor:
            return (shift * t) / (1 + (shift - 1) * t)

        def prepare_latent_image_ids(batch_size: int, height: int, width: int, device, dtype):
            latent_image_ids = torch.zeros(height, width, 3, device=device, dtype=dtype)
            latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height, device=device)[:, None]
            latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width, device=device)[None, :]
            latent_image_ids = latent_image_ids.reshape(height * width, 3)
            return latent_image_ids

        def pack_latents(latents: torch.Tensor, batch_size: int, num_channels_latents: int, height: int, width: int):
            latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
            latents = latents.permute(0, 2, 4, 1, 3, 5)
            latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
            return latents

        def unpack_latents(latents: torch.Tensor, height: int, width: int, vae_scale_factor: int):
            batch_size, num_patches, channels = latents.shape
            height = 2 * (int(height) // (vae_scale_factor * 2))
            width = 2 * (int(width) // (vae_scale_factor * 2))
            latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
            latents = latents.permute(0, 3, 1, 4, 2, 5)
            latents = latents.reshape(batch_size, channels // 4, height, width)
            return latents

        def flux_step(
            model_output: torch.Tensor,
            latents: torch.Tensor,
            eta: float,
            sigmas: torch.Tensor,
            index: int,
            prev_sample: torch.Tensor | None,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            sigma = sigmas[index]
            dsigma = sigmas[index + 1] - sigma
            prev_sample_mean = latents + dsigma * model_output
            pred_original_sample = latents - sigma * model_output
            delta_t = sigma - sigmas[index + 1]
            std_dev_t = eta * math.sqrt(float(delta_t))
            # SDE solver term (aligned with DIscoRL)
            score_estimate = -(latents - pred_original_sample * (1 - sigma)) / sigma**2
            prev_sample_mean = prev_sample_mean + (-0.5 * eta**2 * score_estimate) * dsigma
            if prev_sample is None:
                prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t
            log_prob = (
                -((prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)) ** 2)
                / (2 * (std_dev_t**2))
            ) - math.log(std_dev_t) - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
            log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
            return prev_sample, pred_original_sample, log_prob

        def normalize_text_ids(ids: torch.Tensor) -> torch.Tensor:
            return ids[0] if ids.ndim == 3 and ids.shape[0] == 1 else ids

        with record_function("worker/generate_sequences_flux_dance"):
            encoder_hidden_states = prompts.batch["encoder_hidden_states"].to(self.device)
            pooled_prompt_embeds = prompts.batch["pooled_prompt_embeds"].to(self.device)
            text_ids = prompts.batch["text_ids"].to(self.device)
            caption = prompts.meta_info.get("caption")

            rollout_cfg = self._select("actor_rollout_ref.rollout", {}) or {}
            actor_extra = self._select("actor_rollout_ref.actor.extra", {}) or {}
            dance_cfg = actor_extra.get("dance", {}) if hasattr(actor_extra, "get") else {}

            num_generations = int(rollout_cfg.get("num_generations", 1))
            use_group = bool(rollout_cfg.get("use_group", False))
            use_same_noise = bool(rollout_cfg.get("use_same_noise", False))

            if use_group:
                def _repeat(x: torch.Tensor) -> torch.Tensor:
                    return torch.repeat_interleave(x, num_generations, dim=0)

                encoder_hidden_states = _repeat(encoder_hidden_states)
                pooled_prompt_embeds = _repeat(pooled_prompt_embeds)
                text_ids = _repeat(text_ids)

                if isinstance(caption, str):
                    caption = [caption] * num_generations
                elif isinstance(caption, (list, tuple)):
                    caption = list(caption)
                    caption = [item for item in caption for _ in range(num_generations)]
                else:
                    raise ValueError(f"Unsupported caption type: {type(caption)}")
            elif isinstance(caption, str):
                caption = [caption]
            elif isinstance(caption, tuple):
                caption = list(caption)

            w = int(rollout_cfg.get("width"))
            h = int(rollout_cfg.get("height"))
            sample_steps = int(rollout_cfg.get("sampling_steps"))
            shift = float(rollout_cfg.get("shift", 3.0))
            eta = float(rollout_cfg.get("eta", 0.3))
            guidance_scale = float(rollout_cfg.get("guidance_scale", dance_cfg.get("guidance_scale", 3.5)))

            sigma_schedule = torch.linspace(1, 0, sample_steps + 1, device=self.device, dtype=torch.float32)
            sigma_schedule = sd3_time_shift(shift, sigma_schedule)

            B = int(encoder_hidden_states.shape[0])
            spatial_downsample = 8
            in_channels = 16
            latent_w, latent_h = w // spatial_downsample, h // spatial_downsample

            batch_size = 1
            batch_indices = torch.chunk(torch.arange(B, device=self.device), max(1, B // batch_size))
            all_latents = []
            all_log_probs = []
            all_rewards = []
            all_image_ids = []

            shared_noise = None
            if use_same_noise:
                shared_noise = torch.randn(
                    (1, in_channels, latent_h, latent_w),
                    device=self.device,
                    dtype=torch.bfloat16,
                )

            for _, batch_idx in enumerate(batch_indices):
                batch_encoder_hidden_states = encoder_hidden_states[batch_idx]
                batch_pooled_prompt_embeds = pooled_prompt_embeds[batch_idx]
                batch_text_ids = text_ids[batch_idx]
                batch_caption = [caption[int(i.item())] for i in batch_idx] if caption is not None else [""]

                if shared_noise is not None:
                    input_latents = shared_noise.repeat(len(batch_idx), 1, 1, 1)
                else:
                    input_latents = torch.randn(
                        (len(batch_idx), in_channels, latent_h, latent_w),
                        device=self.device,
                        dtype=torch.bfloat16,
                    )

                z = pack_latents(input_latents, len(batch_idx), in_channels, latent_h, latent_w)
                image_ids = prepare_latent_image_ids(
                    len(batch_idx), latent_h // 2, latent_w // 2, self.device, torch.bfloat16
                )

                with torch.no_grad():
                    latents_path = [z]
                    log_probs_path = []
                    for i in range(sample_steps):
                        sigma = sigma_schedule[i]
                        timestep_value = int(float(sigma) * 1000)
                        timesteps = torch.full(
                            [batch_encoder_hidden_states.shape[0]],
                            timestep_value,
                            device=self.device,
                            dtype=torch.long,
                        )
                        self.transformer.eval()
                        with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
                            txt_ids = normalize_text_ids(batch_text_ids)
                            pred = self.transformer(
                                hidden_states=z,
                                encoder_hidden_states=batch_encoder_hidden_states,
                                timestep=timesteps / 1000,
                                guidance=torch.tensor([guidance_scale], device=self.device, dtype=torch.bfloat16),
                                txt_ids=txt_ids.repeat(batch_encoder_hidden_states.shape[1], 1),
                                pooled_projections=batch_pooled_prompt_embeds,
                                img_ids=image_ids,
                                joint_attention_kwargs=None,
                                return_dict=False,
                            )[0]

                        z, pred_original, log_prob = flux_step(
                            model_output=pred,
                            latents=z.to(torch.float32),
                            eta=eta,
                            sigmas=sigma_schedule,
                            index=i,
                            prev_sample=None,
                        )
                        z = z.to(torch.bfloat16)
                        latents_path.append(z)
                        log_probs_path.append(log_prob)

                all_image_ids.append(image_ids)
                all_latents.append(torch.stack(latents_path, dim=1))
                all_log_probs.append(torch.stack(log_probs_path, dim=1))

                # Decode + reward (HPSv2 / PickScore)
                rewards = torch.zeros((len(batch_idx),), device=self.device, dtype=torch.float32)
                if self.reward_model is not None and self.reward_processor is not None:
                    image_processor = VaeImageProcessor(16)
                    with torch.inference_mode():
                        with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
                            latents_img = unpack_latents(pred_original, h, w, 8)
                            latents_img = (latents_img / 0.3611) + 0.1159
                            image = self.vae.decode(latents_img, return_dict=False)[0]
                            decoded = image_processor.postprocess(image)

                    if self.preprocess_val is not None:
                        img = self.preprocess_val(decoded[0]).unsqueeze(0).to(device=self.device, non_blocking=True)
                        txt = self.reward_processor([batch_caption[0]]).to(device=self.device, non_blocking=True)
                        with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
                            outputs = self.reward_model(img, txt)
                            image_features = outputs["image_features"]
                            text_features = outputs["text_features"]
                            logits_per_image = image_features @ text_features.T
                            score = torch.diagonal(logits_per_image)
                        rewards = score.to(torch.float32)
                    else:
                        # PickScore path: reward_model is a HF model with processor
                        image_inputs = self.reward_processor(
                            images=[decoded[0]],
                            padding=True,
                            truncation=True,
                            max_length=77,
                            return_tensors="pt",
                        ).to(self.device)
                        text_inputs = self.reward_processor(
                            text=[batch_caption[0]],
                            padding=True,
                            truncation=True,
                            max_length=77,
                            return_tensors="pt",
                        ).to(self.device)
                        with torch.no_grad():
                            image_embs = self.reward_model.get_image_features(**image_inputs)
                            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
                            text_embs = self.reward_model.get_text_features(**text_inputs)
                            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
                            score = (text_embs @ image_embs.T)[0]
                        rewards = score.to(torch.float32)

                all_rewards.append(rewards)

            all_latents = torch.cat(all_latents, dim=0)
            all_log_probs = torch.cat(all_log_probs, dim=0)
            all_image_ids = torch.stack(all_image_ids, dim=0)
            all_rewards = torch.cat(all_rewards, dim=0) if len(all_rewards) > 0 else torch.zeros((all_latents.shape[0],), device=self.device)

            batch_size = all_latents.shape[0]
            timestep_value = [int(float(sigma) * 1000) for sigma in sigma_schedule][:sample_steps]
            timesteps = torch.tensor([timestep_value[:] for _ in range(batch_size)], device=self.device, dtype=torch.long)

            samples = {
                "timesteps": timesteps.detach().clone()[:, :-1],
                "latents": all_latents[:, :-1][:, :-1],
                "next_latents": all_latents[:, 1:][:, :-1],
                "log_probs": all_log_probs[:, :-1],
                "rewards": all_rewards.to(torch.float32),
                "image_ids": all_image_ids,
                "text_ids": text_ids,
                "encoder_hidden_states": encoder_hidden_states,
                "pooled_prompt_embeds": pooled_prompt_embeds,
            }
            return DataProto.from_dict(
                tensors=samples,
                meta_info={
                    "sigma_schedule": sigma_schedule.detach().cpu().numpy(),
                    "prompt_caption": caption[0] if isinstance(caption, list) and len(caption) > 0 else caption,
                },
            ).to("cpu")

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        data = data.to(get_device_id())
        return self._update_actor_dance(data)

    def _get_grad_clip(self) -> float:
        grad_clip = self._select("actor_rollout_ref.actor.grad_clip")
        if grad_clip is None:
            grad_clip = self._select("actor_rollout_ref.actor.max_grad_norm", 1.0) or 1.0
        return float(grad_clip)

    def _update_actor_dance(self, data: DataProto) -> DataProto:
        def _flatten_time_chunk(x: torch.Tensor, start: int, end: int) -> torch.Tensor:
            if x.ndim < 2:
                return x
            chunk = x[:, start:end]
            return chunk.reshape(-1, *chunk.shape[2:])

        def _expand_batch_dim(x: torch.Tensor, repeat: int) -> torch.Tensor:
            if x.ndim == 2:
                return x.repeat_interleave(repeat, dim=0)
            return x.repeat_interleave(repeat, dim=0)

        def flux_step(
            model_output: torch.Tensor,
            latents: torch.Tensor,
            eta: float,
            sigmas: torch.Tensor,
            index: int,
            prev_sample: torch.Tensor,
        ) -> torch.Tensor:
            sigma = sigmas[index]
            dsigma = sigmas[index + 1] - sigma
            prev_sample_mean = latents + dsigma * model_output
            pred_original_sample = latents - sigma * model_output
            delta_t = sigma - sigmas[index + 1]
            std_dev_t = eta * math.sqrt(float(delta_t))
            score_estimate = -(latents - pred_original_sample * (1 - sigma)) / sigma**2
            prev_sample_mean = prev_sample_mean + (-0.5 * eta**2 * score_estimate) * dsigma
            log_prob = (
                -((prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)) ** 2)
                / (2 * (std_dev_t**2))
            ) - math.log(std_dev_t) - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
            log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
            return log_prob

        def grpo_one_step(
            latents: torch.Tensor,
            pre_latents: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            pooled_prompt_embeds: torch.Tensor,
            text_ids: torch.Tensor,
            image_ids: torch.Tensor,
            timesteps: torch.Tensor,
            step_ids: torch.Tensor,
            sigma_schedule: torch.Tensor,
            guidance_scale: float,
        ) -> torch.Tensor:
            text_ids = text_ids[0] if text_ids.ndim == 3 and text_ids.shape[0] == 1 else text_ids
            with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
                self.transformer.train()
                model_pred = self.transformer(
                    hidden_states=latents,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timesteps / 1000,
                    guidance=torch.tensor([guidance_scale], device=latents.device, dtype=torch.bfloat16),
                    txt_ids=text_ids.repeat(encoder_hidden_states.shape[1], 1),
                    pooled_projections=pooled_prompt_embeds,
                    img_ids=image_ids,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]
            log_prob = flux_step(
                model_output=model_pred,
                latents=latents.to(torch.float32),
                eta=float(self._select("actor_rollout_ref.rollout.eta", 0.3) or 0.3),
                sigmas=sigma_schedule,
                index=int(step_ids.item()) if torch.numel(step_ids) == 1 else int(step_ids[0].item()),
                prev_sample=pre_latents.to(torch.float32),
            )
            return log_prob

        with record_function("worker/update_actor_flux_dance"):
            self.optimizer.zero_grad()
            samples = {k: data.batch[k].to(self.device) for k in data.batch.keys()}
            sigma_schedule = torch.as_tensor(data.meta_info["sigma_schedule"], device=self.device, dtype=torch.float32)

            rollout_cfg = self._select("actor_rollout_ref.rollout", {}) or {}
            actor_cfg = self._select("actor_rollout_ref.actor", {}) or {}
            dance_cfg = self._select("actor_rollout_ref.actor.extra.dance", {}) or {}

            num_generations = int(rollout_cfg.get("num_generations", 1))
            bestofn = int(rollout_cfg.get("bestofn", num_generations))
            guidance_scale = float(rollout_cfg.get("guidance_scale", dance_cfg.get("guidance_scale", 3.5)))
            timestep_fraction = float(dance_cfg.get("timestep_fraction", 1.0))
            clip_range = float(dance_cfg.get("clip_range", 1e-4))
            adv_clip_max = float(dance_cfg.get("adv_clip_max", 5.0))
            timestep_micro_batch = int(dance_cfg.get("timestep_micro_batch", 1))
            grad_acc_steps = int(actor_cfg.get("gradient_accumulation_steps", 1))

            batch_size = int(samples["latents"].shape[0])
            n_groups = len(samples["rewards"]) // max(1, num_generations)
            advantages = torch.zeros_like(samples["rewards"])
            for gi in range(n_groups):
                s = gi * num_generations
                e = (gi + 1) * num_generations
                group_rewards = samples["rewards"][s:e]
                advantages[s:e] = (group_rewards - group_rewards.mean()) / (group_rewards.std() + 1e-8)
            samples["advantages"] = advantages

            total_scores = samples["advantages"]
            if bestofn > 0 and bestofn <= batch_size and (not bool(data.meta_info.get("skip_bestofn", False))):
                sorted_indices = torch.argsort(total_scores)
                top_indices = sorted_indices[-bestofn // 2 :]
                bottom_indices = sorted_indices[: bestofn // 2]
                selected_indices = torch.cat([top_indices, bottom_indices])
                selected_indices = selected_indices[torch.randperm(len(selected_indices), device=self.device)]
                if num_generations != bestofn:
                    for key in list(samples.keys()):
                        samples[key] = samples[key][selected_indices]
                    batch_size = len(selected_indices)

            perms = torch.stack(
                [torch.randperm(samples["timesteps"].shape[1], device=self.device) for _ in range(batch_size)]
            )
            for key in ["timesteps", "latents", "next_latents", "log_probs"]:
                samples[key] = samples[key][torch.arange(batch_size, device=self.device)[:, None], perms]

            train_timesteps = max(1, int(samples["timesteps"].shape[1] * timestep_fraction))
            avg_loss = torch.tensor(0.0, device=self.device)

            for i in range(batch_size):
                for t_start in range(0, train_timesteps, timestep_micro_batch):
                    t_end = min(t_start + timestep_micro_batch, train_timesteps)
                    chunk_size = t_end - t_start
                    new_log_probs = grpo_one_step(
                        latents=_flatten_time_chunk(samples["latents"][i : i + 1], t_start, t_end),
                        pre_latents=_flatten_time_chunk(samples["next_latents"][i : i + 1], t_start, t_end),
                        encoder_hidden_states=_expand_batch_dim(samples["encoder_hidden_states"][i : i + 1], chunk_size),
                        pooled_prompt_embeds=_expand_batch_dim(samples["pooled_prompt_embeds"][i : i + 1], chunk_size),
                        text_ids=samples["text_ids"][i : i + 1],
                        image_ids=samples["image_ids"][i].squeeze(0) if samples["image_ids"][i].ndim > 2 else samples["image_ids"][i],
                        timesteps=_flatten_time_chunk(samples["timesteps"][i : i + 1], t_start, t_end),
                        step_ids=perms[i][t_start:t_end],
                        sigma_schedule=sigma_schedule,
                        guidance_scale=guidance_scale,
                    )
                    old_log_probs = _flatten_time_chunk(samples["log_probs"][i : i + 1], t_start, t_end)
                    ratio = torch.exp(new_log_probs - old_log_probs)

                    adv = torch.clamp(samples["advantages"][i : i + 1], -adv_clip_max, adv_clip_max)
                    adv = adv.repeat_interleave(chunk_size, dim=0)
                    unclipped = -adv * ratio
                    clipped = -adv * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
                    loss = torch.maximum(unclipped, clipped).sum() / (max(1, grad_acc_steps) * train_timesteps)
                    loss.backward()
                    avg_loss = loss.detach()

                if ((i + 1) % grad_acc_steps) == 0:
                    self.transformer.clip_grad_norm_(self._get_grad_clip())
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

            output = DataProto(meta_info={"metrics": {"actor/loss": float(avg_loss.item())}})
            return output.to("cpu")
