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

from types import SimpleNamespace

import pytest
import torch

from verl import DataProto

pytest.importorskip("diffusers")
from verl.workers.rollout import diffusion_rollout as diffusion_rollout_mod


class _FakePipeline:
    def __init__(self):
        self.transformer = SimpleNamespace(config=SimpleNamespace(in_channels=2))

    def prepare_latents(self, *args):
        # SD3: batch, channels, h, w, dtype, device, generator, latents
        if len(args) == 8:
            batch, channels, h, w, dtype, device, generator, latents = args
            if latents is not None:
                return latents
            return torch.randn((batch, channels, max(h // 8, 1), max(w // 8, 1)), dtype=dtype, device=device, generator=generator)

        # WAN: batch, channels, h, w, num_frames, dtype, device, generator, latents
        batch, channels, h, w, num_frames, dtype, device, generator, latents = args
        if latents is not None:
            return latents
        temporal = max((num_frames - 1) // 4 + 1, 1)
        return torch.randn(
            (batch, channels, temporal, max(h // 8, 1), max(w // 8, 1)),
            dtype=dtype,
            device=device,
            generator=generator,
        )


def _build_noise_signature(batch_size: int, latents: torch.Tensor | None, generator):
    if latents is not None:
        return latents.reshape(batch_size, -1).mean(dim=1).to(torch.float32)
    if isinstance(generator, list):
        values = [torch.rand((), generator=g).item() for g in generator]
        return torch.tensor(values, dtype=torch.float32)
    if isinstance(generator, torch.Generator):
        return torch.rand((batch_size,), generator=generator, dtype=torch.float32)
    return torch.arange(batch_size, dtype=torch.float32)


def _fake_sd3_pipeline_with_logprob(_pipeline, **kwargs):
    batch_size = kwargs["prompt_embeds"].shape[0]
    num_steps = kwargs["num_inference_steps"]
    signature = _build_noise_signature(batch_size, kwargs.get("latents"), kwargs.get("generator"))
    latents = [(signature + float(i)).view(batch_size, 1, 1, 1) for i in range(num_steps + 1)]
    log_probs = [(signature + i * 0.01).to(torch.float32) for i in range(num_steps)]
    kls = [torch.zeros(batch_size, dtype=torch.float32) for _ in range(num_steps)]
    timesteps = torch.arange(num_steps, 0, -1, dtype=torch.long)
    images = torch.zeros((batch_size, 3, 8, 8), dtype=torch.float32)
    return images, latents, log_probs, kls, timesteps


def _fake_wan_pipeline_with_logprob(_pipeline, **kwargs):
    batch_size = kwargs["prompt_embeds"].shape[0]
    num_steps = kwargs["num_inference_steps"]
    signature = _build_noise_signature(batch_size, kwargs.get("latents"), kwargs.get("generator"))
    latents = [(signature + float(i)).view(batch_size, 1, 1, 1, 1) for i in range(num_steps + 1)]
    log_probs = [(signature + i * 0.01).to(torch.float32) for i in range(num_steps)]
    kls = [torch.zeros(batch_size, dtype=torch.float32) for _ in range(num_steps)]
    timesteps = torch.arange(num_steps, 0, -1, dtype=torch.long)
    videos = torch.zeros((batch_size, 2, 2, 8, 8), dtype=torch.float32)
    return videos, latents, log_probs, kls, timesteps


def _make_sd_prompts(batch_size: int, *, use_seed: bool) -> DataProto:
    tensors = {
        "prompt_embeds": torch.zeros((batch_size, 1, 4, 8), dtype=torch.float32),
        "pooled_prompt_embeds": torch.zeros((batch_size, 1, 8), dtype=torch.float32),
        "negative_prompt_embeds": torch.zeros((batch_size, 1, 4, 8), dtype=torch.float32),
        "negative_pooled_prompt_embeds": torch.zeros((batch_size, 1, 8), dtype=torch.float32),
    }
    if use_seed:
        tensors["seed"] = torch.tensor([11 + i for i in range(batch_size)], dtype=torch.long)
    return DataProto.from_dict(
        tensors=tensors,
        meta_info={"diffusion_algo": "dancegrpo", "use_seed": use_seed},
    )


def _make_wan_prompts(batch_size: int, *, use_seed: bool) -> DataProto:
    tensors = {
        "prompt_embeds": torch.zeros((batch_size, 1, 4, 8), dtype=torch.float32),
        "negative_prompt_embeds": torch.zeros((batch_size, 1, 4, 8), dtype=torch.float32),
    }
    if use_seed:
        tensors["seed"] = torch.tensor([31 + i for i in range(batch_size)], dtype=torch.long)
    return DataProto.from_dict(
        tensors=tensors,
        meta_info={"diffusion_algo": "dancegrpo", "use_seed": use_seed},
    )


def _make_sd_rollout(**kwargs):
    rollout = diffusion_rollout_mod.StableDiffusionRollout.__new__(diffusion_rollout_mod.StableDiffusionRollout)
    rollout.config = SimpleNamespace(
        mode=kwargs.get("mode", "sync"),
        use_group=kwargs.get("use_group", True),
        num_generations=kwargs.get("num_generations", 3),
        use_same_noise=kwargs.get("use_same_noise", True),
        num_steps=kwargs.get("num_steps", 4),
        guidance_scale=5.0,
        resolution=64,
    )
    rollout.pipeline = _FakePipeline()
    return rollout


def _make_wan_rollout(**kwargs):
    rollout = diffusion_rollout_mod.WanRollout.__new__(diffusion_rollout_mod.WanRollout)
    rollout.config = SimpleNamespace(
        mode=kwargs.get("mode", "sync"),
        use_group=kwargs.get("use_group", True),
        num_generations=kwargs.get("num_generations", 3),
        use_same_noise=kwargs.get("use_same_noise", True),
        num_steps=kwargs.get("num_steps", 4),
        guidance_scale=5.0,
        height=64,
        width=64,
        num_frames=9,
    )
    rollout.pipeline = _FakePipeline()
    return rollout


def test_sd_rollout_protocol_fields_and_shapes_sync_group_repeat(monkeypatch):
    monkeypatch.setattr(diffusion_rollout_mod, "sd3_pipeline_with_logprob", _fake_sd3_pipeline_with_logprob)
    rollout = _make_sd_rollout(mode="sync", use_group=True, num_generations=3, use_same_noise=False, num_steps=5)
    prompts = _make_sd_prompts(batch_size=2, use_seed=False)

    output = rollout.generate_sequences(prompts)

    assert len(output) == 6
    for key in ["timesteps", "latents", "next_latents", "log_probs", "old_log_probs", "vq_rewards", "mq_rewards"]:
        assert key in output.batch.keys()
    assert output.batch["timesteps"].shape == (6, 5)
    assert output.batch["latents"].shape[0:2] == (6, 5)
    assert output.batch["next_latents"].shape[0:2] == (6, 5)
    assert output.batch["log_probs"].shape == (6, 5)
    assert torch.equal(output.batch["log_probs"], output.batch["old_log_probs"])
    assert torch.all(output.batch["vq_rewards"] == -1)
    assert torch.all(output.batch["mq_rewards"] == -1)


@pytest.mark.parametrize(
    ("use_seed", "expected_batch"),
    [
        (False, 8),
        (True, 2),
    ],
)
def test_sd_rollout_async_group_repeat_condition(monkeypatch, use_seed: bool, expected_batch: int):
    monkeypatch.setattr(diffusion_rollout_mod, "sd3_pipeline_with_logprob", _fake_sd3_pipeline_with_logprob)
    rollout = _make_sd_rollout(mode="async", use_group=True, num_generations=4, use_same_noise=False, num_steps=4)
    prompts = _make_sd_prompts(batch_size=2, use_seed=use_seed)

    output = rollout.generate_sequences(prompts)
    assert len(output) == expected_batch


def test_sd_rollout_seed_reproducibility(monkeypatch):
    monkeypatch.setattr(diffusion_rollout_mod, "sd3_pipeline_with_logprob", _fake_sd3_pipeline_with_logprob)
    rollout = _make_sd_rollout(mode="async", use_group=True, num_generations=4, use_same_noise=False, num_steps=4)
    prompts = _make_sd_prompts(batch_size=2, use_seed=True)

    out_1 = rollout.generate_sequences(prompts)
    out_2 = rollout.generate_sequences(prompts)

    assert torch.equal(out_1.batch["log_probs"], out_2.batch["log_probs"])
    assert torch.equal(out_1.batch["latents"], out_2.batch["latents"])
    assert torch.equal(out_1.batch["next_latents"], out_2.batch["next_latents"])


def test_sd_rollout_use_same_noise(monkeypatch):
    monkeypatch.setattr(diffusion_rollout_mod, "sd3_pipeline_with_logprob", _fake_sd3_pipeline_with_logprob)
    rollout = _make_sd_rollout(mode="sync", use_group=True, num_generations=3, use_same_noise=True, num_steps=3)
    prompts = _make_sd_prompts(batch_size=2, use_seed=False)

    output = rollout.generate_sequences(prompts)
    first_step = output.batch["latents"][:, 0]
    expected = first_step[0].unsqueeze(0).expand_as(first_step)
    assert torch.equal(first_step, expected)


def test_sd_rollout_use_same_noise_ignores_seed_value(monkeypatch):
    monkeypatch.setattr(diffusion_rollout_mod, "sd3_pipeline_with_logprob", _fake_sd3_pipeline_with_logprob)
    rollout = _make_sd_rollout(mode="async", use_group=True, num_generations=4, use_same_noise=True, num_steps=3)

    prompts_a = _make_sd_prompts(batch_size=2, use_seed=True)
    prompts_b = _make_sd_prompts(batch_size=2, use_seed=True)
    prompts_b.batch["seed"] = torch.tensor([101, 202], dtype=torch.long)

    torch.manual_seed(1234)
    out_a = rollout.generate_sequences(prompts_a)
    torch.manual_seed(1234)
    out_b = rollout.generate_sequences(prompts_b)

    assert torch.equal(out_a.batch["latents"], out_b.batch["latents"])
    assert torch.equal(out_a.batch["next_latents"], out_b.batch["next_latents"])
    assert torch.equal(out_a.batch["log_probs"], out_b.batch["log_probs"])


def test_wan_rollout_protocol_group_repeat(monkeypatch):
    monkeypatch.setattr(diffusion_rollout_mod, "wan_pipeline_with_logprob", _fake_wan_pipeline_with_logprob)
    rollout = _make_wan_rollout(mode="sync", use_group=True, num_generations=2, use_same_noise=False, num_steps=4)
    prompts = _make_wan_prompts(batch_size=3, use_seed=False)

    output = rollout.generate_sequences(prompts)

    assert len(output) == 6
    for key in ["timesteps", "latents", "next_latents", "log_probs", "old_log_probs", "vq_rewards", "mq_rewards"]:
        assert key in output.batch.keys()
    assert output.batch["timesteps"].shape == (6, 4)
    assert output.batch["log_probs"].shape == (6, 4)
