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
"""Diffusion RL dataset that outputs prompt embeddings directly."""

import copy
import os

import datasets
from omegaconf import ListConfig
from torch.utils.data import Dataset
from transformers import ProcessorMixin

from verl.utils.diffusion_processor import StableDiffusionProcessor
from verl.utils.fs import copy_to_local
from verl.utils.wan_processor import WanProcessor

# Same negative prompt string used in Long-RL to discourage low-quality generations.
DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, "
    "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly "
    "drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy "
    "background, three legs, many people in the background, walking backwards"
)


class DiffusionDataset(Dataset):
    """Loads prompts and produces diffusion-ready embeddings (no tokenization)."""

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer,  # kept for interface parity; not used
        processor: ProcessorMixin,
        config,
    ):
        if processor is None:
            raise ValueError("processor is required for diffusion dataset.")

        if not isinstance(data_files, (list, ListConfig)):
            data_files = [data_files]
        self.data_files = copy.deepcopy(list(data_files))
        self.original_data_files = copy.deepcopy(self.data_files)
        self.processor = processor

        self.prompt_key = config.get("prompt_key", "prompt")
        self.negative_prompt = config.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
        self.negative_prompt_key = config.get("negative_prompt_key", None)
        self.return_full_prompt = config.get("return_full_prompt", False)

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/diffusion"))
        self.use_shm = config.get("use_shm", False)
        self.num_videos_per_prompt = config.get("num_videos_per_prompt", 1)

        self._download()
        self._read_files()

    def _download(self, use_origin_parquet: bool = False) -> None:
        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _read_files(self) -> None:
        dataframes = []
        for data_file in self.data_files:
            file_ext = os.path.splitext(data_file)[-1].lower()
            loader = "json" if file_ext in {".json", ".jsonl"} else "parquet"
            dataframe = datasets.load_dataset(loader, data_files=data_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> dict:
        example = self.dataframe[idx]

        if self.prompt_key not in example:
            raise KeyError(f"{self.prompt_key} is required in dataset row for diffusion.")
        prompt = example[self.prompt_key]

        negative_prompt = (
            example[self.negative_prompt_key]
            if self.negative_prompt_key is not None and self.negative_prompt_key in example
            else self.negative_prompt
        )

        if isinstance(self.processor, WanProcessor):
            prompt_embeds = self.processor(prompt, num_videos_per_prompt=self.num_videos_per_prompt)
            negative_prompt_embeds = self.processor(negative_prompt, num_videos_per_prompt=self.num_videos_per_prompt)

            result = {
                "prompt_embeds": prompt_embeds,
                "negative_prompt_embeds": negative_prompt_embeds,
            }
            if self.return_full_prompt:
                result["full_prompts"] = prompt
            return result

        if isinstance(self.processor, StableDiffusionProcessor):
            prompt_embeds, pooled_prompt_embeds = self.processor(prompt, num_images_per_prompt=self.num_videos_per_prompt)
            negative_prompt_embeds, negative_pooled_prompt_embeds = self.processor(
                negative_prompt, num_images_per_prompt=self.num_videos_per_prompt
            )

            result = {
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                "negative_prompt_embeds": negative_prompt_embeds,
                "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
            }
            if self.return_full_prompt:
                result["full_prompts"] = prompt
            return result

        raise ValueError(f"Processor {self.processor} is not supported for diffusion dataset.")
