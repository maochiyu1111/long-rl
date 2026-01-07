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
"""Utils for tokenization."""

import os
import warnings
from typing import List, Optional, Union

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    PreTrainedTokenizer,
    ProcessorMixin,
    T5EncoderModel,
    T5TokenizerFast,
    UMT5EncoderModel,
)

from verl.utils.diffusion_processor import StableDiffusionProcessor
try:
    from verl.utils.qwen_omni_utils import Qwen2_5OmniProcessor
except ImportError:  # optional dependency; guard to allow non-Omni use-cases
    Qwen2_5OmniProcessor = None
from verl.utils.wan_processor import WanProcessor

__all__ = ["hf_tokenizer", "hf_processor"]


def set_pad_token_id(tokenizer: PreTrainedTokenizer) -> None:
    """Set pad_token_id to eos_token_id if it is None."""
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        warnings.warn(f"tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}", stacklevel=1)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        warnings.warn(f"tokenizer.pad_token is None. Now set to {tokenizer.eos_token}", stacklevel=1)


def _is_diffusers_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "model_index.json"))


def hf_tokenizer(
    name_or_path: str,
    override_chat_template: Optional[str] = None,
    correct_pad_token: bool = True,
    correct_gemma2: bool = True,
    **kwargs,
) -> Union[PreTrainedTokenizer, List[PreTrainedTokenizer]]:
    """Create a huggingface tokenizer and optionally handle diffusion checkpoints."""
    return get_tokenizer(
        name_or_path,
        override_chat_template=override_chat_template,
        correct_pad_token=correct_pad_token,
        correct_gemma2=correct_gemma2,
        **kwargs,
    )


def hf_processor(
    name_or_path: str,
    num_video_frames: int = 8,
    override_chat_template: Optional[str] = None,
    **kwargs,
) -> Optional[ProcessorMixin]:
    """Create a huggingface processor to process multimodal data."""
    return get_processor(
        name_or_path,
        num_video_frames=num_video_frames,
        override_chat_template=override_chat_template,
        **kwargs,
    )


def get_tokenizer(
    model_path: str,
    override_chat_template: Optional[str] = None,
    correct_pad_token: bool = True,
    correct_gemma2: bool = True,
    **kwargs,
) -> Union[PreTrainedTokenizer, List[PreTrainedTokenizer]]:
    if kwargs.get("diffusion", False):
        return get_diffusion_tokenizer(model_path, **kwargs)

    tokenizer_kwargs = {k: v for k, v in kwargs.items() if k != "diffusion"}

    # VILA checkpoints place the LLM tokenizer under llm/
    if "vila" in model_path.lower():
        model_path = os.path.join(model_path, "llm")

    tokenizer = _load_hf_tokenizer(
        model_path,
        correct_pad_token=correct_pad_token,
        correct_gemma2=correct_gemma2,
        **tokenizer_kwargs,
    )

    # Gemma models expose <bos>/<eos> tokens; align EOS for RL stability.
    if (
        correct_gemma2
        and getattr(tokenizer, "bos_token", None) == "<bos>"
        and getattr(tokenizer, "eos_token", None) == "<eos>"
    ):
        warnings.warn("Found gemma tokenizer. Set eos_token to <end_of_turn>.", stacklevel=1)
        tokenizer.eos_token = "<end_of_turn>"

    if override_chat_template is not None and hasattr(tokenizer, "chat_template"):
        tokenizer.chat_template = override_chat_template

    return tokenizer


def _load_hf_tokenizer(
    name_or_path: str, correct_pad_token: bool = True, correct_gemma2: bool = True, **kwargs
) -> PreTrainedTokenizer:
    token_kwargs = dict(kwargs)
    if correct_gemma2 and isinstance(name_or_path, str) and "gemma-2-2b-it" in name_or_path:
        warnings.warn(
            "Found gemma-2-2b-it tokenizer. Set eos_token and eos_token_id to <end_of_turn> and 107.", stacklevel=1
        )
        token_kwargs["eos_token"] = "<end_of_turn>"
        token_kwargs["eos_token_id"] = 107

    def _maybe_set_pad(tok: PreTrainedTokenizer) -> PreTrainedTokenizer:
        if correct_pad_token and hasattr(tok, "pad_token_id"):
            set_pad_token_id(tok)
        return tok

    try:
        return _maybe_set_pad(AutoTokenizer.from_pretrained(name_or_path, **token_kwargs))
    except Exception:
        # Diffusion checkpoints (e.g., WAN/SD3 diffusers dirs) may not be loadable by transformers' AutoTokenizer.
        if isinstance(name_or_path, str) and _is_diffusers_dir(name_or_path):
            try:
                tok = AutoTokenizer.from_pretrained(name_or_path, subfolder="tokenizer", **token_kwargs)
                return _maybe_set_pad(tok)
            except Exception:
                tok = AutoTokenizer.from_pretrained(name_or_path, **token_kwargs)
                return _maybe_set_pad(tok)
        raise


def get_diffusion_tokenizer(model_path: str, **kwargs) -> Union[PreTrainedTokenizer, List[PreTrainedTokenizer]]:
    kwargs_clean = {k: v for k, v in kwargs.items() if k != "diffusion"}
    if "wan" in model_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer", **kwargs_clean)
        return tokenizer
    if "stable-diffusion" in model_path.lower():
        tokenizer_1 = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        tokenizer_2 = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer_2")
        tokenizer_3 = T5TokenizerFast.from_pretrained(model_path, subfolder="tokenizer_3")
        return [tokenizer_1, tokenizer_2, tokenizer_3]
    raise ValueError(f"Unsupported model: {model_path}")


def get_processor(
    model_path: str, num_video_frames: int = 8, override_chat_template: Optional[str] = None, **kwargs
) -> Optional[ProcessorMixin]:
    if kwargs.get("diffusion", False):
        return get_diffusion_processor(model_path, **kwargs)

    processor_kwargs = {k: v for k, v in kwargs.items() if k != "diffusion"}

    if "vila" in model_path.lower():
        processor_kwargs["trust_remote_code"] = True

    if "omni" in model_path.lower():
        if Qwen2_5OmniProcessor is None:
            raise ImportError("Qwen2_5OmniProcessor is not available. Please add verl.utils.qwen_omni_utils.")
        processor = Qwen2_5OmniProcessor.from_pretrained(model_path, **processor_kwargs)
    else:
        processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)

    if hasattr(processor, "config"):
        processor.config.num_video_frames = num_video_frames
        processor.config.fps = 2
    if override_chat_template is not None:
        processor.chat_template = override_chat_template

    processor.num_video_frames = num_video_frames
    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.52.4/src/transformers/models/auto/processing_auto.py#L386
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None

    return processor


def get_diffusion_processor(model_path: str, **kwargs) -> Optional[ProcessorMixin]:
    kwargs_clean = {k: v for k, v in kwargs.items() if k != "diffusion"}

    if "wan" in model_path.lower():
        text_encoder = UMT5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")
        tokenizer = T5TokenizerFast.from_pretrained(model_path, subfolder="tokenizer", **kwargs_clean)
        text_encoder.requires_grad_(False)

        processor = WanProcessor(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            max_sequence_length=kwargs.get("max_sequence_length", 128),
        )
        return processor
    if "stable-diffusion" in model_path.lower():
        text_encoder_1 = CLIPTextModelWithProjection.from_pretrained(model_path, subfolder="text_encoder")
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_path, subfolder="text_encoder_2")
        text_encoder_3 = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder_3")

        # Freeze parameters to save memory.
        text_encoder_1.requires_grad_(False)
        text_encoder_2.requires_grad_(False)
        text_encoder_3.requires_grad_(False)

        tokenizer_1 = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        tokenizer_2 = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer_2")
        tokenizer_3 = T5TokenizerFast.from_pretrained(model_path, subfolder="tokenizer_3")

        text_encoders = [text_encoder_1, text_encoder_2, text_encoder_3]
        tokenizers = [tokenizer_1, tokenizer_2, tokenizer_3]

        processor = StableDiffusionProcessor(
            text_encoders=text_encoders,
            tokenizers=tokenizers,
            max_sequence_length=kwargs.get("max_sequence_length", 128),
        )

        return processor
    raise ValueError(f"Unsupported model: {model_path}")
