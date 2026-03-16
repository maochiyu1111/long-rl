import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from trl import get_kbit_device_map, get_quantization_config

from fastvideo.models.videoalign.prompt_template import build_prompt
from fastvideo.models.videoalign.utils import ModelConfig, PEFTLoraConfig, TrainingConfig
from fastvideo.models.videoalign.utils import load_model_from_checkpoint
from fastvideo.models.videoalign.vision_process import process_vision_info
from verl.utils.device import get_device_id, get_device_name


@dataclass
class DataConfig:
    meta_data: str = "/path/to/dataset/meta_data.csv"
    data_dir: str = "/path/to/dataset"
    meta_data_test: Optional[str] = None
    max_frame_pixels: int = 240 * 320
    num_frames: Optional[float] = None
    fps: float = 2.0
    p_shuffle_frames: float = 0.0
    p_color_jitter: float = 0.0
    eval_dim: Union[str, List[str]] = "VQ"
    prompt_template_type: str = "none"
    add_noise: bool = False
    sample_type: str = "uniform"
    use_tied_data: bool = True


class Qwen2VLRewardModelBT(Qwen2VLForConditionalGeneration):
    def __init__(self, config, output_dim=4, reward_token="last", special_token_ids=None):
        super().__init__(config)
        self.output_dim = output_dim
        self.rm_head = nn.Linear(config.hidden_size, output_dim, bias=False)
        self.reward_token = reward_token
        self.special_token_ids = special_token_ids
        if self.special_token_ids is not None:
            self.reward_token = "special"

    def _embed_input_ids(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        embedding_layer = None

        if hasattr(self, "get_input_embeddings"):
            embedding_layer = self.get_input_embeddings()
        elif hasattr(self.model, "get_input_embeddings"):
            embedding_layer = self.model.get_input_embeddings()
        elif hasattr(self.model, "embed_tokens"):
            embedding_layer = self.model.embed_tokens

        if embedding_layer is None:
            raise AttributeError("Qwen2VL reward model could not resolve an input embedding layer")

        return embedding_layer(input_ids)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self._embed_input_ids(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.rm_head(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        if self.reward_token == "last":
            pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        elif self.reward_token == "mean":
            valid_lengths = torch.clamp(sequence_lengths, min=0, max=logits.size(1) - 1)
            pooled_logits = torch.stack([logits[i, : valid_lengths[i]].mean(dim=0) for i in range(batch_size)])
        elif self.reward_token == "special":
            special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for special_token_id in self.special_token_ids:
                special_token_mask = special_token_mask | (input_ids == special_token_id)
            pooled_logits = logits[special_token_mask, ...]
            pooled_logits = pooled_logits.view(batch_size, 3, -1)
            if self.output_dim == 3:
                pooled_logits = pooled_logits.diagonal(dim1=1, dim2=2)
            pooled_logits = pooled_logits.view(batch_size, -1)
        else:
            raise ValueError("Invalid reward_token")

        return {"logits": pooled_logits}


def load_configs_from_json(config_path):
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    data_config = dict(config_dict["data_config"])
    data_config.pop("meta_data", None)
    data_config.pop("data_dir", None)

    return (
        data_config,
        None,
        config_dict["model_config"],
        config_dict["peft_lora_config"],
        config_dict["inference_config"] if "inference_config" in config_dict else None,
    )


def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=None):
    linear_cls = torch.nn.Linear
    embedding_cls = torch.nn.Embedding
    lora_namespan_exclude = lora_namespan_exclude or []
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)

    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    return lora_module_names


def _resolve_base_model_name_or_path(model_config: ModelConfig, load_from_pretrained: str) -> str:
    candidate = model_config.model_name_or_path or "./Qwen2-VL-2B-Instruct"
    search_roots = [os.getcwd(), load_from_pretrained, os.path.dirname(load_from_pretrained)]

    if os.path.isabs(candidate) and os.path.exists(candidate):
        return candidate

    if not os.path.isabs(candidate):
        for root in search_roots:
            resolved = os.path.abspath(os.path.join(root, candidate))
            if os.path.exists(resolved):
                return resolved

    aliases = {
        "./Qwen2-VL-2B-Instruct": "Qwen/Qwen2-VL-2B-Instruct",
        "Qwen2-VL-2B-Instruct": "Qwen/Qwen2-VL-2B-Instruct",
    }
    return aliases.get(candidate, aliases.get(os.path.basename(candidate.rstrip("/")), candidate))


def _env_flag_enabled(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _should_force_local_files_only(model_name_or_path: str) -> bool:
    return os.path.isdir(model_name_or_path) or _env_flag_enabled("HF_HUB_OFFLINE") or _env_flag_enabled("TRANSFORMERS_OFFLINE")


def _get_default_device() -> torch.device:
    device_name = get_device_name()
    if device_name == "cpu":
        return torch.device("cpu")
    return torch.device(device_name, get_device_id())


def _get_device_type(device: Union[str, torch.device]) -> str:
    if isinstance(device, torch.device):
        return device.type
    if isinstance(device, str):
        return device.split(":", 1)[0]
    raise TypeError(f"Unsupported device type: {type(device)}")


def create_model_and_processor(model_config, peft_lora_config, training_args, cache_dir=None, device_type=None):
    model_name_or_path = _resolve_base_model_name_or_path(model_config, training_args.load_from_pretrained)
    local_files_only = _should_force_local_files_only(model_name_or_path)
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        use_cache=True if training_args.gradient_checkpointing else False,
    )

    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        padding_side="right",
        cache_dir=cache_dir,
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        local_files_only=local_files_only,
    )

    special_token_ids = None
    if model_config.use_special_tokens:
        special_tokens = ["<|VQ_reward|>", "<|MQ_reward|>", "<|TA_reward|>"]
        processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        special_token_ids = processor.tokenizer.convert_tokens_to_ids(special_tokens)

    attn_implementation = model_config.attn_implementation
    device_type = device_type or get_device_name()
    if attn_implementation is None:
        use_flash_attn2 = device_type in {"cuda", "npu"} and not training_args.disable_flash_attn2
        attn_implementation = "flash_attention_2" if use_flash_attn2 else "sdpa"

    model = Qwen2VLRewardModelBT.from_pretrained(
        model_name_or_path,
        output_dim=model_config.output_dim,
        reward_token=model_config.reward_token,
        special_token_ids=special_token_ids,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        **model_kwargs,
    )
    if model_config.use_special_tokens:
        model.resize_token_embeddings(len(processor.tokenizer))

    if training_args.bf16:
        model.to(torch.bfloat16)
    if training_args.fp16:
        model.to(torch.float16)

    if peft_lora_config.lora_enable:
        target_modules = find_target_linear_names(
            model,
            num_lora_modules=peft_lora_config.num_lora_modules,
            lora_namespan_exclude=peft_lora_config.lora_namespan_exclude,
        )
        peft_config = LoraConfig(
            target_modules=target_modules,
            r=peft_lora_config.lora_r,
            lora_alpha=peft_lora_config.lora_alpha,
            lora_dropout=peft_lora_config.lora_dropout,
            task_type=peft_lora_config.lora_task_type,
            use_rslora=peft_lora_config.use_rslora,
            bias="none",
            modules_to_save=peft_lora_config.lora_modules_to_save,
        )
        model = get_peft_model(model, peft_config)
    else:
        peft_config = None

    model.config.tokenizer_padding_side = processor.tokenizer.padding_side
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    return model, processor, peft_config


class VideoVLMRewardInference():
    def __init__(
        self,
        load_from_pretrained,
        load_from_pretrained_step=-1,
        device=None,
        dtype=torch.bfloat16,
        base_model_name_or_path: Optional[str] = None,
    ):
        if device is None:
            device = _get_default_device()
        elif not isinstance(device, torch.device):
            device = torch.device(device)
        device_type = _get_device_type(device)
        config_path = os.path.join(load_from_pretrained, "model_config.json")
        data_config, _, model_config, peft_lora_config, inference_config = load_configs_from_json(config_path)
        data_config = DataConfig(**data_config)
        model_config = ModelConfig(**model_config)
        if base_model_name_or_path is not None:
            model_config.model_name_or_path = base_model_name_or_path
        peft_lora_config = PEFTLoraConfig(**peft_lora_config)

        training_args = TrainingConfig(
            load_from_pretrained=load_from_pretrained,
            load_from_pretrained_step=load_from_pretrained_step,
            gradient_checkpointing=False,
            disable_flash_attn2=device_type == "cpu",
            bf16=True if dtype == torch.bfloat16 else False,
            fp16=True if dtype == torch.float16 else False,
            output_dir="",
        )

        model, processor, _ = create_model_and_processor(
            model_config=model_config,
            peft_lora_config=peft_lora_config,
            training_args=training_args,
            device_type=device_type,
        )

        self.device = device

        model, checkpoint_step = load_model_from_checkpoint(model, load_from_pretrained, load_from_pretrained_step)
        model.eval()

        self.model = model
        self.processor = processor

        self.model.to(self.device)

        self.data_config = data_config

        self.inference_config = inference_config

    def _norm(self, reward):
        if self.inference_config is None:
            return reward
        else:
            reward['VQ'] = (reward['VQ'] - self.inference_config['VQ_mean']) / self.inference_config['VQ_std']
            reward['MQ'] = (reward['MQ'] - self.inference_config['MQ_mean']) / self.inference_config['MQ_std']
            reward['TA'] = (reward['TA'] - self.inference_config['TA_mean']) / self.inference_config['TA_std']
            return reward

    def _pad_sequence(self, sequences, attention_mask, max_len, padding_side='right'):
        """
        Pad the sequences to the maximum length.
        """
        assert padding_side in ['right', 'left']
        if sequences.shape[1] >= max_len:
            return sequences, attention_mask
        
        pad_len = max_len - sequences.shape[1]
        padding = (0, pad_len) if padding_side == 'right' else (pad_len, 0)

        sequences_padded = torch.nn.functional.pad(sequences, padding, 'constant', self.processor.tokenizer.pad_token_id)
        attention_mask_padded = torch.nn.functional.pad(attention_mask, padding, 'constant', 0)

        return sequences_padded, attention_mask_padded
    
    def _prepare_input(self, data):
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.device}
            ## TODO: Maybe need to add dtype
            # if self.is_deepspeed_enabled and (torch.is_floating_point(data) or torch.is_complex(data)):
            #     # NLP models inputs are int/uint and those get adjusted to the right dtype of the
            #     # embedding. Other models such as wav2vec2's inputs are already float and thus
            #     # may need special handling to match the dtypes of the model
            #     kwargs.update({"dtype": self.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()})
            return data.to(**kwargs)
        return data
    
    def _prepare_inputs(self, inputs):
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError
        return inputs
    
    def prepare_batch(self, video_paths, prompts, fps=None, num_frames=None, max_pixels=None,):
        fps = self.data_config.fps if fps is None else fps
        num_frames = self.data_config.num_frames if num_frames is None else num_frames
        max_pixels = self.data_config.max_frame_pixels if max_pixels is None else max_pixels

        if num_frames is None:
            chat_data = [
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video", 
                                "video": f"file://{video_path}", 
                                "max_pixels": max_pixels, 
                                "fps": fps,
                                "sample_type": self.data_config.sample_type,
                            },
                            {"type": "text", "text": build_prompt(prompt, self.data_config.eval_dim, self.data_config.prompt_template_type)},
                        ],
                    },
                ] for video_path, prompt in zip(video_paths, prompts)
            ]
        else:
            chat_data = [
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": f"file://{video_path}", 
                                "max_pixels": max_pixels, 
                                "nframes": num_frames,
                                "sample_type": self.data_config.sample_type,
                            },
                            {"type": "text", "text": build_prompt(prompt, self.data_config.eval_dim, self.data_config.prompt_template_type)},
                        ],
                    },
                ] for video_path, prompt in zip(video_paths, prompts)
            ]
        image_inputs, video_inputs = process_vision_info(chat_data)

        batch = self.processor(
            text=self.processor.apply_chat_template(chat_data, tokenize=False, add_generation_prompt=True),
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            videos_kwargs={"do_rescale": True},
        )
        batch = self._prepare_inputs(batch)
        return batch


    def prepare_batch_from_videos(self, videos, prompts, fps=None, num_frames=None, max_pixels=None):
        fps = self.data_config.fps if fps is None else fps
        num_frames = self.data_config.num_frames if num_frames is None else num_frames
        max_pixels = self.data_config.max_frame_pixels if max_pixels is None else max_pixels

        # videos: torch.Tensor (B,T,C,H,W) or list[tensor] 等
        # 我们构造 chat_data 时直接把 tensor 塞进 "video"
        if isinstance(videos, torch.Tensor):
            assert videos.ndim == 5, f"Expect videos as 5D tensor [B,T,C,H,W]/[B,T,H,W,C]. Got {tuple(videos.shape)}"
            video_list = [videos[i] for i in range(videos.shape[0])]
        else:
            video_list = videos  # list，每个元素是一个 video tensor

        chat_data = []
        for v, prompt in zip(video_list, prompts):
            content_video = {
                "type": "video",
                "video": v,                   # 关键：直接放 tensor
                "max_pixels": max_pixels,
                "sample_type": self.data_config.sample_type,
            }
            if num_frames is None:
                content_video["fps"] = fps
            else:
                content_video["nframes"] = num_frames

            chat_data.append([
                {
                    "role": "user",
                    "content": [
                        content_video,
                        {"type": "text", "text": build_prompt(prompt, self.data_config.eval_dim, self.data_config.prompt_template_type)},
                    ],
                }
            ])

        image_inputs, video_inputs = process_vision_info(chat_data)

        batch = self.processor(
            text=self.processor.apply_chat_template(chat_data, tokenize=False, add_generation_prompt=True),
            images=image_inputs,
            videos=video_inputs,                 # 这里将是 fetch_video 返回的 TCHW tensor
            padding=True,
            return_tensors="pt",
            videos_kwargs={"do_rescale": True},
        )
        batch = self._prepare_inputs(batch)
        return batch


    def reward(self, video_paths, prompts, fps=None, num_frames=None, max_pixels=None, use_norm=True):
        """
        Inputs:
            video_paths: List[str], B paths of the videos.
            prompts: List[str], B prompts for the videos.
            eval_dims: List[str], N evaluation dimensions.
            fps: float, sample rate of the videos. If None, use the default value in the config.
            num_frames: int, number of frames of the videos. If None, use the default value in the config.
            max_pixels: int, maximum pixels of the videos. If None, use the default value in the config.
            use_norm: bool, whether to rescale the output rewards
        Outputs:
            Rewards: List[dict], N + 1 rewards of the B videos.
        """
        assert fps is None or num_frames is None, "fps and num_frames cannot be set at the same time."
        
        batch = self.prepare_batch(video_paths, prompts, fps, num_frames, max_pixels)
        rewards = self.model(
            return_dict=True,
            **batch
        )["logits"]

        rewards = [{'VQ': reward[0].item(), 'MQ': reward[1].item(), 'TA': reward[2].item()} for reward in rewards]
        for i in range(len(rewards)):
            if use_norm:
                rewards[i] = self._norm(rewards[i])
            rewards[i]['Overall'] = rewards[i]['VQ'] + rewards[i]['MQ'] + rewards[i]['TA']

        return rewards
    def reward_from_videos(self, videos, prompts, fps=None, num_frames=None, max_pixels=None, use_norm=True):
        batch = self.prepare_batch_from_videos(videos, prompts, fps, num_frames, max_pixels)
        rewards = self.model(return_dict=True, **batch)["logits"]
        rewards = [{'VQ': r[0].item(), 'MQ': r[1].item(), 'TA': r[2].item()} for r in rewards]
        for i in range(len(rewards)):
            if use_norm:
                rewards[i] = self._norm(rewards[i])
            rewards[i]['Overall'] = rewards[i]['VQ'] + rewards[i]['MQ'] + rewards[i]['TA']
        return rewards



if __name__ == "__main__":
    load_from_pretrained = "./checkpoints"
    device = _get_default_device()
    dtype = torch.bfloat16

    inferencer = VideoVLMRewardInference(load_from_pretrained, device=device, dtype=dtype)

    video_paths = [
        "datasets/train/videos/example_1_A.mp4",
        "datasets/train/videos/example_1_B.mp4",
        "datasets/train/videos/example_2_A.mp4",
    ]

    prompts = [
        "The camera remains still, a girl with braided hair and wearing a pink dress approached the chair in the room and sat on it, the background is a cozy bedroom, warm indoor lighting.",
        "The camera remains still, a girl with braided hair and wearing a pink dress approached the chair in the room and sat on it, the background is a cozy bedroom, warm indoor lighting.",
        "The camera follows a young explorer through an abandoned urban building at night, exploring hidden corridors and forgotten spaces, with a mix of light and shadow creating a mysterious atmosphere.",
    ]

    with torch.no_grad():
        rewards = inferencer.reward(video_paths, prompts, use_norm=True)
        print(rewards)
