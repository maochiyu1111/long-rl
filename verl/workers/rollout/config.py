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
Rollout config dataclass.
for transferring Long-RL to this repo.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

__all__ = ["RolloutConfig"]


@dataclass
class RolloutConfig:
    name: str = "vllm"
    mode: str = "sync"
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    prompt_length: int = 512
    response_length: int = 512
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.5
    ignore_eos: bool = False
    enforce_eager: bool = True
    free_cache_engine: bool = True
    tensor_model_parallel_size: Optional[int] = 2
    tensor_parallel_size: Optional[int] = None
    max_num_batched_tokens: int = 8192
    max_model_len: Optional[int] = None
    max_num_seqs: int = 1024
    log_prob_micro_batch_size: Optional[int] = None
    log_prob_micro_batch_size_per_gpu: Optional[int] = None
    log_prob_use_dynamic_bsz: bool = False
    log_prob_max_token_len_per_gpu: int = 16384
    disable_log_stats: bool = True
    do_sample: bool = True
    n: int = 1
    resolution: Optional[int] = None
    num_steps: int = 28
    guidance_scale: float = 5.0
    height: Optional[int] = None
    width: Optional[int] = None
    num_frames: Optional[int] = None
    use_group: bool = True
    use_same_noise: bool = True
    num_generations: int = 24
    bestofn: int = 8
    vq_coef: float = 1.0
    mq_coef: float = 0.0
    sampling_steps: int = 20
    shift: int = 5
    eta: float = 0.25
    multi_stage_wake_up: bool = False
    enable_chunked_prefill: bool = False
    engine_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "vllm": {"swap_space": None, "disable_mm_preprocessor_cache": False},
            "sglang": {"attention_backend": None},
        }
    )
    val_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"top_k": -1, "top_p": 1.0, "temperature": 0, "n": 1, "do_sample": False}
    )
    multi_turn: Dict[str, Any] = field(
        default_factory=lambda: {
            "enable": False,
            "max_assistant_turns": None,
            "tool_config_path": None,
            "max_user_turns": None,
            "max_parallel_calls": 1,
            "max_tool_response_length": 256,
            "tool_response_truncate_side": "middle",
            "interaction_config_path": None,
            "use_inference_chat_template": False,
            "tokenization_sanity_check_mode": "strict",
            "format": "hermes",
        }
    )
    calculate_log_probs: bool = False
    agent: Dict[str, Any] = field(
        default_factory=lambda: {
            "num_workers": 8,
            "agent_loop_config_path": None,
            "custom_async_server": {"path": None, "name": None},
        }
    )
    trace: Dict[str, Any] = field(default_factory=lambda: {"backend": None, "token2text": False})
    load_format: str = "dummy_dtensor"
    # When true, gather FSDP shards layer by layer when syncing LoRA weights to vLLM to reduce memory.
    layered_summon: bool = False
    # Maximum bucket size (MB) when updating weights across rollout workers; keeps SGLang TP weight
    # broadcasts bounded and matches the rollout.yaml default.
    update_weights_bucket_megabytes: int = 512

    def __post_init__(self):
        # Keep TP aliases in sync for callers using either key name.
        if self.tensor_parallel_size is None:
            self.tensor_parallel_size = self.tensor_model_parallel_size
        if self.tensor_model_parallel_size is None:
            self.tensor_model_parallel_size = self.tensor_parallel_size

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
