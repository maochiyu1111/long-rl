# Copyright 2025 Individual Contributor: Mert Unsal
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

from collections import defaultdict

import torch

from verl import DataProto
from verl.workers.reward_manager import register


@register("batch")
class BatchRewardManager:
    """
    A batch reward manager that computes rewards for a batch of data.

    Args:
        tokenizer (Tokenizer): The tokenizer to use for decoding the responses.
        num_examine (int): The number of responses to examine.
        compute_score (callable): The function to compute the rewards.
        reward_fn_key (str): The key to use for the reward function.
        reward_kwargs (dict): The keyword arguments to pass to the reward function.
    """

    def __init__(self, tokenizer, num_examine, compute_score, reward_fn_key="data_source", **reward_kwargs):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.reward_kwargs = dict(reward_kwargs)
        self.reward_backend = self._resolve_reward_backend(self.reward_kwargs)
        self.compute_score_kwargs = self._extract_compute_score_kwargs(self.reward_kwargs)

        self.videoalign_inferencer = None
        self.videoalign_use_norm = True
        self.videoalign_init_error = None

        if self.reward_backend == "videoalign":
            self._init_videoalign_backend()

    @staticmethod
    def _resolve_reward_backend(reward_kwargs):
        backend = reward_kwargs.get("backend", None)
        use_videoalign = bool(reward_kwargs.get("use_videoalign", False))
        if backend is None:
            return "videoalign" if use_videoalign else "builtin"
        backend = str(backend).lower()
        if backend not in {"builtin", "videoalign"}:
            raise ValueError(f"Unsupported reward backend: {backend}. Expected one of ['builtin', 'videoalign'].")
        return backend

    @staticmethod
    def _extract_compute_score_kwargs(reward_kwargs):
        kwargs = dict(reward_kwargs)
        kwargs.pop("backend", None)
        kwargs.pop("use_videoalign", None)
        kwargs.pop("videoalign", None)
        kwargs.pop("videoalign_model_path", None)
        kwargs.pop("videoalign_load_from_pretrained", None)
        kwargs.pop("videoalign_load_from_pretrained_step", None)
        kwargs.pop("videoalign_device", None)
        kwargs.pop("videoalign_dtype", None)
        kwargs.pop("videoalign_use_norm", None)
        return kwargs

    def _init_videoalign_backend(self):
        videoalign_cfg = self.reward_kwargs.get("videoalign", {}) or {}
        if not isinstance(videoalign_cfg, dict):
            raise ValueError(
                "reward_kwargs.videoalign must be a dict when backend=videoalign, "
                f"got {type(videoalign_cfg)}."
            )

        model_path = (
            videoalign_cfg.get("load_from_pretrained")
            or self.reward_kwargs.get("videoalign_load_from_pretrained")
            or self.reward_kwargs.get("videoalign_model_path")
        )
        load_step = int(
            videoalign_cfg.get(
                "load_from_pretrained_step", self.reward_kwargs.get("videoalign_load_from_pretrained_step", -1)
            )
        )
        device = videoalign_cfg.get("device", self.reward_kwargs.get("videoalign_device"))
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype_name = str(videoalign_cfg.get("dtype", self.reward_kwargs.get("videoalign_dtype", "bfloat16"))).lower()
        self.videoalign_use_norm = bool(
            videoalign_cfg.get("use_norm", self.reward_kwargs.get("videoalign_use_norm", True))
        )

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        dtype = dtype_map.get(dtype_name)
        if dtype is None:
            raise ValueError(
                f"Unsupported videoalign dtype: {dtype_name}. "
                "Expected one of ['bfloat16', 'float16', 'float32']."
            )

        if not model_path:
            self.videoalign_init_error = "missing_model_path"
            return

        try:
            from fastvideo.models.videoalign.inference import VideoVLMRewardInference

            self.videoalign_inferencer = VideoVLMRewardInference(
                load_from_pretrained=model_path,
                load_from_pretrained_step=load_step,
                device=device,
                dtype=dtype,
            )
        except Exception as exc:
            self.videoalign_init_error = f"{type(exc).__name__}: {exc}"
            self.videoalign_inferencer = None

    @staticmethod
    def _normalize_non_tensor_list(value):
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if torch.is_tensor(value):
            return value.detach().cpu().tolist()
        if hasattr(value, "tolist"):
            converted = value.tolist()
            if isinstance(converted, (list, tuple)):
                return list(converted)
            return [converted]
        return [value]

    def _extract_prompt_texts(self, data: DataProto, batch_size: int):
        candidate_keys = (
            "full_prompts",
            "full_prompt",
            "text",
            "texts",
            "prompt",
            "prompts",
            "caption",
            "captions",
        )
        for key in candidate_keys:
            if key not in data.non_tensor_batch:
                continue
            raw = self._normalize_non_tensor_list(data.non_tensor_batch[key])
            if len(raw) != batch_size:
                continue
            return ["" if item is None else str(item) for item in raw]
        return [""] * batch_size

    def _extract_video_paths(self, data: DataProto, batch_size: int):
        candidate_keys = ("video_paths", "video_path", "videos_path", "videos_paths", "path", "paths")
        for key in candidate_keys:
            if key not in data.non_tensor_batch:
                continue
            raw = self._normalize_non_tensor_list(data.non_tensor_batch[key])
            if len(raw) != batch_size:
                continue
            out = []
            for item in raw:
                if isinstance(item, (list, tuple)) and len(item) > 0:
                    item = item[0]
                if item is None:
                    out = []
                    break
                out.append(str(item))
            if len(out) == batch_size:
                return out
        return None

    @staticmethod
    def _videoalign_fallback_scores(batch_size):
        return [{"VQ": -1.0, "MQ": -1.0, "TA": -1.0, "Overall": -3.0} for _ in range(batch_size)]

    @staticmethod
    def _normalize_videoalign_score(score):
        if not isinstance(score, dict):
            return {
                "VQ": -1.0,
                "MQ": -1.0,
                "TA": -1.0,
                "Overall": float(score),
            }
        vq = float(score.get("VQ", score.get("vq_reward", score.get("vq", -1.0))))
        mq = float(score.get("MQ", score.get("mq_reward", score.get("mq", -1.0))))
        ta = float(score.get("TA", score.get("ta_reward", score.get("ta", -1.0))))
        overall = score.get("Overall", score.get("overall", score.get("overall_reward", None)))
        if overall is None:
            overall = vq + mq + ta
        return {
            "VQ": vq,
            "MQ": mq,
            "TA": ta,
            "Overall": float(overall),
        }

    def _compute_videoalign_scores(self, data: DataProto, modality_key: str):
        batch_size = len(data)
        stats = {
            "videoalign_unavailable_count": 0.0,
            "videoalign_timeout_count": 0.0,
            "videoalign_exception_count": 0.0,
        }

        if modality_key != "videos":
            stats["videoalign_unavailable_count"] = float(batch_size)
            return self._videoalign_fallback_scores(batch_size), stats

        if self.videoalign_inferencer is None:
            stats["videoalign_unavailable_count"] = float(batch_size)
            return self._videoalign_fallback_scores(batch_size), stats

        prompts = self._extract_prompt_texts(data, batch_size)
        videos = [data.batch["videos"][i] for i in range(batch_size)]
        video_paths = self._extract_video_paths(data, batch_size)
        from_videos_error = None

        try:
            scores = self.videoalign_inferencer.reward_from_videos(videos, prompts, use_norm=self.videoalign_use_norm)
            if isinstance(scores, (list, tuple)) and len(scores) == batch_size:
                return list(scores), stats
            raise ValueError(f"videoalign.reward_from_videos returned invalid length: {len(scores)}")
        except TimeoutError as exc:
            from_videos_error = exc
        except Exception as exc:
            from_videos_error = exc

        if video_paths:
            try:
                scores = self.videoalign_inferencer.reward(video_paths, prompts, use_norm=self.videoalign_use_norm)
                if isinstance(scores, (list, tuple)) and len(scores) == batch_size:
                    return list(scores), stats
                raise ValueError(f"videoalign.reward returned invalid length: {len(scores)}")
            except TimeoutError:
                stats["videoalign_timeout_count"] = float(batch_size)
                return self._videoalign_fallback_scores(batch_size), stats
            except Exception:
                stats["videoalign_exception_count"] = float(batch_size)
                return self._videoalign_fallback_scores(batch_size), stats

        if isinstance(from_videos_error, TimeoutError):
            stats["videoalign_timeout_count"] = float(batch_size)
        else:
            stats["videoalign_exception_count"] = float(batch_size)
        return self._videoalign_fallback_scores(batch_size), stats

    def verify(self, data):
        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]

        prompt_len = prompt_ids.shape[-1]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        responses_str = []
        for i in range(len(data)):
            valid_len = valid_response_lengths[i]
            valid_response_ids = response_ids[i][:valid_len]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            responses_str.append(response_str)

        ground_truths = [item.non_tensor_batch["reward_model"].get("ground_truth", None) for item in data]
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        extras = data.non_tensor_batch.get("extra_info", [None] * len(data))

        scores = self.compute_score(
            data_sources=data_sources,
            solution_strs=responses_str,
            ground_truths=ground_truths,
            extra_infos=extras,
            **self.compute_score_kwargs,
        )

        return scores

    def __call__(self, data: DataProto, return_dict=False):
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        # Diffusion rollouts return images/videos instead of text responses
        if "responses" not in data.batch.keys():
            modality_key = None
            if "videos" in data.batch.keys():
                modality_key = "videos"
            elif "images" in data.batch.keys():
                modality_key = "images"
            if modality_key is None:
                raise KeyError('No valid modality found for reward computation (expected "videos" or "images").')

            example_tensor = next((v for v in data.batch.values() if torch.is_tensor(v)), None)
            device = example_tensor.device if example_tensor is not None else None

            reward_tensor = torch.zeros((len(data), 1), dtype=torch.float32, device=device)
            reward_extra_info = defaultdict(list)
            reward_inputs = [{modality_key: data.batch[modality_key][i]} for i in range(len(data))]
            backend_metrics = {
                "backend_videoalign": 0.0,
                "backend_builtin": 1.0,
                "videoalign_unavailable_count": 0.0,
                "videoalign_timeout_count": 0.0,
                "videoalign_exception_count": 0.0,
            }

            if self.reward_backend == "videoalign":
                backend_metrics["backend_videoalign"] = 1.0
                backend_metrics["backend_builtin"] = 0.0
                scores, videoalign_stats = self._compute_videoalign_scores(data, modality_key)
                backend_metrics.update(videoalign_stats)
            else:
                try:
                    scores = self.compute_score(reward_inputs=reward_inputs, **self.compute_score_kwargs)
                except TypeError:
                    scores = self.compute_score(reward_inputs, **self.compute_score_kwargs)

            rewards = []
            for i, score in enumerate(scores):
                if self.reward_backend == "videoalign":
                    normalized_score = self._normalize_videoalign_score(score)
                    reward = normalized_score["Overall"]
                    for key, value in normalized_score.items():
                        reward_extra_info[key].append(float(value))
                elif isinstance(score, dict):
                    reward = score.get("overall", score.get("score", score.get("overall_reward", None)))
                    for key, value in score.items():
                        reward_extra_info[key].append(value)
                else:
                    reward = score

                if reward is None:
                    raise ValueError(
                        "Reward function must return a scalar value under 'overall', 'score', or 'overall_reward'."
                    )

                reward = float(reward)
                reward_tensor[i, 0] = reward
                rewards.append(reward)

            data.batch["acc"] = torch.tensor(rewards, dtype=torch.float32, device=device)
            for key, value in backend_metrics.items():
                reward_extra_info[key].extend([float(value)] * len(data))

            if return_dict:
                return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
            else:
                return reward_tensor

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        prompt_ids = data.batch["prompts"]
        prompt_len = prompt_ids.shape[-1]
        attention_mask = data.batch["attention_mask"]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
        data_sources = data.non_tensor_batch[self.reward_fn_key]

        scores = self.verify(data)
        rewards = []
        already_printed = {}

        for i in range(len(data)):
            length = valid_response_lengths[i].item()
            score = scores[i]

            if isinstance(score, dict):
                reward = score.get("score", score.get("overall", score.get("overall_reward", None)))
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            if reward is None:
                raise ValueError("Reward function must return a scalar value under 'score', 'overall', or 'overall_reward'.")

            rewards.append(reward)
            reward_tensor[i, length - 1] = reward

            data_source = data_sources[i]
            if already_printed.get(data_source, 0) < self.num_examine:
                response_str = self.tokenizer.decode(data.batch["responses"][i][:length], skip_special_tokens=True)
                prompt_str = self.tokenizer.decode(data.batch["prompts"][i], skip_special_tokens=True)
                ground_truth = data[i].non_tensor_batch["reward_model"].get("ground_truth", None)
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", scores[i])
                already_printed[data_source] = already_printed.get(data_source, 0) + 1

        data.batch["acc"] = torch.tensor(rewards, dtype=torch.float32, device=prompt_ids.device)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor
