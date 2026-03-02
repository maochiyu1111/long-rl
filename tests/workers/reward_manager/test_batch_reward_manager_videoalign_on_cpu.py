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

import numpy as np
import torch

from verl import DataProto
from verl.workers.reward_manager.batch import BatchRewardManager


def _build_video_batch(batch_size=2, with_paths=False):
    videos = torch.randn(batch_size, 4, 3, 8, 8, dtype=torch.float32)
    non_tensors = {
        "full_prompts": np.array([f"prompt-{i}" for i in range(batch_size)], dtype=object),
    }
    if with_paths:
        non_tensors["video_paths"] = np.array([f"/tmp/fake_{i}.mp4" for i in range(batch_size)], dtype=object)
    return DataProto.from_dict(tensors={"videos": videos}, non_tensors=non_tensors)


def test_builtin_backend_zero_regression():
    captured = {}

    def compute_score(reward_inputs, **kwargs):
        captured["reward_inputs"] = reward_inputs
        captured["kwargs"] = kwargs
        return [
            {"overall": 0.1, "score": 0.1},
            {"overall": 0.2, "score": 0.2},
        ]

    mgr = BatchRewardManager(tokenizer=None, num_examine=0, compute_score=compute_score, backend="builtin")
    batch = _build_video_batch(batch_size=2)
    result = mgr(batch, return_dict=True)

    assert "reward_inputs" in captured
    assert len(captured["reward_inputs"]) == 2
    assert torch.allclose(result["reward_tensor"], torch.tensor([[0.1], [0.2]], dtype=torch.float32), atol=1e-6)
    assert result["reward_extra_info"]["backend_builtin"] == [1.0, 1.0]
    assert result["reward_extra_info"]["backend_videoalign"] == [0.0, 0.0]


def test_videoalign_backend_return_and_in_memory_video_path(monkeypatch):
    calls = {"reward_from_videos": 0, "reward": 0}

    class FakeInferencer:
        def reward_from_videos(self, videos, prompts, use_norm=True):
            calls["reward_from_videos"] += 1
            assert len(videos) == 2
            assert prompts == ["prompt-0", "prompt-1"]
            return [
                {"VQ": 1.0, "MQ": 2.0, "TA": 3.0, "Overall": 6.0},
                {"VQ": 4.0, "MQ": 5.0, "TA": 6.0, "Overall": 15.0},
            ]

        def reward(self, video_paths, prompts, use_norm=True):
            calls["reward"] += 1
            raise AssertionError("reward() should not be used when reward_from_videos succeeds")

    def _fake_init(self):
        self.videoalign_inferencer = FakeInferencer()
        self.videoalign_use_norm = True
        self.videoalign_init_error = None

    monkeypatch.setattr(BatchRewardManager, "_init_videoalign_backend", _fake_init)

    mgr = BatchRewardManager(tokenizer=None, num_examine=0, compute_score=None, backend="videoalign")
    batch = _build_video_batch(batch_size=2)
    result = mgr(batch, return_dict=True)

    assert calls["reward_from_videos"] == 1
    assert calls["reward"] == 0
    assert torch.allclose(result["reward_tensor"], torch.tensor([[6.0], [15.0]], dtype=torch.float32), atol=1e-6)
    assert result["reward_extra_info"]["VQ"] == [1.0, 4.0]
    assert result["reward_extra_info"]["MQ"] == [2.0, 5.0]
    assert result["reward_extra_info"]["TA"] == [3.0, 6.0]
    assert result["reward_extra_info"]["Overall"] == [6.0, 15.0]
    assert result["reward_extra_info"]["backend_videoalign"] == [1.0, 1.0]
    assert result["reward_extra_info"]["videoalign_exception_count"] == [0.0, 0.0]


def test_videoalign_backend_fallback_to_reward_with_video_paths(monkeypatch):
    calls = {"reward_from_videos": 0, "reward": 0}

    class FakeInferencer:
        def reward_from_videos(self, videos, prompts, use_norm=True):
            calls["reward_from_videos"] += 1
            raise RuntimeError("tensor path failed")

        def reward(self, video_paths, prompts, use_norm=True):
            calls["reward"] += 1
            assert video_paths == ["/tmp/fake_0.mp4", "/tmp/fake_1.mp4"]
            return [
                {"VQ": 0.5, "MQ": 0.5, "TA": 0.0, "Overall": 1.0},
                {"VQ": 1.5, "MQ": 1.5, "TA": 0.0, "Overall": 3.0},
            ]

    def _fake_init(self):
        self.videoalign_inferencer = FakeInferencer()
        self.videoalign_use_norm = True
        self.videoalign_init_error = None

    monkeypatch.setattr(BatchRewardManager, "_init_videoalign_backend", _fake_init)

    mgr = BatchRewardManager(tokenizer=None, num_examine=0, compute_score=None, backend="videoalign")
    batch = _build_video_batch(batch_size=2, with_paths=True)
    result = mgr(batch, return_dict=True)

    assert calls["reward_from_videos"] == 1
    assert calls["reward"] == 1
    assert torch.allclose(result["reward_tensor"], torch.tensor([[1.0], [3.0]], dtype=torch.float32), atol=1e-6)
    assert result["reward_extra_info"]["videoalign_exception_count"] == [0.0, 0.0]
    assert result["reward_extra_info"]["videoalign_timeout_count"] == [0.0, 0.0]


def test_videoalign_backend_exception_fallback_and_counts(monkeypatch):
    class FakeInferencer:
        def reward_from_videos(self, videos, prompts, use_norm=True):
            raise RuntimeError("backend exploded")

        def reward(self, video_paths, prompts, use_norm=True):
            raise AssertionError("reward() should not be called without video paths")

    def _fake_init(self):
        self.videoalign_inferencer = FakeInferencer()
        self.videoalign_use_norm = True
        self.videoalign_init_error = None

    monkeypatch.setattr(BatchRewardManager, "_init_videoalign_backend", _fake_init)

    mgr = BatchRewardManager(tokenizer=None, num_examine=0, compute_score=None, backend="videoalign")
    batch = _build_video_batch(batch_size=2, with_paths=False)
    result = mgr(batch, return_dict=True)

    assert torch.allclose(result["reward_tensor"], torch.tensor([[-3.0], [-3.0]], dtype=torch.float32), atol=1e-6)
    assert result["reward_extra_info"]["VQ"] == [-1.0, -1.0]
    assert result["reward_extra_info"]["MQ"] == [-1.0, -1.0]
    assert result["reward_extra_info"]["videoalign_exception_count"] == [2.0, 2.0]
    assert result["reward_extra_info"]["videoalign_timeout_count"] == [0.0, 0.0]
    assert result["reward_extra_info"]["videoalign_unavailable_count"] == [0.0, 0.0]


def test_videoalign_backend_timeout_fallback_and_counts(monkeypatch):
    class FakeInferencer:
        def reward_from_videos(self, videos, prompts, use_norm=True):
            raise TimeoutError("backend timeout")

        def reward(self, video_paths, prompts, use_norm=True):
            raise AssertionError("reward() should not be called without video paths")

    def _fake_init(self):
        self.videoalign_inferencer = FakeInferencer()
        self.videoalign_use_norm = True
        self.videoalign_init_error = None

    monkeypatch.setattr(BatchRewardManager, "_init_videoalign_backend", _fake_init)

    mgr = BatchRewardManager(tokenizer=None, num_examine=0, compute_score=None, backend="videoalign")
    batch = _build_video_batch(batch_size=2, with_paths=False)
    result = mgr(batch, return_dict=True)

    assert torch.allclose(result["reward_tensor"], torch.tensor([[-3.0], [-3.0]], dtype=torch.float32), atol=1e-6)
    assert result["reward_extra_info"]["videoalign_timeout_count"] == [2.0, 2.0]
    assert result["reward_extra_info"]["videoalign_exception_count"] == [0.0, 0.0]


def test_videoalign_backend_partial_failure_isolated_per_sample(monkeypatch):
    class FakeInferencer:
        def reward_from_videos(self, videos, prompts, use_norm=True):
            # Simulate batch API failure first, then per-sample fallback behavior.
            if len(videos) > 1:
                raise RuntimeError("batch inference failed")
            if prompts[0] == "prompt-0":
                return [{"VQ": 1.0, "MQ": 2.0, "TA": 3.0, "Overall": 6.0}]
            raise RuntimeError("single-sample inference failed")

        def reward(self, video_paths, prompts, use_norm=True):
            raise AssertionError("reward() should not be called when video paths are unavailable")

    def _fake_init(self):
        self.videoalign_inferencer = FakeInferencer()
        self.videoalign_use_norm = True
        self.videoalign_init_error = None

    monkeypatch.setattr(BatchRewardManager, "_init_videoalign_backend", _fake_init)

    mgr = BatchRewardManager(tokenizer=None, num_examine=0, compute_score=None, backend="videoalign")
    batch = _build_video_batch(batch_size=2, with_paths=False)
    result = mgr(batch, return_dict=True)

    assert torch.allclose(result["reward_tensor"], torch.tensor([[6.0], [-3.0]], dtype=torch.float32), atol=1e-6)
    assert result["reward_extra_info"]["VQ"] == [1.0, -1.0]
    assert result["reward_extra_info"]["MQ"] == [2.0, -1.0]
    assert result["reward_extra_info"]["TA"] == [3.0, -1.0]
    assert result["reward_extra_info"]["Overall"] == [6.0, -3.0]
    assert result["reward_extra_info"]["videoalign_exception_count"] == [1.0, 1.0]
    assert result["reward_extra_info"]["videoalign_timeout_count"] == [0.0, 0.0]
