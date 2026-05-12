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
"""Flux-specific DanceGRPO trainer.

This module keeps class names aligned with the migrated Dance stack while
placing Flux-only dataloader and Case4 batch assembly in an isolated file.
"""

from typing import Optional

from torch.utils.data import Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role


class RayPPOTrainerDance(RayPPOTrainer):
    def _create_dance_latent_dataloader(self, mode_name: str) -> None:
        from fastvideo.dataset.latent_flux_rl_datasets import LatentDataset, latent_collate_function

        data_json_path = self.config.data.get("data_json_path", None)
        if data_json_path is None:
            raise ValueError(f"{mode_name}=true requires `data.data_json_path`")

        num_latent_t = int(self.config.data.get("t", 1))
        cfg_rate = float(self.config.data.get("cfg", 0.0))
        num_workers = int(self.config.data.get("dataloader_num_workers", 0))
        train_batch_size = int(self.config.data.get("train_batch_size", 1))

        self.train_dataset = LatentDataset(data_json_path, num_latent_t=num_latent_t, cfg_rate=cfg_rate)
        self.val_dataset = None
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=train_batch_size,
            num_workers=num_workers,
            drop_last=True,
            collate_fn=latent_collate_function,
            sampler=None,
        )
        self.val_dataloader = None

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        print(f"Size of train dataloader: {len(self.train_dataloader)}, validation disabled for {mode_name}")
        self._set_total_training_steps()

    def _create_dance_case1_dataloader(self) -> None:
        self._create_dance_latent_dataloader("dance_case1_mode")

    def _create_dance_case2_dataloader(self) -> None:
        self._create_dance_latent_dataloader("dance_case2_mode")

    def _create_dance_case3_dataloader(self) -> None:
        self._create_dance_latent_dataloader("dance_case3_mode")

    def _create_dance_case4_dataloader(self) -> None:
        self._create_dance_latent_dataloader("dance_case4_mode")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        if self._is_dance_case1_enabled():
            self._create_dance_case1_dataloader()
            return
        if self._is_dance_case2_enabled():
            self._create_dance_case2_dataloader()
            return
        if self._is_dance_case3_enabled():
            self._create_dance_case3_dataloader()
            return
        if self._is_dance_case4_enabled():
            self._create_dance_case4_dataloader()
            return
        super()._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def fit_dance_case4(self):
        from tqdm import tqdm
        from verl.utils.profiler import simple_timer

        if self.train_dataloader is None:
            raise RuntimeError("dance_case4_mode requires a Flux latent train_dataloader")

        self.global_steps = 0
        max_train_steps = int(self.config.trainer.max_train_steps)
        data_iterator = iter(self.train_dataloader)
        progress_bar = tqdm(total=max_train_steps, initial=self.global_steps, desc="Flux Dance Case4")
        self._init_dance_case4_step_timing()

        while self.global_steps < max_train_steps:
            timing_raw = {}
            with simple_timer("trainer.step.total", timing_raw):
                with simple_timer("trainer.make_prompt_batches", timing_raw):
                    try:
                        batch = next(data_iterator)
                    except StopIteration:
                        data_iterator = iter(self.train_dataloader)
                        batch = next(data_iterator)

                    if len(batch) != 4:
                        raise ValueError(
                            "Flux dance Case4 dataloader must return "
                            "(encoder_hidden_states, pooled_prompt_embeds, text_ids, caption)"
                        )
                    encoder_hidden_states, pooled_prompt_embeds, text_ids, caption = batch
                    new_batch = DataProto.from_single_dict(
                        {
                            "encoder_hidden_states": encoder_hidden_states,
                            "pooled_prompt_embeds": pooled_prompt_embeds,
                            "text_ids": text_ids,
                        },
                        meta_info={"caption": caption},
                    )

                with simple_timer("trainer.generate.task", timing_raw):
                    rollout_batch = self.actor_rollout_wg.generate_sequences(new_batch)

                with simple_timer("trainer.update.task", timing_raw):
                    actor_output = self.actor_rollout_wg.update_actor(rollout_batch)

            actor_metrics = actor_output.meta_info.get("metrics", {}) if actor_output is not None else {}
            current_step = self.global_steps + 1
            self._log_step_timing(timing_raw=timing_raw, step=current_step, epoch=0)
            self._record_dance_case4_step_timing(
                epoch=0,
                step=current_step,
                timing_raw=timing_raw,
                actor_metrics=actor_metrics,
                rollout_meta_info=rollout_batch.meta_info,
            )
            if actor_metrics:
                progress_bar.set_postfix({k: f"{v:.4f}" for k, v in actor_metrics.items() if isinstance(v, (int, float))})

            self.global_steps = current_step
            progress_bar.update(1)

        progress_bar.close()
        print(f"[flux_dance_case4_timing] markdown report: {self._dance_case4_step_timing_md_path}")
        print(f"[flux_dance_case4_timing] summary json: {self._dance_case4_step_timing_json_path}")
        return None
