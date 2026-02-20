# the_well/data/multi_datamodule.py

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from the_well.data.multi_dataset import MultiWellConfig, MultiWellDataset
from the_well.data.normalization import ZScoreNormalization
from the_well.data.pde_text_embedder import PDETextEmbedder
from the_well.data.multi_collate import safe_collate
from the_well.data.datamodule import AbstractDataModule


@dataclass
class MultiWellDataModule(AbstractDataModule):
    well_base_path: str
    train_datasets: List[str]
    val_datasets: List[str]
    test_datasets: List[str]

    batch_size: int = 16
    data_workers: int = 4

    n_steps_input: int = 4
    n_steps_output: int = 1
    min_dt_stride: int = 1
    max_dt_stride: int = 1

    use_normalization: bool = True
    target_hw: Tuple[int, int] = (128, 384)

    embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedder_cache: str = "pde_text_embeddings.pt"

    def setup(self):
        embedder = PDETextEmbedder(
            model_name=self.embedder_model,
            cache_path=self.embedder_cache,
            device="cpu",
        )

        self.train_dataset = MultiWellDataset(
            MultiWellConfig(
                dataset_names=self.train_datasets,
                well_base_path=self.well_base_path,
                split="train",
                batch_input_steps=self.n_steps_input,
                batch_output_steps=self.n_steps_output,
                use_normalization=self.use_normalization,
                min_dt_stride=self.min_dt_stride,
                max_dt_stride=self.max_dt_stride,
                target_hw=self.target_hw,
            ),
            embedder=embedder,
            normalization_type=ZScoreNormalization,
        )

        # Use the training dataset's global field schema for val and test
        # This ensures consistent channel ordering across train/val/test splits
        global_field_names = self.train_dataset.global_field_names
        global_constant_scalar_names = self.train_dataset.global_constant_scalar_names

        self.val_dataset = MultiWellDataset(
            MultiWellConfig(
                dataset_names=self.val_datasets,
                well_base_path=self.well_base_path,
                split="valid",
                batch_input_steps=self.n_steps_input,
                batch_output_steps=self.n_steps_output,
                use_normalization=self.use_normalization,
                min_dt_stride=self.min_dt_stride,
                max_dt_stride=self.max_dt_stride,
                target_hw=self.target_hw,
            ),
            embedder=embedder,
            normalization_type=ZScoreNormalization,
            global_field_names=global_field_names,
            global_constant_scalar_names=global_constant_scalar_names,
        )

        self.test_dataset = MultiWellDataset(
            MultiWellConfig(
                dataset_names=self.test_datasets,
                well_base_path=self.well_base_path,
                split="test",
                batch_input_steps=self.n_steps_input,
                batch_output_steps=self.n_steps_output,
                use_normalization=self.use_normalization,
                min_dt_stride=self.min_dt_stride,
                max_dt_stride=self.max_dt_stride,
                target_hw=self.target_hw,
            ),
            embedder=embedder,
            normalization_type=ZScoreNormalization,
            global_field_names=global_field_names,
            global_constant_scalar_names=global_constant_scalar_names,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.data_workers,
            pin_memory=True,
            collate_fn=safe_collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.data_workers,
            pin_memory=True,
            collate_fn=safe_collate,
        )

    def rollout_val_dataloader(self):
        # same as val for now
        return self.val_dataloader()

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.data_workers,
            pin_memory=True,
            collate_fn=safe_collate,
        )

    def rollout_test_dataloader(self):
        return self.test_dataloader()
