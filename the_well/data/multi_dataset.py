# the_well/data/multi_dataset.py

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from the_well.data.datasets import WellDataset
from the_well.data.pde_text_embedder import PDETextEmbedder


def _resize_fields(x: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    """
    x: (T, H, W, C) or (H, W, C)
    returns same rank with resized H,W.
    """
    Ht, Wt = target_hw

    if x.ndim == 4:
        # (T,H,W,C) -> (T,C,H,W)
        x = x.permute(0, 3, 1, 2)
        x = F.interpolate(x, size=(Ht, Wt), mode="bilinear", align_corners=False)
        x = x.permute(0, 2, 3, 1)
        return x

    if x.ndim == 3:
        # (H,W,C) -> (1,C,H,W)
        x = x.permute(2, 0, 1).unsqueeze(0)
        x = F.interpolate(x, size=(Ht, Wt), mode="bilinear", align_corners=False)
        x = x.squeeze(0).permute(1, 2, 0)
        return x

    raise ValueError(f"Unsupported tensor rank for resize: {x.shape}")


def _remap_channels_to_global(
    x: torch.Tensor, local_to_global: torch.Tensor, target_c: int
) -> torch.Tensor:
    """
    x: (..., C_local)
    local_to_global: (C_local,) mapping local channel i -> global channel idx
    returns: (..., target_c) where channels are placed into global space
    """
    if x.shape[-1] != local_to_global.numel():
        raise ValueError(
            f"Channel remap mismatch: x has C={x.shape[-1]} "
            f"but local_to_global has {local_to_global.numel()}"
        )

    out = torch.zeros(*x.shape[:-1], target_c, dtype=x.dtype, device=x.device)
    out[..., local_to_global] = x
    return out


def _remap_vector_to_global(
    v: torch.Tensor, local_to_global: torch.Tensor, target_dim: int
) -> torch.Tensor:
    """
    v: (D_local,)
    local_to_global: (D_local,) mapping local index -> global index
    returns: (target_dim,)
    """
    if v.ndim != 1:
        raise ValueError(f"Expected 1D vector, got {v.shape}")

    if v.numel() != local_to_global.numel():
        raise ValueError(
            f"Vector remap mismatch: v has dim={v.numel()} "
            f"but local_to_global has {local_to_global.numel()}"
        )

    out = torch.zeros(target_dim, dtype=v.dtype, device=v.device)
    out[local_to_global] = v
    return out


@dataclass
class MultiWellConfig:
    dataset_names: List[str]
    well_base_path: str
    split: str
    batch_input_steps: int
    batch_output_steps: int
    use_normalization: bool
    min_dt_stride: int
    max_dt_stride: int
    target_hw: Tuple[int, int] = (128, 384)
    seed: int = 0


class MultiWellDataset(Dataset):
    """
    Wraps multiple WellDataset objects and returns:
      - input_fields/output_fields remapped into global channel schema
      - constant_scalars remapped into global scalar schema
      - pde_embedding (D,)
      - dataset_id
    """

    def __init__(
        self,
        cfg: MultiWellConfig,
        embedder: PDETextEmbedder,
        normalization_type=None,
        cache_embeddings: bool = True,
        global_field_names: Optional[List[str]] = None,
        global_constant_scalar_names: Optional[List[str]] = None,
    ):
        super().__init__()
        random.seed(cfg.seed)

        self.cfg = cfg
        self.embedder = embedder

        # -----------------------------
        # Load datasets
        # -----------------------------
        self.datasets: List[WellDataset] = []
        for name in cfg.dataset_names:
            dset = WellDataset(
                well_base_path=cfg.well_base_path,
                well_dataset_name=name,
                well_split_name=cfg.split,
                use_normalization=cfg.use_normalization,
                normalization_type=normalization_type,
                min_dt_stride=cfg.min_dt_stride,
                max_dt_stride=cfg.max_dt_stride,
                n_steps_input=cfg.batch_input_steps,
                n_steps_output=cfg.batch_output_steps,
                return_grid=False,
                boundary_return_type=None,
            )
            self.datasets.append(dset)

        if len(self.datasets) == 0:
            raise ValueError("MultiWellDataset got empty dataset_names")

        # Use provided global field names, or build from current datasets
        if global_field_names is not None:
            self.global_field_names = global_field_names
        else:
            # Build GLOBAL FIELD schema from current datasets
            self.global_field_names: List[str] = []
            for dset in self.datasets:
                meta = dset.metadata
                field_names_dict = getattr(meta, "field_names", {})
                for order in sorted(field_names_dict.keys()):
                    for fn in field_names_dict[order]:
                        if fn not in self.global_field_names:
                            self.global_field_names.append(fn)

        self.c_max = len(self.global_field_names)

        # Use provided global constant scalar names, or build from current datasets
        if global_constant_scalar_names is not None:
            self.global_constant_scalar_names = global_constant_scalar_names
        else:
            # Build GLOBAL CONSTANT SCALAR schema
            self.global_constant_scalar_names: List[str] = []
            for dset in self.datasets:
                meta = dset.metadata
                names = getattr(meta, "constant_scalar_names", None)
                if names is None:
                    names = getattr(meta, "constant_scalars_names", None)
                if names is None:
                    names = []

                for sn in list(names):
                    if sn not in self.global_constant_scalar_names:
                        self.global_constant_scalar_names.append(sn)

        self.s_max = len(self.global_constant_scalar_names)

        # local channel -> global channel mapping per dataset
        self.field_channel_maps: List[torch.Tensor] = []
        for dset in self.datasets:
            meta = dset.metadata
            field_names_dict = getattr(meta, "field_names", {})
            local_names = []
            # Collect all field names in order of tensor order
            for order in sorted(field_names_dict.keys()):
                local_names.extend(field_names_dict[order])
            
            local_to_global = [self.global_field_names.index(fn) for fn in local_names]
            self.field_channel_maps.append(torch.tensor(local_to_global, dtype=torch.long))

        # Build scalar mapping per dataset
        self.scalar_maps: List[Optional[torch.Tensor]] = []
        for dset in self.datasets:
            meta = dset.metadata
            names = getattr(meta, "constant_scalar_names", None)
            if names is None:
                names = getattr(meta, "constant_scalars_names", None)
            if names is None:
                names = []

            if len(names) == 0:
                self.scalar_maps.append(None)
                continue

            local_to_global = [self.global_constant_scalar_names.index(sn) for sn in list(names)]
            self.scalar_maps.append(torch.tensor(local_to_global, dtype=torch.long))

        # -----------------------------
        # Offsets for global indexing
        # -----------------------------
        self.offsets = [0]
        for dset in self.datasets:
            self.offsets.append(self.offsets[-1] + len(dset))

        # -----------------------------
        # Dataset-level PDE embeddings
        # -----------------------------
        self.dataset_embeddings: List[torch.Tensor] = []
        for dset in self.datasets:
            meta = dset.metadata
            text = self.embedder.get_text_for_dataset(
                dataset_name=meta.dataset_name,
                field_names=dset.core_field_names,  # IMPORTANT: use local names
                bc_types=meta.boundary_condition_types,
                spatial_res=meta.spatial_resolution,
                n_spatial_dims=meta.n_spatial_dims,
            )
            emb = self.embedder.embed(meta.dataset_name, text)
            self.dataset_embeddings.append(emb)

        self.embedding_dim = int(self.dataset_embeddings[0].shape[0])

        # -----------------------------
        # Expose correct metadata for downstream plotting/metrics
        # -----------------------------
        # We reuse metadata object from dataset[0] but overwrite the relevant parts.
        self.metadata = self.datasets[0].metadata

        # These are used by plotting/metrics - use the global field names list
        if hasattr(self.metadata, "core_field_names"):
            self.metadata.core_field_names = list(self.global_field_names)

        # field_names metadata should preserve dict structure for compatibility with utils functions
        # We put all merged field names under a single key (order 0) since we don't track
        # individual tensor orders in the merged dataset
        if hasattr(self.metadata, "field_names"):
            self.metadata.field_names = {0: list(self.global_field_names)}

        # Similarly for constant fields
        if hasattr(self.metadata, "constant_field_names"):
            self.metadata.constant_field_names = {0: list(self.global_constant_scalar_names)}

        # Scalar names if present
        if hasattr(self.metadata, "constant_scalar_names"):
            self.metadata.constant_scalar_names = list(self.global_constant_scalar_names)

        self.use_normalization = False
        self.norm = None

    def __len__(self):
        return self.offsets[-1]

    def _locate(self, idx: int):
        ds_id = 0
        while ds_id + 1 < len(self.offsets) and idx >= self.offsets[ds_id + 1]:
            ds_id += 1
        local_idx = idx - self.offsets[ds_id]
        return ds_id, local_idx

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ds_id, local_idx = self._locate(idx)
        dset = self.datasets[ds_id]
        sample = dset[local_idx]

        # Remove things we don't want
        sample.pop("boundary_conditions", None)
        sample.pop("space_grid", None)
        sample.pop("input_time_grid", None)
        sample.pop("output_time_grid", None)

        # -----------------------------
        # Resize fields
        # -----------------------------
        sample["input_fields"] = _resize_fields(sample["input_fields"], self.cfg.target_hw)
        sample["output_fields"] = _resize_fields(sample["output_fields"], self.cfg.target_hw)

        # -----------------------------
        # Remap fields into global schema
        # -----------------------------
        field_map = self.field_channel_maps[ds_id].to(sample["input_fields"].device)

        sample["input_fields"] = _remap_channels_to_global(
            sample["input_fields"], field_map, self.c_max
        )
        sample["output_fields"] = _remap_channels_to_global(
            sample["output_fields"], field_map, self.c_max
        )

        # -----------------------------
        # Remap constant scalars into global schema
        # -----------------------------
        # Some datasets might not have them, or you might have removed them earlier.
        const_scalars = sample.get("constant_scalars", None)
        if const_scalars is None:
            const_scalars = torch.zeros(0, dtype=torch.float32)

        if const_scalars.numel() == 0 or self.s_max == 0:
            sample["constant_scalars"] = torch.zeros(self.s_max, dtype=torch.float32)
        else:
            scalar_map = self.scalar_maps[ds_id]
            if scalar_map is None:
                # dataset had scalars but we couldn't map them by name
                # safest: drop into first dims (still stable but semantically weak)
                out = torch.zeros(self.s_max, dtype=const_scalars.dtype)
                k = min(self.s_max, const_scalars.numel())
                out[:k] = const_scalars[:k]
                sample["constant_scalars"] = out
            else:
                scalar_map = scalar_map.to(const_scalars.device)
                sample["constant_scalars"] = _remap_vector_to_global(
                    const_scalars, scalar_map, self.s_max
                )

        # -----------------------------
        # Add conditioning
        # -----------------------------
        sample["pde_embedding"] = self.dataset_embeddings[ds_id].clone()  # (D,)
        sample["dataset_id"] = torch.tensor(ds_id, dtype=torch.long)

        # -----------------------------
        # Ensure stable keys for trainer/model
        # -----------------------------
        if "constant_fields" not in sample:
            sample["constant_fields"] = torch.zeros(*self.cfg.target_hw, 0)

        if "input_scalars" not in sample:
            sample["input_scalars"] = torch.zeros(self.cfg.batch_input_steps, 0)

        if "output_scalars" not in sample:
            sample["output_scalars"] = torch.zeros(self.cfg.batch_output_steps, 0)

        return sample
