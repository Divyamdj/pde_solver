# the_well/data/multi_collate.py
from __future__ import annotations
from typing import Any, Dict, List
import torch


def safe_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate that avoids "Trying to resize storage that is not resizable"
    by cloning tensors into fresh contiguous storage before stacking.
    """
    out: Dict[str, Any] = {}

    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]

        # stack tensors
        if torch.is_tensor(vals[0]):
            vals = [v.contiguous().clone() for v in vals]
            out[k] = torch.stack(vals, dim=0)
        else:
            # keep non-tensors as list
            out[k] = vals

    return out
