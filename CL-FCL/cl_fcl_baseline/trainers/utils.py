from __future__ import annotations

import random
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def move_to_device(batch: Any, device: torch.device | str) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_to_device(item, device) for item in batch)
    if isinstance(batch, dict):
        return {key: move_to_device(value, device) for key, value in batch.items()}
    return batch


def detach_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}