from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class NpyTrajConfig:
    dt: float = 0.01
    device: str = "cpu"
    dtype: torch.dtype = torch.float32


class TrajectoryDataset(Dataset):
    """
    读取 shape [samples, time_steps, state_dim] 的 .npy 轨迹数据。
    """

    def __init__(self, npy_path: str):
        self.X = np.load(npy_path, mmap_mode="r")
        if self.X.ndim != 3:
            raise ValueError(f"Expected 3D array [samples,time_steps,state_dim], got {self.X.shape}")

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> np.ndarray:
        return np.asarray(self.X[idx], dtype=np.float32)


def batch_to_ode_format(batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    batch: [B, T, D] -> (y0: [B,D], true_y: [T,B,D])
    """
    if batch.ndim != 3:
        raise ValueError(f"batch must be [B,T,D], got {batch.shape}")
    y0 = batch[:, 0, :]
    true_y = batch.transpose(0, 1).contiguous()
    return y0, true_y

