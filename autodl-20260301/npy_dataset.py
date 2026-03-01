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
    def __init__(self, npy_path: str, in_memory: bool = True):
        # 【核心优化】：对于 4090，如果数据量不到几十GB，强烈建议 in_memory=True
        # 彻底解除 SSD 的 I/O 读写瓶颈，让显卡满载运行。
        if in_memory:
            print(f"[{npy_path}] 正在将数据全量加载至内存中，以最大化读取速度...")
            self.X = np.load(npy_path) # 直接进 RAM
        else:
            print(f"[{npy_path}] 启用 mmap 模式读取 (适用于超大规模数据集)...")
            self.X = np.load(npy_path, mmap_mode="r")

        if self.X.ndim != 3:
            raise ValueError(f"Expected 3D array [samples,time_steps,state_dim], got {self.X.shape}")

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> np.ndarray:
        # DataLoader 会把 numpy -> torch
        # copy 一份避免 PyTorch 警告与潜在未定义行为
        return np.asarray(self.X[idx], dtype=np.float32).copy()

def batch_to_ode_format(batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    batch: [B, T, D] -> (y0: [B,D], true_y: [T,B,D])
    """
    if batch.ndim != 3:
        raise ValueError(f"batch must be [B,T,D], got {batch.shape}")
        
    y0 = batch[:, 0, :]
    # 使用 .contiguous() 优化内存布局，加速后续计算
    true_y = batch.transpose(0, 1).contiguous()
    
    return y0, true_y


