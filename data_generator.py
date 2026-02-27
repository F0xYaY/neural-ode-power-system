"""
并行生成电力系统暂态稳定性轨迹数据（4机简化 Swing 方程，8维状态）。

输出：
  - X_train.npy, X_val.npy, X_test.npy
  - shape: [samples, time_steps, state_dim] 其中 state_dim=8, time_steps = int(T/dt)+1

核心特性（低惯量/新能源等效）：
  - 每条样本随机选择 1~2 台发电机，将惯量 M 缩小到原来的 10%（IBR 接入等效）

运行示例（PowerShell）：
  python .\data_generator.py --n_samples 5000 --n_workers 20 --seed 123
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None

import multiprocessing as mp


@dataclass(frozen=True)
class SimConfig:
    n_machines: int = 4
    dt: float = 0.01
    t_final: float = 2.0
    t_fault_on: float = 0.5
    # 故障期间耦合缩放（越小越严重）
    fault_k_scale_min: float = 0.05
    fault_k_scale_max: float = 0.4
    fault_duration_min: float = 0.05
    fault_duration_max: float = 0.25
    # 初始扰动范围
    delta0_std: float = 0.15
    omega0_std: float = 0.30
    # 参数随机范围（基准值附近扰动）
    m_base: float = 5.0
    m_jitter: float = 0.25
    d_base: float = 1.0
    d_jitter: float = 0.30
    # 机械功率输入：
    # 为了让“误差状态系”的平衡点在 0（便于后续直接训练多维 Neural ODE/Lyapunov），
    # 这里默认取 Pm=0，使得 δ=0, ω=0 为平衡点（无外部扭矩）。
    # 随机性主要来自 M/D/K 及故障扰动与初始扰动。
    pm_base: float = 0.0
    pm_jitter: float = 0.0
    # 耦合强度
    k_base: float = 2.0
    k_jitter: float = 0.35
    # 数值稳定：限制角速度
    omega_clip: float = 8.0

    @property
    def t(self) -> np.ndarray:
        steps = int(round(self.t_final / self.dt))
        return np.linspace(0.0, self.t_final, steps + 1, dtype=np.float32)


def _make_symmetric_coupling(rng: np.random.Generator, n: int, k_base: float, k_jitter: float) -> np.ndarray:
    # 完全图耦合，K_ii=0，K_ij>0 且对称
    K = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            kij = k_base * (1.0 + k_jitter * rng.normal())
            kij = float(np.clip(kij, 0.2 * k_base, 3.0 * k_base))
            K[i, j] = kij
            K[j, i] = kij
    return K


def _swing_rhs(
    y: np.ndarray,
    t: float,
    M: np.ndarray,
    D: np.ndarray,
    Pm: np.ndarray,
    K_prefault: np.ndarray,
    fault_on: float,
    fault_off: float,
    fault_k_scale: float,
    omega_clip: float,
) -> np.ndarray:
    """
    4机 Swing 方程（8维）：
      dδ_i/dt = ω_i
      dω_i/dt = (Pm_i - sum_j K_ij * sin(δ_i-δ_j) - D_i*ω_i) / M_i
    """
    n = M.shape[0]
    delta = y[:n]
    omega = y[n:]
    omega = np.clip(omega, -omega_clip, omega_clip)

    if fault_on <= t <= fault_off:
        K = K_prefault * fault_k_scale
    else:
        K = K_prefault

    # electrical power Pe_i = sum_j K_ij sin(delta_i - delta_j)
    # 向量化计算：Pe = sum_j K_ij * sin(delta_i - delta_j)
    dmat = delta.reshape(-1, 1) - delta.reshape(1, -1)
    Pe = (K * np.sin(dmat)).sum(axis=1)

    d_delta = omega
    d_omega = (Pm - Pe - D * omega) / M
    return np.concatenate([d_delta, d_omega], axis=0).astype(np.float32)


def _rk4_integrate(
    y0: np.ndarray,
    t: np.ndarray,
    rhs_kwargs: dict,
) -> np.ndarray:
    """纯 numpy RK4，避免依赖 scipy。"""
    y = np.empty((t.shape[0], y0.shape[0]), dtype=np.float32)
    y[0] = y0.astype(np.float32)
    dt = float(t[1] - t[0])
    for k in range(t.shape[0] - 1):
        tk = float(t[k])
        yk = y[k]
        k1 = _swing_rhs(yk, tk, **rhs_kwargs)
        k2 = _swing_rhs(yk + 0.5 * dt * k1, tk + 0.5 * dt, **rhs_kwargs)
        k3 = _swing_rhs(yk + 0.5 * dt * k2, tk + 0.5 * dt, **rhs_kwargs)
        k4 = _swing_rhs(yk + dt * k3, tk + dt, **rhs_kwargs)
        y[k + 1] = yk + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # 轻微限制角度漂移（只为避免数值爆炸；不做 wrap 也可以）
        y[k + 1, : rhs_kwargs["M"].shape[0]] = np.clip(y[k + 1, : rhs_kwargs["M"].shape[0]], -math.pi * 6, math.pi * 6)
        if not np.isfinite(y[k + 1]).all():
            raise FloatingPointError("trajectory became non-finite")
    return y


def _sample_one(cfg: SimConfig, rng: np.random.Generator) -> np.ndarray:
    n = cfg.n_machines

    # 基准参数 + 随机扰动
    M = cfg.m_base * (1.0 + cfg.m_jitter * rng.normal(size=n)).astype(np.float32)
    M = np.clip(M, 0.5, 20.0).astype(np.float32)

    D = cfg.d_base * (1.0 + cfg.d_jitter * rng.normal(size=n)).astype(np.float32)
    D = np.clip(D, 0.05, 5.0).astype(np.float32)

    # 误差状态系：Pm=0 -> 原点为平衡点
    Pm = np.zeros((n,), dtype=np.float32)

    K = _make_symmetric_coupling(rng, n, cfg.k_base, cfg.k_jitter)

    # 低惯量机器：随机 1~2 台缩小到 10%
    n_low = int(rng.integers(1, 3))
    low_idx = rng.choice(n, size=n_low, replace=False)
    M[low_idx] = (0.1 * M[low_idx]).astype(np.float32)

    # 故障参数随机
    fault_duration = float(rng.uniform(cfg.fault_duration_min, cfg.fault_duration_max))
    fault_on = float(cfg.t_fault_on)
    fault_off = float(min(cfg.t_fault_on + fault_duration, cfg.t_final))
    fault_k_scale = float(rng.uniform(cfg.fault_k_scale_min, cfg.fault_k_scale_max))

    # 初始状态：小扰动
    delta0 = (cfg.delta0_std * rng.normal(size=n)).astype(np.float32)
    omega0 = (cfg.omega0_std * rng.normal(size=n)).astype(np.float32)
    omega0 = np.clip(omega0, -1.5, 1.5).astype(np.float32)
    y0 = np.concatenate([delta0, omega0], axis=0).astype(np.float32)

    rhs_kwargs = dict(
        M=M,
        D=D,
        Pm=Pm,
        K_prefault=K,
        fault_on=fault_on,
        fault_off=fault_off,
        fault_k_scale=fault_k_scale,
        omega_clip=cfg.omega_clip,
    )

    traj = _rk4_integrate(y0=y0, t=cfg.t, rhs_kwargs=rhs_kwargs)
    return traj


def _worker_generate(worker_id: int, n_samples: int, cfg: SimConfig, base_seed: int, max_tries: int = 50) -> np.ndarray:
    rng = np.random.default_rng(base_seed + 10007 * worker_id)
    out = np.empty((n_samples, cfg.t.shape[0], cfg.n_machines * 2), dtype=np.float32)
    filled = 0
    tries = 0
    while filled < n_samples:
        tries += 1
        if tries > max_tries * n_samples:
            raise RuntimeError(f"worker {worker_id} exceeded max tries; too many unstable samples")
        try:
            out[filled] = _sample_one(cfg, rng)
            filled += 1
        except Exception:
            # 失败样本直接丢弃重采样（避免 NaN/爆炸轨迹）
            continue
    return out


def _split_counts(n: int, train_ratio: float, val_ratio: float) -> Tuple[int, int, int]:
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val
    if n_test < 0:
        n_test = 0
        n_val = n - n_train
    return n_train, n_val, n_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--n_workers", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out_dir", type=str, default=".")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    args = parser.parse_args()

    cfg = SimConfig()
    os.makedirs(args.out_dir, exist_ok=True)

    n_train, n_val, n_test = _split_counts(args.n_samples, args.train_ratio, args.val_ratio)
    print(f"[INFO] total={args.n_samples} -> train={n_train}, val={n_val}, test={n_test}")
    print(f"[INFO] dt={cfg.dt}, T={cfg.t_final}, time_steps={cfg.t.shape[0]}, state_dim={cfg.n_machines*2}")
    print(f"[INFO] low-inertia: random 1~2 machines per sample (M*=0.1)")
    print(f"[INFO] workers={args.n_workers} (Windows 请确保从 __main__ 运行)")

    # 分配每个 worker 的样本数
    counts = [args.n_samples // args.n_workers] * args.n_workers
    for i in range(args.n_samples % args.n_workers):
        counts[i] += 1

    ctx = mp.get_context("spawn")  # Windows 安全
    with ctx.Pool(processes=args.n_workers) as pool:
        jobs = []
        for wid, c in enumerate(counts):
            if c <= 0:
                continue
            jobs.append(pool.apply_async(_worker_generate, (wid, c, cfg, args.seed)))

        it = jobs
        if tqdm is not None:
            it = tqdm(jobs, desc="Generating", total=len(jobs))

        chunks = []
        for j in it:
            chunks.append(j.get())

    X = np.concatenate(chunks, axis=0)
    # 随机打乱
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(X.shape[0])
    X = X[perm]

    X_train = X[:n_train]
    X_val = X[n_train : n_train + n_val]
    X_test = X[n_train + n_val :]

    np.save(os.path.join(args.out_dir, "X_train.npy"), X_train)
    np.save(os.path.join(args.out_dir, "X_val.npy"), X_val)
    np.save(os.path.join(args.out_dir, "X_test.npy"), X_test)

    print("[OK] saved:")
    print(f"  - {os.path.join(args.out_dir, 'X_train.npy')} {X_train.shape} {X_train.dtype}")
    print(f"  - {os.path.join(args.out_dir, 'X_val.npy')}   {X_val.shape} {X_val.dtype}")
    print(f"  - {os.path.join(args.out_dir, 'X_test.npy')}  {X_test.shape} {X_test.dtype}")


if __name__ == "__main__":
    main()

