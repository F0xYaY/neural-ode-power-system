from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchdiffeq import odeint

from model import DynamicsNetwork, ICNNLyapunovNet, get_lie_derivative
from npy_dataset import TrajectoryDataset, batch_to_ode_format


def _make_time_vector(time_steps: int, dt: float, device: torch.device) -> torch.Tensor:
    t_final = dt * (time_steps - 1)
    return torch.linspace(0.0, t_final, time_steps, device=device, dtype=torch.float32)


@torch.no_grad()
def eval_dynamics_mse(
    model: torch.nn.Module,
    loader: DataLoader,
    t: torch.Tensor,
    step_size: float,
    device: torch.device,
) -> float:
    model.eval()
    mses = []
    for batch_np in loader:
        batch = batch_np.to(device=device, dtype=torch.float32)
        y0, true_y = batch_to_ode_format(batch)  # [B,D], [T,B,D]
        pred_y = odeint(model, y0, t, method="rk4", options={"step_size": step_size})
        mses.append(F.mse_loss(pred_y, true_y).item())
    return float(np.mean(mses)) if mses else float("nan")


def train_dynamics(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    t: torch.Tensor,
    step_size: float,
    device: torch.device,
    epochs: int = 200,
    lr: float = 1e-3,
) -> Tuple[torch.nn.Module, Dict[str, list]]:
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist: Dict[str, list] = {"train_mse": [], "val_mse": []}

    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for batch_np in train_loader:
            batch = batch_np.to(device=device, dtype=torch.float32)
            y0, true_y = batch_to_ode_format(batch)

            opt.zero_grad(set_to_none=True)
            pred_y = odeint(model, y0, t, method="rk4", options={"step_size": step_size})
            loss = F.mse_loss(pred_y, true_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

        train_mse = float(np.mean(losses)) if losses else float("nan")
        val_mse = eval_dynamics_mse(model, val_loader, t, step_size, device)
        hist["train_mse"].append(train_mse)
        hist["val_mse"].append(val_mse)

        if ep % 10 == 0 or ep == 1:
            print(f"[Dynamics] ep={ep:04d} train_mse={train_mse:.6f} val_mse={val_mse:.6f}")

    return model, hist


def train_lyapunov_relaxed_exp(
    lyap: torch.nn.Module,
    dynamics: torch.nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    alpha: float = 0.5,
    epochs: int = 300,
    lr: float = 1e-3,
    states_per_batch: int = 4096,
) -> Tuple[torch.nn.Module, Dict[str, list]]:
    """
    Relaxed Exponential Stability loss:
      L = E[ relu(dV_dt + alpha * V) ]
    """
    lyap = lyap.to(device)
    dynamics = dynamics.to(device)
    dynamics.eval()  # 训练 Lyapunov 时固定动力学

    opt = torch.optim.Adam(lyap.parameters(), lr=lr)
    hist: Dict[str, list] = {"lyap_loss": []}

    for ep in range(1, epochs + 1):
        lyap.train()
        losses = []
        for batch_np in train_loader:
            batch = batch_np.to(device=device, dtype=torch.float32)  # [B,T,D]
            B, T, D = batch.shape
            flat = batch.reshape(B * T, D)

            # 从轨迹状态中抽样（比均匀采样更贴近数据分布）
            if flat.shape[0] > states_per_batch:
                idx = torch.randint(0, flat.shape[0], (states_per_batch,), device=device)
                x = flat[idx]
            else:
                x = flat

            opt.zero_grad(set_to_none=True)
            V = lyap(x)
            dV_dt = get_lie_derivative(lyap, dynamics, x)
            loss = torch.relu(dV_dt + alpha * V).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lyap.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())

        loss_ep = float(np.mean(losses)) if losses else float("nan")
        hist["lyap_loss"].append(loss_ep)
        if ep % 10 == 0 or ep == 1:
            print(f"[Lyapunov] ep={ep:04d} loss={loss_ep:.6f}")

    return lyap, hist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=str, default=os.path.join("..", "X_train.npy"))
    p.add_argument("--val", type=str, default=os.path.join("..", "X_val.npy"))
    p.add_argument("--test", type=str, default=os.path.join("..", "X_test.npy"))
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs_dyn", type=int, default=200)
    p.add_argument("--epochs_lyap", type=int, default=300)
    p.add_argument("--lr_dyn", type=float, default=1e-3)
    p.add_argument("--lr_lyap", type=float, default=1e-3)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save", type=str, default="trained_models_multidim.pt")
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"[INFO] device={device}")

    ds_train = TrajectoryDataset(args.train)
    ds_val = TrajectoryDataset(args.val)
    ds_test = TrajectoryDataset(args.test)

    # 推断维度
    sample0 = ds_train[0]
    time_steps = int(sample0.shape[0])
    input_dim = int(sample0.shape[1])
    print(f"[INFO] dataset: time_steps={time_steps}, input_dim={input_dim}")

    t = _make_time_vector(time_steps=time_steps, dt=args.dt, device=device)
    step_size = float(args.dt)

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False)

    # Dynamics：多维直接输入（不做 2D 变换/三角编码）
    dyn = DynamicsNetwork(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        use_spectral_norm=True,
        angle_encoding=False,
    )

    dyn, dyn_hist = train_dynamics(
        model=dyn,
        train_loader=train_loader,
        val_loader=val_loader,
        t=t,
        step_size=step_size,
        device=device,
        epochs=args.epochs_dyn,
        lr=args.lr_dyn,
    )

    test_mse = eval_dynamics_mse(dyn, test_loader, t, step_size, device)
    print(f"[OK] dynamics test_mse={test_mse:.6f}")

    # Lyapunov：ICNN 变体
    lyap = ICNNLyapunovNet(input_dim=input_dim, hidden_dims=(256, 256, 256, 256), epsilon=1e-3)
    lyap, lyap_hist = train_lyapunov_relaxed_exp(
        lyap=lyap,
        dynamics=dyn,
        train_loader=train_loader,
        device=device,
        alpha=args.alpha,
        epochs=args.epochs_lyap,
        lr=args.lr_lyap,
    )

    torch.save(
        {
            "input_dim": input_dim,
            "dt": args.dt,
            "dynamics_state_dict": dyn.state_dict(),
            "lyapunov_state_dict": lyap.state_dict(),
            "dyn_hist": dyn_hist,
            "lyap_hist": lyap_hist,
            "test_mse": test_mse,
            "alpha": args.alpha,
        },
        args.save,
    )
    print(f"[OK] saved {args.save}")


if __name__ == "__main__":
    main()

