import torch
import matplotlib.pyplot as plt
from torchdiffeq import odeint
from model import PowerSystemPhysics


def plot_lyapunov_contours(v_net, range_val=2.0, delta_s=0.52):
    """
    可视化 Lyapunov 函数的等高线图（稳定域）
    
    修正坐标轴标签，使其科学严谨
    
    Args:
        v_net: Lyapunov 网络
        range_val: 绘图范围
        delta_s: 稳定平衡点的 delta 值（用于标注）
    """
    # 检测设备并确保 grid 在正确的设备上
    device = next(v_net.parameters()).device
    
    x = torch.linspace(-range_val, range_val, 100, device=device)
    y = torch.linspace(-range_val, range_val, 100, device=device)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    grid = torch.stack([X, Y], dim=-1).reshape(-1, 2)
    
    with torch.no_grad():
        V_val = v_net(grid).reshape(100, 100).cpu()
    
    # X 和 Y 也需要移到 CPU 用于绘图
    X_cpu = X.cpu() if X.device.type == 'cuda' else X
    Y_cpu = Y.cpu() if Y.device.type == 'cuda' else Y
        
    plt.figure(figsize=(8, 8))
    # 绘制等高线，颜色越深代表能量越低
    cp = plt.contourf(X_cpu, Y_cpu, V_val, levels=20, cmap='viridis')
    plt.colorbar(cp, label='Lyapunov Function V(x)')
    
    # 修正标签：使用准确的物理量名称
    plt.xlabel("Rotor Angle Deviation (rad)", fontsize=12)
    plt.ylabel("Frequency Deviation (p.u.)", fontsize=12)
    
    # 添加标注说明坐标是相对于稳定平衡点的
    plt.title(f"Learned Lyapunov Function V(x)\nCoordinates centered at Stable Equilibrium Point (δ_s ≈ {delta_s:.2f} rad)", 
              fontsize=12)
    
    # 验证：叠加真实的物理稳定点
    plt.plot(0, 0, 'rx', markersize=12, label='Stable Equilibrium', linewidth=2)
    plt.legend(fontsize=10)
    
    # 确保等比例，避免轨迹形状失真
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_phase_space_trajectories(true_y, pred_y, t, sample_indices=None, delta_s=0.52):
    """
    绘制相空间轨迹对比图（Phase Space）
    
    修正坐标轴标签，使其科学严谨
    
    Args:
        true_y: 真实轨迹，shape (time_steps, batch_size, 2)
        pred_y: 预测轨迹，shape (time_steps, batch_size, 2)
        t: 时间点
        sample_indices: 要可视化的样本索引列表，如果为 None 则选择前4个
        delta_s: 稳定平衡点的 delta 值
    """
    if sample_indices is None:
        sample_indices = [0, 1, 2, 3]
    
    num_samples = len(sample_indices)
    cols = 2
    rows = (num_samples + 1) // 2
    
    plt.figure(figsize=(12, 6 * rows))
    
    for idx, sample_idx in enumerate(sample_indices):
        plt.subplot(rows, cols, idx + 1)
        
        # 真实轨迹（移到 CPU）
        plt.plot(true_y[:, sample_idx, 0].detach().cpu(), 
                true_y[:, sample_idx, 1].detach().cpu(), 
                'k-', label='True', linewidth=2, alpha=0.8)
        
        # 预测轨迹（移到 CPU）
        if pred_y.dim() == 3:
            plt.plot(pred_y[:, sample_idx, 0].detach().cpu(), 
                    pred_y[:, sample_idx, 1].detach().cpu(), 
                    'r--', label='Neural ODE', linewidth=2, alpha=0.8)
        else:
            # 如果 pred_y 是单个轨迹
            plt.plot(pred_y[:, 0].detach().cpu(), 
                    pred_y[:, 1].detach().cpu(), 
                    'r--', label='Neural ODE', linewidth=2, alpha=0.8)
        
        # 修正标签
        plt.xlabel("Rotor Angle Deviation (rad)", fontsize=11)
        plt.ylabel("Frequency Deviation (p.u.)", fontsize=11)
        plt.title(f'Sample {sample_idx} - Phase Plane Trajectory', fontsize=11)
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)
        
        # 确保等比例
        plt.axis('equal')
    
    plt.tight_layout()
    plt.suptitle(f"Phase Space Analysis\nCoordinates centered at Stable Equilibrium Point (δ_s ≈ {delta_s:.2f} rad)", 
                 fontsize=13, y=1.00)
    plt.show()


def plot_time_domain_response(t, true_y, pred_y, sample_indices=None, delta_s=0.52):
    """
    绘制时域波形图
    
    Args:
        t: 时间点
        true_y: 真实轨迹
        pred_y: 预测轨迹
        sample_indices: 要可视化的样本索引列表
        delta_s: 稳定平衡点的 delta 值
    """
    if sample_indices is None:
        sample_indices = [0, 1, 2, 3]
    
    num_samples = len(sample_indices)
    cols = 2
    rows = num_samples
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    # 确保 t 也在 CPU 上
    t_cpu = t.cpu() if isinstance(t, torch.Tensor) and t.device.type == 'cuda' else t
    
    for idx, sample_idx in enumerate(sample_indices):
        # Delta 时间序列（移到 CPU）
        ax = axes[idx, 0]
        ax.plot(t_cpu, true_y[:, sample_idx, 0].detach().cpu(), 'k-', label='True Delta', alpha=0.7, linewidth=2)
        ax.plot(t_cpu, pred_y[:, sample_idx, 0].detach().cpu(), 'r--', label='Neural ODE Delta', linewidth=2)
        ax.set_xlabel("Time (s)", fontsize=11)
        ax.set_ylabel("Rotor Angle Deviation (rad)", fontsize=11)
        ax.set_title(f"Sample {sample_idx} - Rotor Angle", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Omega 时间序列（移到 CPU）
        ax = axes[idx, 1]
        ax.plot(t_cpu, true_y[:, sample_idx, 1].detach().cpu(), 'k-', label='True Omega', alpha=0.7, linewidth=2)
        ax.plot(t_cpu, pred_y[:, sample_idx, 1].detach().cpu(), 'r--', label='Neural ODE Omega', linewidth=2)
        ax.set_xlabel("Time (s)", fontsize=11)
        ax.set_ylabel("Frequency Deviation (p.u.)", fontsize=11)
        ax.set_title(f"Sample {sample_idx} - Frequency", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle(f"Time Domain Response\nCoordinates centered at Stable Equilibrium Point (δ_s ≈ {delta_s:.2f} rad)", 
                 fontsize=13, y=1.00)
    plt.show()


def plot_vector_field(model, range_val=2.0, n_grid=20, delta_s=0.52):
    """
    绘制向量场（Vector Field）
    
    Args:
        model: Neural ODE 模型
        range_val: 绘图范围
        n_grid: 网格点数
        delta_s: 稳定平衡点的 delta 值
    """
    # 检测设备并确保 grid 在正确的设备上
    device = next(model.parameters()).device
    
    y_mesh, x_mesh = torch.meshgrid(
        torch.linspace(-range_val, range_val, n_grid, device=device),
        torch.linspace(-range_val, range_val, n_grid, device=device),
        indexing='xy'
    )
    grid = torch.stack([x_mesh, y_mesh], dim=-1).reshape(-1, 2)
    
    with torch.no_grad():
        v = model(0, grid)
    
    # 移到 CPU 用于绘图
    grid_cpu = grid.cpu()
    v_cpu = v.cpu()
    
    plt.figure(figsize=(8, 8))
    plt.quiver(grid_cpu[:, 0], grid_cpu[:, 1], v_cpu[:, 0], v_cpu[:, 1], 
               color='blue', alpha=0.4, scale=20, width=0.003)
    
    plt.xlabel("Rotor Angle Deviation (rad)", fontsize=12)
    plt.ylabel("Frequency Deviation (p.u.)", fontsize=12)
    plt.title(f"Learned Vector Field\nCoordinates centered at Stable Equilibrium Point (δ_s ≈ {delta_s:.2f} rad)", 
              fontsize=12)
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
